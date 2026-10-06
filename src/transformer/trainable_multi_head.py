from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix


@dataclass(frozen=True)
class TrainableMultiHeadResult:
    queries: tuple[Matrix, ...]
    keys: tuple[Matrix, ...]
    values: tuple[Matrix, ...]
    weights: tuple[Matrix, ...]
    head_outputs: tuple[Matrix, ...]
    concatenated: Matrix
    output: Matrix


class TrainableMultiHeadAttention:
    """Multi-head self-attention with trainable Q/K/V projections."""

    def __init__(
        self,
        model_dimension: int,
        head_dimension: int,
        seed: int = 42,
        causal: bool = False,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError("model_dimension must be positive.")

        if head_dimension <= 0:
            raise ValueError("head_dimension must be positive.")

        if model_dimension % head_dimension != 0:
            raise ValueError(
                "model_dimension must be divisible by head_dimension."
            )

        self._model_dimension = model_dimension
        self._head_dimension = head_dimension
        self._number_of_heads = model_dimension // head_dimension
        self._causal = causal

        self._query_weights = self._initialize_projection_weights(
            seed + 101
        )
        self._key_weights = self._initialize_projection_weights(
            seed + 211
        )
        self._value_weights = self._initialize_projection_weights(
            seed + 307
        )

        rng = np.random.default_rng(seed + 401)
        scale = 1.0 / np.sqrt(model_dimension)
        self._output_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(model_dimension, model_dimension),
            )
        )

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

    @property
    def head_dimension(self) -> int:
        return self._head_dimension

    @property
    def number_of_heads(self) -> int:
        return self._number_of_heads

    @property
    def causal(self) -> bool:
        return self._causal

    @property
    def query_weights(self) -> tuple[Matrix, ...]:
        return tuple(Matrix(weight.data) for weight in self._query_weights)

    @property
    def key_weights(self) -> tuple[Matrix, ...]:
        return tuple(Matrix(weight.data) for weight in self._key_weights)

    @property
    def value_weights(self) -> tuple[Matrix, ...]:
        return tuple(Matrix(weight.data) for weight in self._value_weights)

    @property
    def output_weights(self) -> Matrix:
        return Matrix(self._output_weights.data)

    def forward(
        self,
        inputs: Matrix,
    ) -> TrainableMultiHeadResult:
        if inputs.columns != self._model_dimension:
            raise ValueError(
                "Input dimension does not match model dimension."
            )

        queries: list[Matrix] = []
        keys: list[Matrix] = []
        values: list[Matrix] = []
        weights: list[Matrix] = []
        head_outputs: list[Matrix] = []

        scale = np.sqrt(self._head_dimension)

        for query_weight, key_weight, value_weight in zip(
            self._query_weights,
            self._key_weights,
            self._value_weights,
        ):
            query = inputs.multiply(query_weight)
            key = inputs.multiply(key_weight)
            value = inputs.multiply(value_weight)

            scores = (query.data @ key.data.T) / scale

            if self._causal:
                scores = self._apply_causal_mask(scores)

            head_weights = self._softmax_rows(scores)
            output = Matrix(head_weights @ value.data)

            queries.append(query)
            keys.append(key)
            values.append(value)
            weights.append(Matrix(head_weights))
            head_outputs.append(output)

        concatenated = Matrix(
            np.concatenate(
                [output.data for output in head_outputs],
                axis=1,
            )
        )

        output = concatenated.multiply(self._output_weights)

        return TrainableMultiHeadResult(
            queries=tuple(queries),
            keys=tuple(keys),
            values=tuple(values),
            weights=tuple(weights),
            head_outputs=tuple(head_outputs),
            concatenated=concatenated,
            output=output,
        )

    def apply_gradients(
        self,
        query_gradients: tuple[Matrix, ...],
        key_gradients: tuple[Matrix, ...],
        value_gradients: tuple[Matrix, ...],
        output_gradient: Matrix,
        learning_rate: float,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        self._validate_gradient_collection(
            query_gradients,
            self._query_weights,
            "Query",
        )
        self._validate_gradient_collection(
            key_gradients,
            self._key_weights,
            "Key",
        )
        self._validate_gradient_collection(
            value_gradients,
            self._value_weights,
            "Value",
        )

        if output_gradient.shape != self._output_weights.shape:
            raise ValueError(
                "Output projection gradient shape does not match."
            )

        self._query_weights = self._updated_weights(
            self._query_weights,
            query_gradients,
            learning_rate,
        )
        self._key_weights = self._updated_weights(
            self._key_weights,
            key_gradients,
            learning_rate,
        )
        self._value_weights = self._updated_weights(
            self._value_weights,
            value_gradients,
            learning_rate,
        )
        self._output_weights = Matrix(
            self._output_weights.data
            - learning_rate * output_gradient.data
        )

    def _initialize_projection_weights(
        self,
        seed: int,
    ) -> tuple[Matrix, ...]:
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(self._model_dimension)

        return tuple(
            Matrix(
                rng.normal(
                    0.0,
                    scale,
                    size=(
                        self._model_dimension,
                        self._head_dimension,
                    ),
                )
            )
            for _ in range(self._number_of_heads)
        )

    @staticmethod
    def _validate_gradient_collection(
        gradients: tuple[Matrix, ...],
        weights: tuple[Matrix, ...],
        label: str,
    ) -> None:
        if len(gradients) != len(weights):
            raise ValueError(
                f"{label} gradient count must match head count."
            )

        for gradient, weight in zip(gradients, weights):
            if gradient.shape != weight.shape:
                raise ValueError(
                    f"{label} gradient shape does not match."
                )

    @staticmethod
    def _updated_weights(
        weights: tuple[Matrix, ...],
        gradients: tuple[Matrix, ...],
        learning_rate: float,
    ) -> tuple[Matrix, ...]:
        return tuple(
            Matrix(
                weight.data
                - learning_rate * gradient.data
            )
            for weight, gradient in zip(weights, gradients)
        )

    @staticmethod
    def _apply_causal_mask(
        scores: np.ndarray,
    ) -> np.ndarray:
        masked = scores.copy()
        length = masked.shape[0]
        future = np.triu(
            np.ones((length, length), dtype=bool),
            k=1,
        )
        masked[future] = -np.inf
        return masked

    @staticmethod
    def _softmax_rows(
        scores: np.ndarray,
    ) -> np.ndarray:
        shifted = scores - np.max(
            scores,
            axis=1,
            keepdims=True,
        )
        exponentials = np.exp(shifted)
        normalized = exponentials / np.sum(
            exponentials,
            axis=1,
            keepdims=True,
        )
        return np.asarray(normalized, dtype=np.float64)
