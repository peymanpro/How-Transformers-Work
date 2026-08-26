from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix


@dataclass(frozen=True)
class FeedForwardResult:
    hidden: Matrix
    output: Matrix


class FeedForwardNetwork:
    def __init__(
        self,
        model_dimension: int,
        hidden_dimension: int,
        seed: int = 42,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError(
                "model_dimension must be positive."
            )

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be positive."
            )

        self._model_dimension = model_dimension
        self._hidden_dimension = hidden_dimension

        rng = np.random.default_rng(seed)

        first_scale = 1.0 / np.sqrt(model_dimension)
        second_scale = 1.0 / np.sqrt(hidden_dimension)

        self._weights_1 = Matrix(
            rng.normal(
                0.0,
                first_scale,
                size=(
                    model_dimension,
                    hidden_dimension,
                ),
            )
        )

        self._weights_2 = Matrix(
            rng.normal(
                0.0,
                second_scale,
                size=(
                    hidden_dimension,
                    model_dimension,
                ),
            )
        )

        self._bias_1 = Matrix(
            np.zeros(
                (1, hidden_dimension),
                dtype=np.float64,
            )
        )

        self._bias_2 = Matrix(
            np.zeros(
                (1, model_dimension),
                dtype=np.float64,
            )
        )

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

    @property
    def hidden_dimension(self) -> int:
        return self._hidden_dimension

    @property
    def weights_1(self) -> Matrix:
        return Matrix(self._weights_1.data)

    @property
    def weights_2(self) -> Matrix:
        return Matrix(self._weights_2.data)

    @property
    def bias_1(self) -> Matrix:
        return Matrix(self._bias_1.data)

    @property
    def bias_2(self) -> Matrix:
        return Matrix(self._bias_2.data)
    def apply_gradients(
        self,
        weights_1_gradient: Matrix,
        bias_1_gradient: Matrix,
        weights_2_gradient: Matrix,
        bias_2_gradient: Matrix,
        learning_rate: float,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        if weights_1_gradient.shape != self._weights_1.shape:
            raise ValueError(
                "weights_1 gradient shape does not match."
            )

        if bias_1_gradient.shape != self._bias_1.shape:
            raise ValueError(
                "bias_1 gradient shape does not match."
            )

        if weights_2_gradient.shape != self._weights_2.shape:
            raise ValueError(
                "weights_2 gradient shape does not match."
            )

        if bias_2_gradient.shape != self._bias_2.shape:
            raise ValueError(
                "bias_2 gradient shape does not match."
            )

        self._weights_1 = Matrix(
            self._weights_1.data
            - learning_rate * weights_1_gradient.data
        )

        self._bias_1 = Matrix(
            self._bias_1.data
            - learning_rate * bias_1_gradient.data
        )

        self._weights_2 = Matrix(
            self._weights_2.data
            - learning_rate * weights_2_gradient.data
        )

        self._bias_2 = Matrix(
            self._bias_2.data
            - learning_rate * bias_2_gradient.data
        )
    def forward(
        self,
        inputs: Matrix,
    ) -> FeedForwardResult:
        if inputs.columns != self._model_dimension:
            raise ValueError(
                "Input dimension does not match model dimension."
            )

        hidden_linear = self._add_bias(
            inputs.multiply(
                self._weights_1
            ),
            self._bias_1,
        )

        hidden = Matrix(
            np.maximum(
                hidden_linear.data,
                0.0,
            )
        )

        output_linear = self._add_bias(
            hidden.multiply(
                self._weights_2
            ),
            self._bias_2,
        )

        return FeedForwardResult(
            hidden=hidden,
            output=output_linear,
        )

    @staticmethod
    def _add_bias(
        values: Matrix,
        bias: Matrix,
    ) -> Matrix:
        if bias.rows != 1:
            raise ValueError(
                "Bias must contain exactly one row."
            )

        if bias.columns != values.columns:
            raise ValueError(
                "Bias dimension must match matrix columns."
            )

        repeated_bias = np.broadcast_to(
            bias.data,
            values.shape,
        )

        return Matrix(
            values.data + repeated_bias
        )


