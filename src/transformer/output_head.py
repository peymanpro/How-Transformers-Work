from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix


@dataclass(frozen=True)
class TokenLogits:
    values: Matrix


class VocabularyProjection:
    def __init__(
        self,
        model_dimension: int,
        vocabulary_size: int,
        seed: int = 42,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError(
                "model_dimension must be positive."
            )

        if vocabulary_size <= 0:
            raise ValueError(
                "vocabulary_size must be positive."
            )

        self._model_dimension = model_dimension
        self._vocabulary_size = vocabulary_size

        rng = np.random.default_rng(seed)

        scale = 1.0 / np.sqrt(model_dimension)

        self._weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(
                    model_dimension,
                    vocabulary_size,
                ),
            )
        )

        self._bias = Matrix(
            np.zeros(
                (1, vocabulary_size),
                dtype=np.float64,
            )
        )

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    @property
    def weights(self) -> Matrix:
        return Matrix(
            self._weights.data
        )
    @property
    def bias(self) -> Matrix:
        return Matrix(
            self._bias.data
        )
    def replace_weights(
        self,
        weights: Matrix,
    ) -> None:
        if weights.shape != self._weights.shape:
            raise ValueError(
                "Replacement weights must match current weight shape."
            )

        self._weights = Matrix(
            weights.data
        )
    def forward(
        self,
        hidden_states: Matrix,
    ) -> TokenLogits:
        if hidden_states.columns != self._model_dimension:
            raise ValueError(
                "Hidden-state dimension does not match model dimension."
            )

        logits = (
            hidden_states.multiply(
                self._weights
            )
        )

        repeated_bias = np.broadcast_to(
            self._bias.data,
            logits.shape,
        )

        return TokenLogits(
            values=Matrix(
                logits.data + repeated_bias
            )
        )



