from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix


class LayerNormalization:
    def __init__(
        self,
        dimension: int,
        epsilon: float = 1e-5,
    ) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension must be positive."
            )

        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive and finite."
            )

        self._dimension = dimension
        self._epsilon = epsilon

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def forward(
        self,
        inputs: Matrix,
    ) -> Matrix:
        if inputs.columns != self._dimension:
            raise ValueError(
                "Input dimension does not match LayerNorm dimension."
            )

        data = inputs.data

        mean = np.mean(
            data,
            axis=1,
            keepdims=True,
        )

        variance = np.mean(
            (data - mean) ** 2,
            axis=1,
            keepdims=True,
        )

        normalized = (
            (data - mean)
            / np.sqrt(
                variance + self._epsilon
            )
        )

        return Matrix(normalized)
