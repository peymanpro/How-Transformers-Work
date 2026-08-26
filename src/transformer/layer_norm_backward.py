from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix


class LayerNormalizationBackward:
    def __init__(
        self,
        epsilon: float = 1e-5,
    ) -> None:
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive and finite."
            )

        self._epsilon = epsilon

    def backward(
        self,
        inputs: Matrix,
        output_gradient: Matrix,
    ) -> Matrix:
        if inputs.shape != output_gradient.shape:
            raise ValueError(
                "Input and output gradient shapes must match."
            )

        data = inputs.data

        mean = np.mean(
            data,
            axis=1,
            keepdims=True,
        )

        centered = data - mean

        variance = np.mean(
            centered * centered,
            axis=1,
            keepdims=True,
        )

        standard_deviation = np.sqrt(
            variance + self._epsilon
        )

        normalized = (
            centered
            / standard_deviation
        )

        dimension = inputs.columns

        gradient_sum = np.sum(
            output_gradient.data,
            axis=1,
            keepdims=True,
        )

        normalized_gradient_sum = np.sum(
            output_gradient.data
            * normalized,
            axis=1,
            keepdims=True,
        )

        gradient = (
            (
                output_gradient.data * dimension
                - gradient_sum
                - normalized
                * normalized_gradient_sum
            )
            / (
                dimension
                * standard_deviation
            )
        )

        return Matrix(gradient)
