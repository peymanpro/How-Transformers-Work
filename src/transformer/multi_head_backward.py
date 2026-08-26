from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix
from src.transformer.multi_head import (
    MultiHeadResult,
    SyntheticMultiHeadAttention,
)


@dataclass(frozen=True)
class MultiHeadGradients:
    inputs: Matrix
    output_weights: Matrix


class SyntheticMultiHeadBackward:
    def backward(
        self,
        attention: SyntheticMultiHeadAttention,
        forward_result: MultiHeadResult,
        inputs: Matrix,
        output_gradient: Matrix,
    ) -> MultiHeadGradients:
        if (
            forward_result.output.shape
            != output_gradient.shape
        ):
            raise ValueError(
                "Output gradient must match attention output."
            )

        if inputs.columns != attention.model_dimension:
            raise ValueError(
                "Input dimension does not match attention dimension."
            )

        concatenated = forward_result.concatenated

        d_output_weights = Matrix(
            concatenated.data.T
            @ output_gradient.data
        )

        d_concatenated = (
            output_gradient.data
            @ attention.output_weights.data.T
        )

        d_inputs = np.zeros_like(
            inputs.data
        )

        head_dimension = (
            attention.model_dimension
            // attention.number_of_heads
        )

        for head_index, weights in enumerate(
            forward_result.head_weights
        ):
            start = (
                head_index
                * head_dimension
            )

            end = start + head_dimension

            head_gradient = (
                d_concatenated[
                    :,
                    start:end,
                ]
            )

            d_head_input = (
                weights.data.T
                @ head_gradient
            )

            d_inputs[
                :,
                :head_dimension,
            ] += d_head_input

        return MultiHeadGradients(
            inputs=Matrix(d_inputs),
            output_weights=d_output_weights,
        )
