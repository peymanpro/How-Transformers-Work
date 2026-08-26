from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix
from src.transformer.feed_forward import FeedForwardResult


@dataclass(frozen=True)
class FeedForwardGradients:
    inputs: Matrix
    weights_1: Matrix
    bias_1: Matrix
    weights_2: Matrix
    bias_2: Matrix


class FeedForwardBackward:
    def backward(
        self,
        inputs: Matrix,
        weights_1: Matrix,
        weights_2: Matrix,
        forward_result: FeedForwardResult,
        output_gradient: Matrix,
    ) -> FeedForwardGradients:
        if (
            forward_result.output.shape
            != output_gradient.shape
        ):
            raise ValueError(
                "Output gradient shape must match FFN output."
            )

        if inputs.columns != weights_1.rows:
            raise ValueError(
                "Input dimension must match first-layer weights."
            )

        if weights_1.columns != weights_2.rows:
            raise ValueError(
                "Hidden dimension must match second-layer weights."
            )

        if weights_2.columns != inputs.columns:
            raise ValueError(
                "Second-layer output dimension must match input dimension."
            )

        d_weights_2 = Matrix(
            forward_result.hidden.data.T
            @ output_gradient.data
        )

        d_bias_2 = Matrix(
            np.sum(
                output_gradient.data,
                axis=0,
                keepdims=True,
            )
        )

        d_hidden = Matrix(
            output_gradient.data
            @ weights_2.data.T
        )

        # Reconstruct the pre-activation state from
        # the forward input and first-layer parameters.
        hidden_pre_activation = (
            inputs.data @ weights_1.data
        )

        d_hidden_pre_activation = Matrix(
            d_hidden.data
            * (
                hidden_pre_activation > 0.0
            )
        )

        d_weights_1 = Matrix(
            inputs.data.T
            @ d_hidden_pre_activation.data
        )

        d_bias_1 = Matrix(
            np.sum(
                d_hidden_pre_activation.data,
                axis=0,
                keepdims=True,
            )
        )

        d_inputs = Matrix(
            d_hidden_pre_activation.data
            @ weights_1.data.T
        )

        return FeedForwardGradients(
            inputs=d_inputs,
            weights_1=d_weights_1,
            bias_1=d_bias_1,
            weights_2=d_weights_2,
            bias_2=d_bias_2,
        )
