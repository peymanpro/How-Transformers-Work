from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.feed_forward import FeedForwardResult
from src.transformer.feed_forward_backward import (
    FeedForwardBackward,
    FeedForwardGradients,
)
from src.transformer.layer_norm_backward import (
    LayerNormalizationBackward,
)
from src.transformer.residual_backward import (
    backward_residual,
    combine_residual_gradients,
)


@dataclass(frozen=True)
class FeedForwardSublayerGradients:
    input: Matrix
    feed_forward: FeedForwardGradients


class FeedForwardSublayerBackward:
    def backward(
        self,
        inputs: Matrix,
        feed_forward_weights_1: Matrix,
        feed_forward_weights_2: Matrix,
        feed_forward_result: FeedForwardResult,
        output_gradient: Matrix,
    ) -> FeedForwardSublayerGradients:
        normalized_gradient = (
            LayerNormalizationBackward().backward(
                inputs=self._residual_output(
                    inputs,
                    feed_forward_result.output,
                ),
                output_gradient=output_gradient,
            )
        )

        direct_gradient, sublayer_gradient = (
            backward_residual(
                normalized_gradient
            )
        )

        feed_forward_gradients = (
            FeedForwardBackward().backward(
                inputs=inputs,
                weights_1=feed_forward_weights_1,
                weights_2=feed_forward_weights_2,
                forward_result=feed_forward_result,
                output_gradient=sublayer_gradient,
            )
        )

        combined_input_gradient = (
            combine_residual_gradients(
                direct_gradient,
                feed_forward_gradients.inputs,
            )
        )

        return FeedForwardSublayerGradients(
            input=combined_input_gradient,
            feed_forward=feed_forward_gradients,
        )

    @staticmethod
    def _residual_output(
        inputs: Matrix,
        feed_forward_output: Matrix,
    ) -> Matrix:
        return inputs.add(
            feed_forward_output
        )
