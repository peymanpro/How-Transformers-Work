from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.attention_sublayer_backward import (
    AttentionSublayerBackward,
    AttentionSublayerGradients,
)
from src.transformer.encoder_block import (
    TransformerBlockResult,
)
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
from src.transformer.trainable_multi_head import (
    TrainableMultiHeadAttention,
)


@dataclass(frozen=True)
class TransformerBlockGradients:
    input: Matrix
    attention: AttentionSublayerGradients
    feed_forward: FeedForwardGradients


class TransformerEncoderBlockBackward:
    def backward(
        self,
        block: TrainableMultiHeadAttention,
        forward_result: TransformerBlockResult,
        output_gradient: Matrix,
        feed_forward_weights_1: Matrix,
        feed_forward_weights_2: Matrix,
    ) -> TransformerBlockGradients:
        if (
            forward_result.output.shape
            != output_gradient.shape
        ):
            raise ValueError(
                "Output gradient must match block output."
            )

        # Final LayerNorm:
        #
        # output = LayerNorm(
        #     after_attention_sublayer + ffn_output
        # )
        final_residual = (
            forward_result.after_attention_sublayer.add(
                forward_result.feed_forward_output
            )
        )

        final_norm_gradient = (
            LayerNormalizationBackward().backward(
                inputs=final_residual,
                output_gradient=output_gradient,
            )
        )

        # Residual:
        #
        # final_residual =
        #     after_attention_sublayer + ffn_output
        direct_attention_gradient, ffn_output_gradient = (
            backward_residual(
                final_norm_gradient
            )
        )

        # FFN only.
        #
        # The final LayerNorm has already been handled above,
        # so do NOT use FeedForwardSublayerBackward here.
        feed_forward_gradients = (
            FeedForwardBackward().backward(
                inputs=(
                    forward_result.after_attention_sublayer
                ),
                weights_1=feed_forward_weights_1,
                weights_2=feed_forward_weights_2,
                forward_result=(
                    forward_result.feed_forward
                ),
                output_gradient=ffn_output_gradient,
            )
        )

        # Gradient arriving at the output of the
        # attention sublayer.
        attention_sublayer_gradient = (
            combine_residual_gradients(
                direct_attention_gradient,
                feed_forward_gradients.inputs,
            )
        )

        attention_gradients = (
            AttentionSublayerBackward().backward(
                attention=block,
                inputs=forward_result.input,
                forward_result=forward_result.attention,
                output_gradient=attention_sublayer_gradient,
            )
        )

        return TransformerBlockGradients(
            input=attention_gradients.input,
            attention=attention_gradients,
            feed_forward=feed_forward_gradients,
        )
