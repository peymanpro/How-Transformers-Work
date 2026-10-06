from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.attention_sublayer_backward import (
    AttentionSublayerBackward,
    AttentionSublayerGradients,
)
from src.transformer.decoder_block import (
    TransformerDecoderBlockResult,
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
class TransformerDecoderBlockGradients:
    input: Matrix
    attention: AttentionSublayerGradients
    feed_forward: FeedForwardGradients


class TransformerDecoderBlockBackward:
    def backward(
        self,
        attention_module: TrainableMultiHeadAttention,
        forward_result: TransformerDecoderBlockResult,
        output_gradient: Matrix,
        feed_forward_weights_1: Matrix,
        feed_forward_weights_2: Matrix,
    ) -> TransformerDecoderBlockGradients:
        if (
            forward_result.output.shape
            != output_gradient.shape
        ):
            raise ValueError(
                "Output gradient must match decoder output."
            )

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

        direct_attention_gradient, ffn_output_gradient = (
            backward_residual(
                final_norm_gradient
            )
        )

        feed_forward_gradients = (
            FeedForwardBackward().backward(
                inputs=forward_result.after_attention_sublayer,
                weights_1=feed_forward_weights_1,
                weights_2=feed_forward_weights_2,
                forward_result=forward_result.feed_forward,
                output_gradient=ffn_output_gradient,
            )
        )

        attention_sublayer_gradient = (
            combine_residual_gradients(
                direct_attention_gradient,
                feed_forward_gradients.inputs,
            )
        )

        attention_gradients = (
            AttentionSublayerBackward().backward(
                attention=attention_module,
                inputs=forward_result.input,
                forward_result=forward_result.attention,
                output_gradient=attention_sublayer_gradient,
            )
        )

        return TransformerDecoderBlockGradients(
            input=attention_gradients.input,
            attention=attention_gradients,
            feed_forward=feed_forward_gradients,
        )

