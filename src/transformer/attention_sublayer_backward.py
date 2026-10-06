from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.layer_norm_backward import (
    LayerNormalizationBackward,
)
from src.transformer.multi_head_backward import (
    TrainableMultiHeadBackward,
    TrainableMultiHeadGradients,
)
from src.transformer.trainable_multi_head import (
    TrainableMultiHeadAttention,
    TrainableMultiHeadResult,
)
from src.transformer.residual_backward import (
    backward_residual,
    combine_residual_gradients,
)


@dataclass(frozen=True)
class AttentionSublayerGradients:
    input: Matrix
    attention: TrainableMultiHeadGradients


class AttentionSublayerBackward:
    def backward(
        self,
        attention: TrainableMultiHeadAttention,
        inputs: Matrix,
        forward_result: TrainableMultiHeadResult,
        output_gradient: Matrix,
    ) -> AttentionSublayerGradients:
        residual_output = inputs.add(
            forward_result.output
        )

        normalized_gradient = (
            LayerNormalizationBackward().backward(
                inputs=residual_output,
                output_gradient=output_gradient,
            )
        )

        direct_gradient, attention_gradient = (
            backward_residual(
                normalized_gradient
            )
        )

        attention_gradients = (
            TrainableMultiHeadBackward().backward(
                attention=attention,
                forward_result=forward_result,
                inputs=inputs,
                output_gradient=attention_gradient,
            )
        )

        input_gradient = (
            combine_residual_gradients(
                direct_gradient,
                attention_gradients.inputs,
            )
        )

        return AttentionSublayerGradients(
            input=input_gradient,
            attention=attention_gradients,
        )
