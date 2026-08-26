from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.attention_sublayer import (
    AttentionSublayer,
)
from src.transformer.feed_forward import (
    FeedForwardNetwork,
)
from src.transformer.layer_norm import LayerNormalization
from src.transformer.multi_head import (
    MultiHeadResult,
    SyntheticMultiHeadAttention,
)
from src.transformer.residual import add_residual


@dataclass(frozen=True)
class TransformerBlockResult:
    input: Matrix
    attention: MultiHeadResult
    after_attention_sublayer: Matrix
    feed_forward_output: Matrix
    output: Matrix


class TransformerEncoderBlock:
    def __init__(
        self,
        model_dimension: int,
        head_dimension: int,
        head_focuses: list[int],
        feed_forward_dimension: int,
        seed: int = 42,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError(
                "model_dimension must be positive."
            )

        if feed_forward_dimension <= 0:
            raise ValueError(
                "feed_forward_dimension must be positive."
            )

        self._attention = SyntheticMultiHeadAttention(
            model_dimension=model_dimension,
            head_dimension=head_dimension,
            focuses=head_focuses,
            seed=seed,
        )

        self._attention_sublayer = AttentionSublayer(
            dimension=model_dimension,
        )

        self._feed_forward = FeedForwardNetwork(
            model_dimension=model_dimension,
            hidden_dimension=feed_forward_dimension,
            seed=seed,
        )

        self._output_normalization = LayerNormalization(
            dimension=model_dimension,
        )

    def forward(
        self,
        inputs: Matrix,
    ) -> TransformerBlockResult:
        attention_result = self._attention.forward(
            inputs
        )

        attention_sublayer_result = (
            self._attention_sublayer.forward(
                inputs,
                attention_result.output,
            )
        )

        feed_forward_result = self._feed_forward.forward(
            attention_sublayer_result.normalized_output
        )

        output_residual = add_residual(
            attention_sublayer_result.normalized_output,
            feed_forward_result.output,
        )

        output = self._output_normalization.forward(
            output_residual
        )

        return TransformerBlockResult(
            input=inputs,
            attention=attention_result,
            after_attention_sublayer=(
                attention_sublayer_result.normalized_output
            ),
            feed_forward_output=(
                feed_forward_result.output
            ),
            output=output,
        )
