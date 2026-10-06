from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.attention_sublayer import (
    AttentionSublayer,
)
from src.transformer.feed_forward import (
    FeedForwardNetwork,
    FeedForwardResult,
)
from src.transformer.layer_norm import (
    LayerNormalization,
)
from src.transformer.trainable_multi_head import (
    TrainableMultiHeadAttention,
    TrainableMultiHeadResult,
)
from src.transformer.residual import add_residual


@dataclass(frozen=True)
class TransformerDecoderBlockResult:
    input: Matrix
    attention: MultiHeadResult
    after_attention_sublayer: Matrix
    feed_forward: FeedForwardResult
    feed_forward_output: Matrix
    output: Matrix


class TransformerDecoderBlock:
    def __init__(
        self,
        model_dimension: int,
        head_dimension: int,
        feed_forward_dimension: int,
        seed: int = 42,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError(
                "model_dimension must be positive."
            )

        self._attention = TrainableMultiHeadAttention(
            model_dimension=model_dimension,
            head_dimension=head_dimension,
            seed=seed,
            causal=True,
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

    @property
    def attention_module(
        self,
    ) -> SyntheticMultiHeadAttention:
        return self._attention

    @property
    def feed_forward_weights_1(self) -> Matrix:
        return self._feed_forward.weights_1

    @property
    def feed_forward_weights_2(self) -> Matrix:
        return self._feed_forward.weights_2
    def forward(
        self,
        inputs: Matrix,
    ) -> TransformerDecoderBlockResult:
        attention_result = self._attention.forward(
            inputs
        )

        attention_sublayer_result = (
            self._attention_sublayer.forward(
                inputs,
                attention_result.output,
            )
        )

        feed_forward_result = (
            self._feed_forward.forward(
                attention_sublayer_result.normalized_output
            )
        )

        output_residual = add_residual(
            attention_sublayer_result.normalized_output,
            feed_forward_result.output,
        )

        output = self._output_normalization.forward(
            output_residual
        )

        return TransformerDecoderBlockResult(
            input=inputs,
            attention=attention_result,
            after_attention_sublayer=(
                attention_sublayer_result.normalized_output
            ),
            feed_forward=feed_forward_result,
            feed_forward_output=(
                feed_forward_result.output
            ),
            output=output,
        )

