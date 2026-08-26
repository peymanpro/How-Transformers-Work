from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.decoder_block_backward import (
    TransformerDecoderBlockBackward,
    TransformerDecoderBlockGradients,
)
from src.transformer.embedding_backward import (
    TokenEmbeddingBackward,
)
from src.transformer.language_model import (
    TinyTransformerLanguageModel,
    TransformerForwardResult,
)
from src.transformer.output_head_backward import (
    VocabularyProjectionBackward,
)


@dataclass(frozen=True)
class TransformerBackwardResult:
    token_embedding: Matrix
    decoder: TransformerDecoderBlockGradients
    vocabulary_weights: Matrix
    vocabulary_bias: Matrix


class TinyTransformerBackward:
    def backward(
        self,
        model: TinyTransformerLanguageModel,
        token_ids: list[int],
        forward_result: TransformerForwardResult,
        output_gradient: Matrix,
    ) -> TransformerBackwardResult:
        if (
            forward_result.logits.values.shape
            != output_gradient.shape
        ):
            raise ValueError(
                "Output gradient must match model logits."
            )

        vocabulary_gradients = (
            VocabularyProjectionBackward().backward(
                hidden_states=forward_result.decoder_output,
                weights=model.vocabulary_weights,
                logits=forward_result.logits,
                output_gradient=output_gradient,
            )
        )

        decoder_gradients = (
            TransformerDecoderBlockBackward().backward(
                attention_module=model.attention_module,
                forward_result=forward_result.decoder,
                output_gradient=(
                    vocabulary_gradients.hidden_states
                ),
                feed_forward_weights_1=(
                    model.feed_forward_weights_1
                ),
                feed_forward_weights_2=(
                    model.feed_forward_weights_2
                ),
            )
        )

        embedding_gradient = (
            TokenEmbeddingBackward().backward(
                token_ids=token_ids,
                output_gradient=decoder_gradients.input,
                vocabulary_size=model.vocabulary_size,
            )
        )

        return TransformerBackwardResult(
            token_embedding=embedding_gradient,
            decoder=decoder_gradients,
            vocabulary_weights=(
                vocabulary_gradients.weights
            ),
            vocabulary_bias=(
                vocabulary_gradients.bias
            ),
        )

