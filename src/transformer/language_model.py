from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.decoder_block import (
    TransformerDecoderBlock,
)
from src.transformer.embedding import TokenEmbedding
from src.transformer.output_head import (
    TokenLogits,
    VocabularyProjection,
)
from src.transformer.positional_encoding import (
    SinusoidalPositionalEncoding,
)


@dataclass(frozen=True)
class TransformerForwardResult:
    embeddings: Matrix
    decoder_output: Matrix
    logits: TokenLogits


class TinyTransformerLanguageModel:
    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        head_dimension: int,
        head_focuses: list[int],
        feed_forward_dimension: int,
        maximum_sequence_length: int,
        seed: int = 42,
    ) -> None:
        self._embedding = TokenEmbedding(
            vocabulary_size=vocabulary_size,
            embedding_dimension=model_dimension,
            seed=seed,
        )

        self._positional_encoding = (
            SinusoidalPositionalEncoding(
                maximum_sequence_length=maximum_sequence_length,
                embedding_dimension=model_dimension,
            )
        )

        self._decoder = TransformerDecoderBlock(
            model_dimension=model_dimension,
            head_dimension=head_dimension,
            head_focuses=head_focuses,
            feed_forward_dimension=feed_forward_dimension,
            seed=seed,
        )

        self._vocabulary_projection = VocabularyProjection(
            model_dimension=model_dimension,
            vocabulary_size=vocabulary_size,
            seed=seed,
        )

    def forward(
        self,
        token_ids: list[int],
    ) -> TransformerForwardResult:
        token_embeddings = self._embedding.encode(
            token_ids
        )

        transformer_input = (
            self._positional_encoding.add_to(
                token_embeddings
            )
        )

        decoder_result = self._decoder.forward(
            transformer_input
        )

        logits = self._vocabulary_projection.forward(
            decoder_result.output
        )

        return TransformerForwardResult(
            embeddings=transformer_input,
            decoder_output=decoder_result.output,
            logits=logits,
        )
