import numpy as np

from src.math.matrix import Matrix
from src.transformer.decoder_block import (
    TransformerDecoderBlock,
)


def create_block() -> TransformerDecoderBlock:
    return TransformerDecoderBlock(
        model_dimension=8,
        head_dimension=2,
        head_focuses=[0, 1, 2, 2],
        feed_forward_dimension=16,
        seed=42,
    )


def create_inputs() -> Matrix:
    return Matrix.from_values(
        [
            [1.0] * 8,
            [2.0] * 8,
            [3.0] * 8,
            [4.0] * 8,
        ]
    )


def test_decoder_block_should_preserve_shape() -> None:
    result = create_block().forward(
        create_inputs()
    )

    assert result.output.shape == (4, 8)


def test_decoder_block_should_produce_finite_output() -> None:
    result = create_block().forward(
        create_inputs()
    )

    assert np.all(
        np.isfinite(
            result.output.data
        )
    )


def test_decoder_block_should_include_attention_and_ffn() -> None:
    result = create_block().forward(
        create_inputs()
    )

    assert result.attention.output.shape == (4, 8)
    assert result.after_attention_sublayer.shape == (4, 8)
    assert result.feed_forward_output.shape == (4, 8)
    assert result.output.shape == (4, 8)
