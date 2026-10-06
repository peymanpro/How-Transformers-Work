import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.encoder_block import (
    TransformerEncoderBlock,
)


def create_block() -> TransformerEncoderBlock:
    return TransformerEncoderBlock(
        model_dimension=8,
        head_dimension=2,
        feed_forward_dimension=16,
        seed=42,
    )


def create_inputs() -> Matrix:
    return Matrix.from_values(
        [
            [1.0, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            [0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.7, 0.5],
            [0.6, 0.3, 0.9, 0.2, 0.5, 0.8, 0.1, 0.4],
            [0.5, 0.4, 0.2, 0.9, 0.7, 0.3, 0.6, 0.8],
        ]
    )


def test_encoder_block_should_preserve_model_shape() -> None:
    block = create_block()

    result = block.forward(
        create_inputs()
    )

    assert result.output.shape == (
        4,
        8,
    )


def test_encoder_block_should_preserve_sequence_length() -> None:
    block = create_block()

    result = block.forward(
        create_inputs()
    )

    assert result.output.rows == 4


def test_encoder_block_should_produce_finite_output() -> None:
    block = create_block()

    result = block.forward(
        create_inputs()
    )

    assert np.all(
        np.isfinite(
            result.output.data
        )
    )


def test_encoder_block_should_apply_attention_and_feed_forward() -> None:
    block = create_block()

    result = block.forward(
        create_inputs()
    )

    assert result.attention.output.shape == (
        4,
        8,
    )

    assert result.after_attention_sublayer.shape == (
        4,
        8,
    )

    assert result.feed_forward_output.shape == (
        4,
        8,
    )

    assert result.output.shape == (
        4,
        8,
    )


def test_encoder_block_should_be_deterministic_for_same_configuration() -> None:
    inputs = create_inputs()

    first = create_block().forward(
        inputs
    )

    second = create_block().forward(
        inputs
    )

    np.testing.assert_allclose(
        first.output.data,
        second.output.data,
    )


def test_encoder_block_should_reject_invalid_model_dimension() -> None:
    with pytest.raises(ValueError):
        TransformerEncoderBlock(
            model_dimension=0,
            head_dimension=1,
            feed_forward_dimension=4,
        )
