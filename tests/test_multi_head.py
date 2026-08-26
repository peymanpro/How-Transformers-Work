import pytest

from src.math.matrix import Matrix
from src.transformer.multi_head import (
    SyntheticMultiHeadAttention,
)


def test_multi_head_should_create_expected_number_of_heads() -> None:
    attention = SyntheticMultiHeadAttention(
        model_dimension=8,
        head_dimension=2,
        focuses=[0, 1, 2, 2],
    )

    assert attention.number_of_heads == 4


def test_multi_head_should_return_expected_shapes() -> None:
    attention = SyntheticMultiHeadAttention(
        model_dimension=8,
        head_dimension=2,
        focuses=[0, 1, 2, 2],
    )

    inputs = Matrix.from_values(
        [
            [1.0] * 8,
            [2.0] * 8,
            [3.0] * 8,
        ]
    )

    result = attention.forward(inputs)

    assert len(result.head_outputs) == 4
    assert result.concatenated.shape == (3, 8)
    assert result.output.shape == (3, 8)


def test_multi_head_should_reject_incompatible_dimensions() -> None:
    with pytest.raises(ValueError):
        SyntheticMultiHeadAttention(
            model_dimension=8,
            head_dimension=3,
            focuses=[0, 1],
        )


def test_multi_head_should_require_at_least_one_head() -> None:
    with pytest.raises(ValueError):
        SyntheticMultiHeadAttention(
            model_dimension=8,
            head_dimension=8,
            focuses=[],
        )



def test_multi_head_should_support_causal_attention() -> None:
    attention = SyntheticMultiHeadAttention(
        model_dimension=8,
        head_dimension=2,
        focuses=[0, 1, 2, 2],
        causal=True,
    )

    inputs = Matrix.from_values(
        [
            [1.0] * 8,
            [2.0] * 8,
            [3.0] * 8,
            [4.0] * 8,
        ]
    )

    result = attention.forward(inputs)

    for head_output_index in range(
        attention.number_of_heads
    ):
        del head_output_index

    assert result.output.shape == (4, 8)
