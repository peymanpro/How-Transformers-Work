import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.trainable_multi_head import (
    TrainableMultiHeadAttention,
)


def create_inputs() -> Matrix:
    return Matrix.from_values(
        [
            [0.2, 0.5, 0.2, 0.8],
            [0.5, 0.1, 0.7, 0.3],
            [0.6, 0.4, 0.9, 0.2],
        ]
    )


def test_trainable_multi_head_should_expose_expected_head_count() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
    )

    assert attention.number_of_heads == 2


def test_trainable_multi_head_should_return_expected_shapes() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
    )

    result = attention.forward(create_inputs())

    assert len(result.queries) == 2
    assert len(result.keys) == 2
    assert len(result.values) == 2
    assert len(result.weights) == 2
    assert result.concatenated.shape == (3, 4)
    assert result.output.shape == (3, 4)


def test_trainable_multi_head_should_normalize_attention_rows() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
    )

    result = attention.forward(create_inputs())

    for weights in result.weights:
        np.testing.assert_allclose(
            weights.data.sum(axis=1),
            np.ones(3),
            rtol=1e-10,
            atol=1e-10,
        )


def test_trainable_multi_head_should_enforce_causal_visibility() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
        causal=True,
    )

    result = attention.forward(create_inputs())

    for weights in result.weights:
        for row in range(3):
            np.testing.assert_allclose(
                weights.data[row, row + 1 :],
                0.0,
                rtol=0.0,
                atol=1e-12,
            )


def test_trainable_multi_head_should_apply_parameter_updates() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
    )

    result = attention.forward(create_inputs())
    output_gradient = Matrix.from_values(
        [
            [0.1, -0.2, 0.3, -0.1],
            [0.2, 0.1, -0.4, 0.2],
            [-0.1, 0.3, 0.2, 0.4],
            [0.2, -0.1, 0.1, -0.3],
        ]
    )

    query_before = attention.query_weights[0].data
    output_before = result.output.data

    query_gradients = tuple(
        Matrix(np.ones(weight.shape))
        for weight in attention.query_weights
    )
    zeros = tuple(
        Matrix(np.zeros(weight.shape))
        for weight in attention.key_weights
    )

    attention.apply_gradients(
        query_gradients=query_gradients,
        key_gradients=zeros,
        value_gradients=zeros,
        output_gradient=output_gradient,
        learning_rate=0.05,
    )

    assert not np.array_equal(
        query_before,
        attention.query_weights[0].data,
    )
    assert not np.array_equal(
        output_before,
        attention.forward(create_inputs()).output.data,
    )


def test_trainable_multi_head_should_reject_incompatible_dimensions() -> None:
    with pytest.raises(ValueError):
        TrainableMultiHeadAttention(
            model_dimension=5,
            head_dimension=2,
        )
