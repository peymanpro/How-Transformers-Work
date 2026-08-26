import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.output_head import (
    VocabularyProjection,
)


def test_vocabulary_projection_should_return_token_logits() -> None:
    projection = VocabularyProjection(
        model_dimension=8,
        vocabulary_size=6,
    )

    hidden_states = Matrix.from_values(
        [
            [1.0] * 8,
            [2.0] * 8,
            [3.0] * 8,
        ]
    )

    result = projection.forward(
        hidden_states
    )

    assert result.values.shape == (3, 6)


def test_vocabulary_projection_should_produce_finite_logits() -> None:
    projection = VocabularyProjection(
        model_dimension=4,
        vocabulary_size=5,
        seed=42,
    )

    hidden_states = Matrix.from_values(
        [
            [0.1, 0.2, 0.3, 0.4],
            [-0.5, 0.6, -0.7, 0.8],
        ]
    )

    result = projection.forward(
        hidden_states
    )

    assert np.all(
        np.isfinite(
            result.values.data
        )
    )


def test_vocabulary_projection_should_be_deterministic() -> None:
    hidden_states = Matrix.from_values(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    first = VocabularyProjection(
        model_dimension=2,
        vocabulary_size=4,
        seed=42,
    ).forward(hidden_states)

    second = VocabularyProjection(
        model_dimension=2,
        vocabulary_size=4,
        seed=42,
    ).forward(hidden_states)

    np.testing.assert_allclose(
        first.values.data,
        second.values.data,
    )


def test_vocabulary_projection_should_reject_wrong_dimension() -> None:
    projection = VocabularyProjection(
        model_dimension=8,
        vocabulary_size=6,
    )

    hidden_states = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        projection.forward(hidden_states)
def test_vocabulary_projection_should_update_weights_and_bias() -> None:
    projection = VocabularyProjection(
        model_dimension=2,
        vocabulary_size=3,
        seed=42,
    )

    weights_before = projection.weights.data
    bias_before = projection.bias.data

    weight_gradient = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    bias_gradient = Matrix.from_values(
        [
            [0.5, -0.5, 1.0],
        ]
    )

    projection.apply_gradients(
        weight_gradient=weight_gradient,
        bias_gradient=bias_gradient,
        learning_rate=0.1,
    )

    np.testing.assert_allclose(
        projection.weights.data,
        weights_before
        - 0.1 * weight_gradient.data,
    )

    np.testing.assert_allclose(
        projection.bias.data,
        bias_before
        - 0.1 * bias_gradient.data,
    )
