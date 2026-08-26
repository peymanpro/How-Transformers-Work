import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.output_head import (
    VocabularyProjection,
)
from src.transformer.output_head_backward import (
    VocabularyProjectionBackward,
)


def test_output_head_backward_should_return_expected_shapes() -> None:
    projection = VocabularyProjection(
        model_dimension=4,
        vocabulary_size=6,
        seed=42,
    )

    hidden_states = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )

    logits = projection.forward(
        hidden_states
    )

    gradient = Matrix.from_values(
        [
            [0.1, -0.2, 0.1, 0.0, 0.1, -0.1],
            [0.2, 0.1, -0.1, 0.0, -0.1, -0.1],
        ]
    )

    result = VocabularyProjectionBackward().backward(
        hidden_states=hidden_states,
        weights=_projection_weights(
            projection
        ),
        logits=logits,
        output_gradient=gradient,
    )

    assert result.hidden_states.shape == (
        2,
        4,
    )

    assert result.weights.shape == (
        4,
        6,
    )

    assert result.bias.shape == (
        1,
        6,
    )


def test_output_head_backward_should_compute_weight_gradient() -> None:
    hidden_states = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    projection = VocabularyProjection(
        model_dimension=2,
        vocabulary_size=3,
        seed=42,
    )

    logits = projection.forward(
        hidden_states
    )

    output_gradient = Matrix.from_values(
        [
            [0.5, -0.2, -0.3],
            [0.1, 0.4, -0.5],
        ]
    )

    result = VocabularyProjectionBackward().backward(
        hidden_states=hidden_states,
        weights=_projection_weights(
            projection
        ),
        logits=logits,
        output_gradient=output_gradient,
    )

    expected = (
        hidden_states.data.T
        @ output_gradient.data
    )

    np.testing.assert_allclose(
        result.weights.data,
        expected,
    )


def test_output_head_backward_should_compute_bias_gradient() -> None:
    hidden_states = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    projection = VocabularyProjection(
        model_dimension=2,
        vocabulary_size=3,
        seed=42,
    )

    logits = projection.forward(
        hidden_states
    )

    output_gradient = Matrix.from_values(
        [
            [0.5, -0.2, -0.3],
            [0.1, 0.4, -0.5],
        ]
    )

    result = VocabularyProjectionBackward().backward(
        hidden_states=hidden_states,
        weights=_projection_weights(
            projection
        ),
        logits=logits,
        output_gradient=output_gradient,
    )

    np.testing.assert_allclose(
        result.bias.data,
        np.sum(
            output_gradient.data,
            axis=0,
            keepdims=True,
        ),
    )


def test_output_head_backward_should_reject_invalid_shapes() -> None:
    projection = VocabularyProjection(
        model_dimension=4,
        vocabulary_size=6,
    )

    hidden_states = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    logits = projection.forward(
        Matrix.from_values(
            [[1.0, 2.0, 3.0, 4.0]]
        )
    )

    gradient = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )

    with pytest.raises(ValueError):
        VocabularyProjectionBackward().backward(
            hidden_states=hidden_states,
            weights=_projection_weights(
                projection
            ),
            logits=logits,
            output_gradient=gradient,
        )


def _projection_weights(
    projection: VocabularyProjection,
) -> Matrix:
    """
    Temporary read access for the backward calculation.

    This will become a proper model-parameter API
    when parameter updates are introduced.
    """
    return projection.weights
def test_output_head_hidden_gradient_should_match_numerical_gradient() -> None:
    projection = VocabularyProjection(
        model_dimension=3,
        vocabulary_size=4,
        seed=42,
    )

    hidden_states = Matrix.from_values(
        [
            [0.2, -0.4, 0.7],
            [0.5, 0.1, -0.3],
        ]
    )

    logits = projection.forward(hidden_states)

    output_gradient = Matrix.from_values(
        [
            [0.3, -0.2, 0.1, -0.2],
            [-0.1, 0.4, -0.2, -0.1],
        ]
    )

    analytic = VocabularyProjectionBackward().backward(
        hidden_states=hidden_states,
        weights=projection.weights,
        logits=logits,
        output_gradient=output_gradient,
    )

    epsilon = 1e-6
    numerical = np.zeros_like(hidden_states.data)

    def scalar_loss(values: np.ndarray) -> float:
        current = Matrix(values)
        current_logits = projection.forward(current)

        return float(
            np.sum(
                current_logits.values.data
                * output_gradient.data
            )
        )

    for row in range(hidden_states.rows):
        for column in range(hidden_states.columns):
            plus = hidden_states.data.copy()
            minus = hidden_states.data.copy()

            plus[row, column] += epsilon
            minus[row, column] -= epsilon

            numerical[row, column] = (
                scalar_loss(plus)
                - scalar_loss(minus)
            ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic.hidden_states.data,
        numerical,
        rtol=1e-5,
        atol=1e-5,
    )
