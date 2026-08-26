import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.layer_norm import (
    LayerNormalization,
)
from src.transformer.layer_norm_backward import (
    LayerNormalizationBackward,
)


def test_layer_norm_backward_should_preserve_shape() -> None:
    inputs = Matrix.from_values(
        [
            [0.2, -0.4, 0.7, 0.1],
            [0.5, 0.1, -0.3, 0.8],
        ]
    )

    output_gradient = Matrix.from_values(
        [
            [0.3, -0.2, 0.1, -0.4],
            [-0.1, 0.4, -0.2, 0.2],
        ]
    )

    result = LayerNormalizationBackward().backward(
        inputs,
        output_gradient,
    )

    assert result.shape == inputs.shape


def test_layer_norm_backward_should_match_numerical_gradient() -> None:
    normalization = LayerNormalization(
        dimension=4,
    )

    inputs = Matrix.from_values(
        [
            [0.2, -0.4, 0.7, 0.1],
            [0.5, 0.1, -0.3, 0.8],
        ]
    )

    output_gradient = Matrix.from_values(
        [
            [0.3, -0.2, 0.1, -0.4],
            [-0.1, 0.4, -0.2, 0.2],
        ]
    )

    analytic = LayerNormalizationBackward().backward(
        inputs,
        output_gradient,
    )

    epsilon = 1e-6
    numerical = np.zeros_like(
        inputs.data
    )

    def scalar_loss(
        values: np.ndarray,
    ) -> float:
        result = normalization.forward(
            Matrix(values)
        )

        return float(
            np.sum(
                result.data
                * output_gradient.data
            )
        )

    for row in range(inputs.rows):
        for column in range(inputs.columns):
            plus = inputs.data.copy()
            minus = inputs.data.copy()

            plus[row, column] += epsilon
            minus[row, column] -= epsilon

            numerical[row, column] = (
                scalar_loss(plus)
                - scalar_loss(minus)
            ) / (
                2.0 * epsilon
            )

    np.testing.assert_allclose(
        analytic.data,
        numerical,
        rtol=1e-5,
        atol=1e-5,
    )


def test_layer_norm_backward_should_reject_shape_mismatch() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    output_gradient = Matrix.from_values(
        [
            [1.0, 2.0],
        ]
    )

    with pytest.raises(ValueError):
        LayerNormalizationBackward().backward(
            inputs,
            output_gradient,
        )
