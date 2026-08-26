import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.feed_forward import (
    FeedForwardNetwork,
)
from src.transformer.feed_forward_backward import (
    FeedForwardBackward,
)


def test_feed_forward_backward_should_return_correct_shapes() -> None:
    network = FeedForwardNetwork(
        model_dimension=3,
        hidden_dimension=5,
        seed=42,
    )

    inputs = Matrix.from_values(
        [
            [0.2, -0.4, 0.7],
            [0.5, 0.1, -0.3],
        ]
    )

    forward_result = network.forward(
        inputs
    )

    output_gradient = Matrix.from_values(
        [
            [0.3, -0.2, 0.1],
            [-0.1, 0.4, -0.2],
        ]
    )

    gradients = FeedForwardBackward().backward(
        inputs=inputs,
        weights_1=network.weights_1,
        weights_2=network.weights_2,
        forward_result=forward_result,
        output_gradient=output_gradient,
    )

    assert gradients.inputs.shape == inputs.shape
    assert gradients.weights_1.shape == (
        3,
        5,
    )
    assert gradients.bias_1.shape == (
        1,
        5,
    )
    assert gradients.weights_2.shape == (
        5,
        3,
    )
    assert gradients.bias_2.shape == (
        1,
        3,
    )


def test_feed_forward_input_gradient_should_match_numerical_gradient() -> None:
    network = FeedForwardNetwork(
        model_dimension=3,
        hidden_dimension=5,
        seed=42,
    )

    inputs = Matrix.from_values(
        [
            [0.2, -0.4, 0.7],
            [0.5, 0.1, -0.3],
        ]
    )

    forward_result = network.forward(
        inputs
    )

    output_gradient = Matrix.from_values(
        [
            [0.3, -0.2, 0.1],
            [-0.1, 0.4, -0.2],
        ]
    )

    analytic = FeedForwardBackward().backward(
        inputs=inputs,
        weights_1=network.weights_1,
        weights_2=network.weights_2,
        forward_result=forward_result,
        output_gradient=output_gradient,
    )

    epsilon = 1e-6
    numerical = np.zeros_like(
        inputs.data
    )

    def scalar_loss(
        values: np.ndarray,
    ) -> float:
        result = network.forward(
            Matrix(values)
        )

        return float(
            np.sum(
                result.output.data
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
        analytic.inputs.data,
        numerical,
        rtol=1e-5,
        atol=1e-5,
    )


def test_feed_forward_backward_should_reject_invalid_gradient_shape() -> None:
    network = FeedForwardNetwork(
        model_dimension=3,
        hidden_dimension=5,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    forward_result = network.forward(
        inputs
    )

    with pytest.raises(ValueError):
        FeedForwardBackward().backward(
            inputs=inputs,
            weights_1=network.weights_1,
            weights_2=network.weights_2,
            forward_result=forward_result,
            output_gradient=Matrix.from_values(
                [[1.0, 2.0]]
            ),
        )
