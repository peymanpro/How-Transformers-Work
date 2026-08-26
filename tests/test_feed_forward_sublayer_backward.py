import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.feed_forward import (
    FeedForwardNetwork,
)
from src.transformer.feed_forward_sublayer_backward import (
    FeedForwardSublayerBackward,
)


def create_network() -> FeedForwardNetwork:
    return FeedForwardNetwork(
        model_dimension=4,
        hidden_dimension=8,
        seed=42,
    )


def create_inputs() -> Matrix:
    return Matrix.from_values(
        [
            [0.2, -0.4, 0.7, 0.1],
            [0.5, 0.1, -0.3, 0.8],
        ]
    )


def create_output_gradient() -> Matrix:
    return Matrix.from_values(
        [
            [0.3, -0.2, 0.1, -0.4],
            [-0.1, 0.4, -0.2, 0.2],
        ]
    )


def test_feed_forward_sublayer_backward_should_return_expected_shapes() -> None:
    network = create_network()
    inputs = create_inputs()

    forward_result = network.forward(
        inputs
    )

    gradients = FeedForwardSublayerBackward().backward(
        inputs=inputs,
        feed_forward_weights_1=network.weights_1,
        feed_forward_weights_2=network.weights_2,
        feed_forward_result=forward_result,
        output_gradient=create_output_gradient(),
    )

    assert gradients.input.shape == inputs.shape
    assert gradients.feed_forward.weights_1.shape == (
        4,
        8,
    )
    assert gradients.feed_forward.weights_2.shape == (
        8,
        4,
    )


def test_feed_forward_sublayer_backward_should_reject_wrong_gradient_shape() -> None:
    network = create_network()
    inputs = create_inputs()

    forward_result = network.forward(
        inputs
    )

    with pytest.raises(ValueError):
        FeedForwardSublayerBackward().backward(
            inputs=inputs,
            feed_forward_weights_1=network.weights_1,
            feed_forward_weights_2=network.weights_2,
            feed_forward_result=forward_result,
            output_gradient=Matrix.from_values(
                [[1.0, 2.0]]
            ),
        )


def test_feed_forward_sublayer_input_gradient_should_match_numerical_gradient() -> None:
    network = create_network()
    inputs = create_inputs()
    output_gradient = create_output_gradient()

    forward_result = network.forward(
        inputs
    )

    analytic = FeedForwardSublayerBackward().backward(
        inputs=inputs,
        feed_forward_weights_1=network.weights_1,
        feed_forward_weights_2=network.weights_2,
        feed_forward_result=forward_result,
        output_gradient=output_gradient,
    )

    epsilon = 1e-6
    numerical = np.zeros_like(
        inputs.data
    )

    def scalar_loss(
        values: np.ndarray,
    ) -> float:
        current_inputs = Matrix(values)

        current_ffn = network.forward(
            current_inputs
        )

        residual = current_inputs.add(
            current_ffn.output
        )

        # Same post-LN operation used by the sublayer.
        from src.transformer.layer_norm import (
            LayerNormalization,
        )

        normalized = LayerNormalization(
            dimension=4,
        ).forward(
            residual
        )

        return float(
            np.sum(
                normalized.data
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
        analytic.input.data,
        numerical,
        rtol=1e-5,
        atol=1e-5,
    )
