import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.encoder_block import (
    TransformerEncoderBlock,
)
from src.transformer.encoder_block_backward import (
    TransformerEncoderBlockBackward,
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
            [0.2, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            [0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.7, 0.5],
            [0.6, 0.3, 0.9, 0.2, 0.5, 0.8, 0.1, 0.4],
            [0.5, 0.4, 0.2, 0.9, 0.7, 0.3, 0.6, 0.8],
        ]
    )


def create_output_gradient() -> Matrix:
    return Matrix.from_values(
        [
            [0.3, -0.2, 0.1, -0.4, 0.2, 0.1, -0.3, 0.5],
            [-0.1, 0.4, -0.2, 0.2, 0.1, -0.3, 0.6, -0.2],
            [0.2, 0.1, -0.4, 0.3, -0.2, 0.5, 0.1, -0.1],
            [-0.3, 0.2, 0.4, -0.1, 0.5, -0.2, 0.3, 0.1],
        ]
    )


def test_encoder_block_backward_should_return_input_gradient() -> None:
    block = create_block()
    inputs = create_inputs()

    forward_result = block.forward(
        inputs
    )

    gradients = TransformerEncoderBlockBackward().backward(
        block=block._attention,
        forward_result=forward_result,
        output_gradient=create_output_gradient(),
        feed_forward_weights_1=block.feed_forward_weights_1,
        feed_forward_weights_2=block.feed_forward_weights_2,
    )

    assert gradients.input.shape == inputs.shape


def test_encoder_block_backward_should_produce_finite_gradients() -> None:
    block = create_block()
    inputs = create_inputs()

    forward_result = block.forward(
        inputs
    )

    gradients = TransformerEncoderBlockBackward().backward(
        block=block._attention,
        forward_result=forward_result,
        output_gradient=create_output_gradient(),
        feed_forward_weights_1=block.feed_forward_weights_1,
        feed_forward_weights_2=block.feed_forward_weights_2,
    )

    assert np.all(
        np.isfinite(
            gradients.input.data
        )
    )

    assert np.all(
        np.isfinite(
            gradients.feed_forward.weights_1.data
        )
    )

    assert np.all(
        np.isfinite(
            gradients.feed_forward.weights_2.data
        )
    )


def test_encoder_block_backward_should_reject_wrong_output_gradient() -> None:
    block = create_block()
    inputs = create_inputs()

    forward_result = block.forward(
        inputs
    )

    with pytest.raises(ValueError):
        TransformerEncoderBlockBackward().backward(
            block=block._attention,
            forward_result=forward_result,
            output_gradient=Matrix.from_values(
                [[1.0, 2.0]]
            ),
            feed_forward_weights_1=block.feed_forward_weights_1,
            feed_forward_weights_2=block.feed_forward_weights_2,
        )

def test_encoder_block_input_gradient_should_match_numerical_gradient() -> None:
    block = create_block()
    inputs = create_inputs()

    forward_result = block.forward(
        inputs
    )

    output_gradient = create_output_gradient()

    analytic = TransformerEncoderBlockBackward().backward(
        block=block._attention,
        forward_result=forward_result,
        output_gradient=output_gradient,
        feed_forward_weights_1=block.feed_forward_weights_1,
        feed_forward_weights_2=block.feed_forward_weights_2,
    )

    epsilon = 1e-6

    numerical = np.zeros_like(
        inputs.data
    )

    def scalar_loss(
        values: np.ndarray,
    ) -> float:
        current_inputs = Matrix(values)

        current_result = block.forward(
            current_inputs
        )

        return float(
            np.sum(
                current_result.output.data
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
        rtol=1e-4,
        atol=1e-5,
    )

