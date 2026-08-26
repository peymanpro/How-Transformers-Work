import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.decoder_block import (
    TransformerDecoderBlock,
)
from src.transformer.decoder_block_backward import (
    TransformerDecoderBlockBackward,
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
            [0.2, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            [0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.7, 0.5],
            [0.6, 0.3, 0.9, 0.2, 0.5, 0.8, 0.1, 0.4],
            [0.5, 0.4, 0.2, 0.9, 0.7, 0.3, 0.6, 0.8],
        ]
    )


def create_gradient() -> Matrix:
    return Matrix.from_values(
        [
            [0.3, -0.2, 0.1, -0.4, 0.2, 0.1, -0.3, 0.5],
            [-0.1, 0.4, -0.2, 0.2, 0.1, -0.3, 0.6, -0.2],
            [0.2, 0.1, -0.4, 0.3, -0.2, 0.5, 0.1, -0.1],
            [-0.3, 0.2, 0.4, -0.1, 0.5, -0.2, 0.3, 0.1],
        ]
    )


def test_decoder_block_backward_should_preserve_shape() -> None:
    block = create_block()
    inputs = create_inputs()

    forward_result = block.forward(
        inputs
    )

    gradients = TransformerDecoderBlockBackward().backward(
        attention_module=block.attention_module,
        forward_result=forward_result,
        output_gradient=create_gradient(),
        feed_forward_weights_1=block.feed_forward_weights_1,
        feed_forward_weights_2=block.feed_forward_weights_2,
    )

    assert gradients.input.shape == inputs.shape


def test_decoder_block_backward_should_produce_finite_gradients() -> None:
    block = create_block()
    inputs = create_inputs()

    forward_result = block.forward(
        inputs
    )

    gradients = TransformerDecoderBlockBackward().backward(
        attention_module=block.attention_module,
        forward_result=forward_result,
        output_gradient=create_gradient(),
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


def test_decoder_block_backward_should_reject_wrong_gradient() -> None:
    block = create_block()
    inputs = create_inputs()

    forward_result = block.forward(
        inputs
    )

    with pytest.raises(ValueError):
        TransformerDecoderBlockBackward().backward(
            attention_module=block.attention_module,
            forward_result=forward_result,
            output_gradient=Matrix.from_values(
                [[1.0, 2.0]]
            ),
            feed_forward_weights_1=block.feed_forward_weights_1,
            feed_forward_weights_2=block.feed_forward_weights_2,
        )
def test_decoder_block_input_gradient_should_match_numerical_gradient() -> None:
    block = create_block()
    inputs = create_inputs()

    forward_result = block.forward(
        inputs
    )

    output_gradient = create_gradient()

    analytic = TransformerDecoderBlockBackward().backward(
        attention_module=block.attention_module,
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
        current = Matrix(values)

        result = block.forward(
            current
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
        analytic.input.data,
        numerical,
        rtol=1e-4,
        atol=1e-5,
    )
