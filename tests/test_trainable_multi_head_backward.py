import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.multi_head_backward import (
    TrainableMultiHeadBackward,
)
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


def create_output_gradient() -> Matrix:
    return Matrix.from_values(
        [
            [0.3, -0.2, 0.1, -0.4],
            [0.2, 0.1, -0.3, 0.5],
            [-0.1, 0.4, 0.2, -0.2],
        ]
    )


def test_trainable_multi_head_backward_should_preserve_shapes() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
        causal=True,
    )
    inputs = create_inputs()
    forward_result = attention.forward(inputs)

    gradients = TrainableMultiHeadBackward().backward(
        attention=attention,
        forward_result=forward_result,
        inputs=inputs,
        output_gradient=create_output_gradient(),
    )

    assert gradients.inputs.shape == inputs.shape
    assert len(gradients.query_weights) == 2
    assert gradients.query_weights[0].shape == (4, 2)
    assert gradients.key_weights[0].shape == (4, 2)
    assert gradients.value_weights[0].shape == (4, 2)
    assert gradients.output_weights.shape == (4, 4)


def test_trainable_multi_head_backward_should_match_input_numerical_gradient() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
        causal=True,
    )
    inputs = create_inputs()
    output_gradient = create_output_gradient()

    forward_result = attention.forward(inputs)
    analytic = TrainableMultiHeadBackward().backward(
        attention=attention,
        forward_result=forward_result,
        inputs=inputs,
        output_gradient=output_gradient,
    )

    epsilon = 1e-6
    numerical = np.zeros_like(inputs.data)

    def scalar_loss(values: np.ndarray) -> float:
        current = attention.forward(Matrix(values))
        return float(
            np.sum(
                current.output.data
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
            ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic.inputs.data,
        numerical,
        rtol=1e-4,
        atol=1e-5,
    )


def test_trainable_multi_head_backward_should_match_query_weight_gradient() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
        causal=False,
    )
    inputs = create_inputs()
    output_gradient = create_output_gradient()
    forward_result = attention.forward(inputs)

    analytic = TrainableMultiHeadBackward().backward(
        attention=attention,
        forward_result=forward_result,
        inputs=inputs,
        output_gradient=output_gradient,
    )

    epsilon = 1e-6
    numerical = np.zeros_like(
        attention.query_weights[0].data
    )

    def scalar_loss(values: np.ndarray) -> float:
        original = attention.query_weights
        replaced = list(original)
        replaced[0] = Matrix(values)
        attention._query_weights = tuple(replaced)

        current = attention.forward(inputs)
        value = float(
            np.sum(
                current.output.data
                * output_gradient.data
            )
        )

        attention._query_weights = original
        return value

    weight = attention.query_weights[0].data

    for row in range(weight.shape[0]):
        for column in range(weight.shape[1]):
            plus = weight.copy()
            minus = weight.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon

            numerical[row, column] = (
                scalar_loss(plus)
                - scalar_loss(minus)
            ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        analytic.query_weights[0].data,
        numerical,
        rtol=1e-4,
        atol=1e-5,
    )


def test_trainable_multi_head_backward_should_produce_finite_gradients() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
        causal=True,
    )
    inputs = create_inputs()
    result = attention.forward(inputs)

    gradients = TrainableMultiHeadBackward().backward(
        attention=attention,
        forward_result=result,
        inputs=inputs,
        output_gradient=create_output_gradient(),
    )

    assert np.isfinite(gradients.inputs.data).all()
    assert np.isfinite(gradients.output_weights.data).all()
