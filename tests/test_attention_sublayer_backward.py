import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.attention_sublayer_backward import (
    AttentionSublayerBackward,
)
from src.transformer.trainable_multi_head import (
    TrainableMultiHeadAttention,
)


def create_attention() -> TrainableMultiHeadAttention:
    return TrainableMultiHeadAttention(
        model_dimension=8,
        head_dimension=2,
        seed=42,
        causal=True,
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


def test_attention_sublayer_backward_should_return_correct_shapes() -> None:
    attention = create_attention()
    inputs = create_inputs()

    forward_result = attention.forward(
        inputs
    )

    gradients = AttentionSublayerBackward().backward(
        attention=attention,
        inputs=inputs,
        forward_result=forward_result,
        output_gradient=create_gradient(),
    )

    assert gradients.input.shape == inputs.shape
    assert gradients.attention.inputs.shape == inputs.shape
    assert gradients.attention.output_weights.shape == (
        8,
        8,
    )


def test_attention_sublayer_backward_input_gradient_should_be_finite() -> None:
    attention = create_attention()
    inputs = create_inputs()

    forward_result = attention.forward(
        inputs
    )

    gradients = AttentionSublayerBackward().backward(
        attention=attention,
        inputs=inputs,
        forward_result=forward_result,
        output_gradient=create_gradient(),
    )

    assert np.all(
        np.isfinite(
            gradients.input.data
        )
    )


def test_attention_sublayer_backward_should_reject_wrong_gradient_shape() -> None:
    attention = create_attention()
    inputs = create_inputs()

    forward_result = attention.forward(
        inputs
    )

    with pytest.raises(ValueError):
        AttentionSublayerBackward().backward(
            attention=attention,
            inputs=inputs,
            forward_result=forward_result,
            output_gradient=Matrix.from_values(
                [[1.0, 2.0]]
            ),
        )
