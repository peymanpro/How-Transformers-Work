import numpy as np

from src.math.matrix import Matrix
from src.transformer.attention_sublayer import (
    AttentionSublayer,
)


def test_attention_sublayer_should_apply_residual_then_layer_norm() -> None:
    sublayer = AttentionSublayer(
        dimension=4,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ]
    )

    attention_output = Matrix.from_values(
        [
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    result = sublayer.forward(
        inputs,
        attention_output,
    )

    np.testing.assert_allclose(
        result.residual_output.data,
        np.array(
            [
                [1.5, 2.5, 3.5, 4.5],
                [3.0, 5.0, 7.0, 9.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        np.mean(
            result.normalized_output.data,
            axis=1,
        ),
        np.zeros(2),
        atol=1e-10,
    )
