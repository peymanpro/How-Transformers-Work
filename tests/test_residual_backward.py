import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.residual_backward import (
    backward_residual,
    combine_residual_gradients,
)


def test_residual_backward_should_copy_gradient_to_both_paths() -> None:
    gradient = Matrix.from_values(
        [
            [1.0, -2.0],
            [0.5, 3.0],
        ]
    )

    direct, sublayer = backward_residual(
        gradient
    )

    np.testing.assert_allclose(
        direct.data,
        gradient.data,
    )

    np.testing.assert_allclose(
        sublayer.data,
        gradient.data,
    )


def test_residual_backward_should_combine_two_paths() -> None:
    direct = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    sublayer = Matrix.from_values(
        [
            [0.5, -1.0],
            [2.0, 0.5],
        ]
    )

    result = combine_residual_gradients(
        direct,
        sublayer,
    )

    np.testing.assert_allclose(
        result.data,
        np.array(
            [
                [1.5, 1.0],
                [5.0, 4.5],
            ]
        ),
    )


def test_residual_backward_should_reject_mismatched_shapes() -> None:
    direct = Matrix.from_values(
        [
            [1.0, 2.0],
        ]
    )

    sublayer = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        combine_residual_gradients(
            direct,
            sublayer,
        )
def test_residual_backward_should_preserve_gradient_flow_when_sublayer_is_zero() -> None:
    gradient = Matrix.from_values(
        [
            [0.2, -0.4, 0.8],
        ]
    )

    direct, _ = backward_residual(
        gradient
    )

    np.testing.assert_allclose(
        direct.data,
        gradient.data,
    )
