import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.residual import add_residual


def test_residual_should_add_input_and_sublayer_output() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    output = Matrix.from_values(
        [
            [0.5, 1.5],
            [2.0, -1.0],
        ]
    )

    result = add_residual(
        inputs,
        output,
    )

    np.testing.assert_allclose(
        result.data,
        np.array(
            [
                [1.5, 3.5],
                [5.0, 3.0],
            ]
        ),
    )


def test_residual_should_preserve_shape() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    output = Matrix.from_values(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    result = add_residual(
        inputs,
        output,
    )

    assert result.shape == inputs.shape


def test_residual_should_reject_different_shapes() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 2.0],
        ]
    )

    output = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        add_residual(
            inputs,
            output,
        )
