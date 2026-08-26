import numpy as np
import pytest

from src.math.matrix import Matrix


def test_matrix_should_preserve_shape() -> None:
    matrix = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    assert matrix.shape == (2, 2)
    assert matrix.rows == 2
    assert matrix.columns == 2


def test_matrix_should_transpose() -> None:
    matrix = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    np.testing.assert_array_equal(
        matrix.transpose().data,
        np.array(
            [
                [1.0, 3.0],
                [2.0, 4.0],
            ]
        ),
    )


def test_matrix_should_multiply() -> None:
    left = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    right = Matrix.from_values(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    np.testing.assert_array_equal(
        left.multiply(right).data,
        np.array(
            [
                [19.0, 22.0],
                [43.0, 50.0],
            ]
        ),
    )


def test_matrix_should_add() -> None:
    first = Matrix.from_values(
        [[1.0, 2.0]]
    )

    second = Matrix.from_values(
        [[3.0, 4.0]]
    )

    np.testing.assert_array_equal(
        first.add(second).data,
        np.array([[4.0, 6.0]]),
    )


def test_matrix_should_reject_incompatible_shapes() -> None:
    first = Matrix.from_values(
        [[1.0, 2.0, 3.0]]
    )

    second = Matrix.from_values(
        [[1.0, 2.0]]
    )

    with pytest.raises(ValueError):
        first.multiply(second)
