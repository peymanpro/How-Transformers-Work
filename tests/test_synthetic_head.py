import numpy as np
import pytest

from src.attention.synthetic_head import (
    SyntheticAttentionHead,
)
from src.math.matrix import Matrix


def test_synthetic_head_should_return_expected_shape() -> None:
    head = SyntheticAttentionHead(
        model_dimension=6,
        head_dimension=2,
        focus=1,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ]
    )

    result = head.forward(inputs)

    assert result.weights.shape == (3, 3)
    assert result.output.shape == (3, 2)


def test_synthetic_head_should_focus_on_configured_position() -> None:
    head = SyntheticAttentionHead(
        model_dimension=4,
        head_dimension=2,
        focus=1,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )

    result = head.forward(inputs)

    np.testing.assert_allclose(
        result.weights.data,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )


def test_synthetic_head_should_project_selected_context() -> None:
    head = SyntheticAttentionHead(
        model_dimension=4,
        head_dimension=2,
        focus=1,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )

    result = head.forward(inputs)

    np.testing.assert_allclose(
        result.output.data,
        np.array(
            [
                [1.0, 2.0],
                [5.0, 6.0],
                [5.0, 6.0],
            ]
        ),
    )


def test_synthetic_head_should_reject_focus_beyond_sequence_length() -> None:
    head = SyntheticAttentionHead(
        model_dimension=4,
        head_dimension=2,
        focus=3,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )

    with pytest.raises(ValueError):
        head.forward(inputs)


def test_synthetic_head_should_reject_too_large_dimension() -> None:
    head = SyntheticAttentionHead(
        model_dimension=4,
        head_dimension=5,
        focus=0,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
        ]
    )

    with pytest.raises(ValueError):
        head.forward(inputs)

