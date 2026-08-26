import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.cross_entropy import (
    CrossEntropyFromLogits,
)


def test_cross_entropy_should_calculate_loss_from_logits() -> None:
    loss = CrossEntropyFromLogits()

    logits = Matrix.from_values(
        [
            [0.0, 2.0, 0.0],
        ]
    )

    result = loss.loss(
        logits,
        [1],
    )

    expected = np.log(
        np.exp(0.0)
        + np.exp(2.0)
        + np.exp(0.0)
    ) - 2.0

    assert result == pytest.approx(
        expected
    )


def test_cross_entropy_gradient_should_have_logits_shape() -> None:
    loss = CrossEntropyFromLogits()

    logits = Matrix.from_values(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
        ]
    )

    result = loss.gradient(
        logits,
        [1, 2],
    )

    assert result.shape == logits.shape


def test_cross_entropy_gradient_rows_should_sum_to_zero() -> None:
    loss = CrossEntropyFromLogits()

    logits = Matrix.from_values(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
        ]
    )

    gradient = loss.gradient(
        logits,
        [1, 2],
    )

    np.testing.assert_allclose(
        np.sum(
            gradient.data,
            axis=1,
        ),
        np.zeros(2),
        atol=1e-12,
    )


def test_cross_entropy_should_reject_invalid_target() -> None:
    loss = CrossEntropyFromLogits()

    logits = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(IndexError):
        loss.loss(
            logits,
            [3],
        )


def test_cross_entropy_should_reject_mismatched_targets() -> None:
    loss = CrossEntropyFromLogits()

    logits = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    with pytest.raises(ValueError):
        loss.loss(
            logits,
            [1],
        )


def test_cross_entropy_should_be_numerically_stable() -> None:
    loss = CrossEntropyFromLogits()

    logits = Matrix.from_values(
        [
            [1000.0, 1001.0, 1002.0],
        ]
    )

    result = loss.loss(
        logits,
        [2],
    )

    assert np.isfinite(result)
