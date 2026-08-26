import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.feed_forward import (
    FeedForwardNetwork,
)


def test_feed_forward_should_preserve_sequence_and_model_dimensions() -> None:
    network = FeedForwardNetwork(
        model_dimension=4,
        hidden_dimension=8,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )

    result = network.forward(inputs)

    assert result.hidden.shape == (3, 8)
    assert result.output.shape == (3, 4)


def test_feed_forward_should_apply_relu() -> None:
    network = FeedForwardNetwork(
        model_dimension=2,
        hidden_dimension=2,
    )

    inputs = Matrix.from_values(
        [
            [-1.0, -2.0],
            [1.0, 2.0],
        ]
    )

    result = network.forward(inputs)

    assert np.all(
        result.hidden.data >= 0.0
    )


def test_feed_forward_should_produce_finite_values() -> None:
    network = FeedForwardNetwork(
        model_dimension=4,
        hidden_dimension=8,
    )

    inputs = Matrix.from_values(
        [
            [0.1, 0.2, 0.3, 0.4],
            [-0.5, 0.6, -0.7, 0.8],
        ]
    )

    result = network.forward(inputs)

    assert np.all(
        np.isfinite(result.hidden.data)
    )

    assert np.all(
        np.isfinite(result.output.data)
    )


def test_feed_forward_should_be_deterministic_for_same_seed() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    first = FeedForwardNetwork(
        model_dimension=2,
        hidden_dimension=4,
        seed=42,
    ).forward(inputs)

    second = FeedForwardNetwork(
        model_dimension=2,
        hidden_dimension=4,
        seed=42,
    ).forward(inputs)

    np.testing.assert_allclose(
        first.output.data,
        second.output.data,
    )


def test_feed_forward_should_reject_wrong_input_dimension() -> None:
    network = FeedForwardNetwork(
        model_dimension=4,
        hidden_dimension=8,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        network.forward(inputs)


def test_feed_forward_dimensions_should_be_exposed() -> None:
    network = FeedForwardNetwork(
        model_dimension=4,
        hidden_dimension=16,
    )

    assert network.model_dimension == 4
    assert network.hidden_dimension == 16
