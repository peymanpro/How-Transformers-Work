import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.layer_norm import LayerNormalization


def test_layer_norm_should_preserve_shape() -> None:
    normalization = LayerNormalization(
        dimension=4,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )

    result = normalization.forward(inputs)

    assert result.shape == inputs.shape


def test_layer_norm_should_produce_zero_mean_per_token() -> None:
    normalization = LayerNormalization(
        dimension=4,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 7.0, 9.0, 11.0],
        ]
    )

    result = normalization.forward(inputs)

    np.testing.assert_allclose(
        np.mean(
            result.data,
            axis=1,
        ),
        np.zeros(2),
        atol=1e-10,
    )


def test_layer_norm_should_produce_unit_variance_per_token() -> None:
    normalization = LayerNormalization(
        dimension=4,
        epsilon=1e-8,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 7.0, 9.0, 11.0],
        ]
    )

    result = normalization.forward(inputs)

    np.testing.assert_allclose(
        np.var(
            result.data,
            axis=1,
        ),
        np.ones(2),
        atol=1e-7,
    )


def test_layer_norm_should_normalize_each_token_independently() -> None:
    normalization = LayerNormalization(
        dimension=3,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ]
    )

    result = normalization.forward(inputs)

    np.testing.assert_allclose(
        result.data[0],
        result.data[1],
        atol=1e-4,
    )


def test_layer_norm_should_reject_wrong_dimension() -> None:
    normalization = LayerNormalization(
        dimension=4,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        normalization.forward(inputs)


def test_layer_norm_should_reject_invalid_epsilon() -> None:
    with pytest.raises(ValueError):
        LayerNormalization(
            dimension=4,
            epsilon=0.0,
        )

