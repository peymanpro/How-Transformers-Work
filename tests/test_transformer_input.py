import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.embedding import TokenEmbedding
from src.transformer.positional_encoding import (
    SinusoidalPositionalEncoding,
)


def test_token_embedding_should_return_one_vector_per_token() -> None:
    embedding = TokenEmbedding(
        vocabulary_size=10,
        embedding_dimension=6,
    )

    result = embedding.encode(
        [1, 4, 7]
    )

    assert result.shape == (3, 6)


def test_token_embedding_should_be_deterministic_for_same_seed() -> None:
    first = TokenEmbedding(
        vocabulary_size=10,
        embedding_dimension=4,
        seed=42,
    ).encode([1, 2])

    second = TokenEmbedding(
        vocabulary_size=10,
        embedding_dimension=4,
        seed=42,
    ).encode([1, 2])

    np.testing.assert_allclose(
        first.data,
        second.data,
    )


def test_token_embedding_should_reject_invalid_token() -> None:
    embedding = TokenEmbedding(
        vocabulary_size=5,
        embedding_dimension=4,
    )

    with pytest.raises(ValueError):
        embedding.encode([5])


def test_positional_encoding_should_have_expected_shape() -> None:
    encoding = SinusoidalPositionalEncoding(
        maximum_sequence_length=8,
        embedding_dimension=6,
    )

    result = encoding.encode(4)

    assert result.shape == (4, 6)


def test_positional_encoding_should_depend_on_position() -> None:
    encoding = SinusoidalPositionalEncoding(
        maximum_sequence_length=8,
        embedding_dimension=6,
    )

    result = encoding.encode(4)

    assert not np.allclose(
        result.data[0],
        result.data[1],
    )


def test_first_position_should_have_expected_sinusoidal_values() -> None:
    encoding = SinusoidalPositionalEncoding(
        maximum_sequence_length=8,
        embedding_dimension=4,
    )

    result = encoding.encode(1)

    np.testing.assert_allclose(
        result.data[0],
        np.array([0.0, 1.0, 0.0, 1.0]),
        atol=1e-12,
    )


def test_positional_encoding_should_reject_too_long_sequence() -> None:
    encoding = SinusoidalPositionalEncoding(
        maximum_sequence_length=4,
        embedding_dimension=6,
    )

    with pytest.raises(ValueError):
        encoding.encode(5)


def test_positional_encoding_should_be_addable_to_embeddings() -> None:
    embeddings = Matrix.from_values(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    encoding = SinusoidalPositionalEncoding(
        maximum_sequence_length=4,
        embedding_dimension=4,
    )

    result = encoding.add_to(
        embeddings
    )

    assert result.shape == embeddings.shape

    np.testing.assert_allclose(
        result.data[0],
        embeddings.data[0]
        + np.array([0.0, 1.0, 0.0, 1.0]),
    )
