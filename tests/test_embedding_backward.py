import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.embedding_backward import (
    TokenEmbeddingBackward,
)


def test_embedding_backward_should_accumulate_repeated_tokens() -> None:
    gradient = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    result = TokenEmbeddingBackward().backward(
        token_ids=[1, 2, 1],
        output_gradient=gradient,
        vocabulary_size=4,
    )

    np.testing.assert_allclose(
        result.data,
        np.array(
            [
                [0.0, 0.0],
                [6.0, 8.0],
                [3.0, 4.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_embedding_backward_should_return_full_vocabulary_shape() -> None:
    gradient = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    result = TokenEmbeddingBackward().backward(
        token_ids=[2],
        output_gradient=gradient,
        vocabulary_size=5,
    )

    assert result.shape == (5, 3)


def test_embedding_backward_should_reject_invalid_token() -> None:
    with pytest.raises(IndexError):
        TokenEmbeddingBackward().backward(
            token_ids=[5],
            output_gradient=Matrix.from_values(
                [[1.0, 2.0]]
            ),
            vocabulary_size=5,
        )
