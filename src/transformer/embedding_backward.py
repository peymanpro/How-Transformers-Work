from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix


class TokenEmbeddingBackward:
    def backward(
        self,
        token_ids: list[int],
        output_gradient: Matrix,
        vocabulary_size: int,
    ) -> Matrix:
        if not token_ids:
            raise ValueError(
                "token_ids cannot be empty."
            )

        if output_gradient.rows != len(token_ids):
            raise ValueError(
                "Gradient rows must match token count."
            )

        if output_gradient.columns <= 0:
            raise ValueError(
                "Embedding dimension must be positive."
            )

        for token_id in token_ids:
            if not 0 <= token_id < vocabulary_size:
                raise IndexError(
                    "Token ID is outside the vocabulary."
                )

        gradient = np.zeros(
            (
                vocabulary_size,
                output_gradient.columns,
            ),
            dtype=np.float64,
        )

        for row, token_id in enumerate(token_ids):
            gradient[token_id] += (
                output_gradient.data[row]
            )

        return Matrix(gradient)
