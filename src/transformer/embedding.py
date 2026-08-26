from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix


class TokenEmbedding:
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        seed: int = 42,
    ) -> None:
        if vocabulary_size <= 0:
            raise ValueError(
                "vocabulary_size must be positive."
            )

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be positive."
            )

        self._vocabulary_size = vocabulary_size
        self._embedding_dimension = embedding_dimension

        rng = np.random.default_rng(seed)

        scale = 1.0 / np.sqrt(embedding_dimension)

        self._weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(
                    vocabulary_size,
                    embedding_dimension,
                ),
            )
        )

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def encode(
        self,
        token_ids: list[int],
    ) -> Matrix:
        if not token_ids:
            raise ValueError(
                "token_ids cannot be empty."
            )

        for token_id in token_ids:
            if not 0 <= token_id < self._vocabulary_size:
                raise ValueError(
                    "token_id is outside the vocabulary."
                )

        return Matrix(
            self._weights.data[token_ids]
        )
