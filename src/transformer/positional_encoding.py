from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix


class SinusoidalPositionalEncoding:
    def __init__(
        self,
        maximum_sequence_length: int,
        embedding_dimension: int,
    ) -> None:
        if maximum_sequence_length <= 0:
            raise ValueError(
                "maximum_sequence_length must be positive."
            )

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be positive."
            )

        self._maximum_sequence_length = (
            maximum_sequence_length
        )
        self._embedding_dimension = embedding_dimension

        positions = np.arange(
            maximum_sequence_length,
            dtype=np.float64,
        )[:, np.newaxis]

        dimensions = np.arange(
            embedding_dimension,
            dtype=np.float64,
        )[np.newaxis, :]

        angle_rates = np.power(
            10000.0,
            -(2.0 * np.floor(dimensions / 2.0))
            / embedding_dimension,
        )

        angles = positions * angle_rates

        encoding = np.zeros_like(angles)

        encoding[:, 0::2] = np.sin(
            angles[:, 0::2]
        )

        encoding[:, 1::2] = np.cos(
            angles[:, 1::2]
        )

        self._encoding = Matrix(encoding)

    @property
    def maximum_sequence_length(self) -> int:
        return self._maximum_sequence_length

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def encode(
        self,
        sequence_length: int,
    ) -> Matrix:
        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
            )

        if sequence_length > self._maximum_sequence_length:
            raise ValueError(
                "sequence_length exceeds maximum_sequence_length."
            )

        return Matrix(
            self._encoding.data[
                :sequence_length
            ]
        )

    def add_to(
        self,
        embeddings: Matrix,
    ) -> Matrix:
        sequence_length = embeddings.rows

        if embeddings.columns != self._embedding_dimension:
            raise ValueError(
                "Embedding dimension does not match "
                "positional encoding dimension."
            )

        positions = self.encode(
            sequence_length
        )

        return embeddings.add(
            positions
        )
