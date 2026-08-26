from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix


@dataclass(frozen=True)
class SyntheticHeadResult:
    weights: Matrix
    output: Matrix


class SyntheticAttentionHead:
    def __init__(
        self,
        model_dimension: int,
        head_dimension: int,
        focus: int,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError(
                "model_dimension must be positive."
            )

        if head_dimension <= 0:
            raise ValueError(
                "head_dimension must be positive."
            )

        if focus < 0:
            raise ValueError(
                "focus must be non-negative."
            )

        self._model_dimension = model_dimension
        self._head_dimension = head_dimension
        self._focus = focus

    def forward(
        self,
        inputs: Matrix,
    ) -> SyntheticHeadResult:
        if inputs.columns != self._model_dimension:
            raise ValueError(
                "Input dimension does not match model dimension."
            )

        sequence_length = inputs.rows

        if self._focus >= sequence_length:
            raise ValueError(
                "focus must be smaller than sequence length."
            )

        weights = np.zeros(
            (
                sequence_length,
                sequence_length,
            ),
            dtype=np.float64,
        )

        for row in range(sequence_length):
            if row < self._focus:
                weights[row, row] = 1.0
            else:
                weights[row, self._focus] = 1.0

        output_source = inputs.data

        if self._head_dimension > self._model_dimension:
            raise ValueError(
                "head_dimension cannot exceed model_dimension "
                "for the synthetic head."
            )

        output = weights @ output_source[
            :,
            :self._head_dimension,
        ]

        return SyntheticHeadResult(
            weights=Matrix(weights),
            output=Matrix(output),
        )
