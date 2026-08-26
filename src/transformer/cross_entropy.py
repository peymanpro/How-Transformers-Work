from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix


class CrossEntropyFromLogits:
    def loss(
        self,
        logits: Matrix,
        targets: list[int],
    ) -> float:
        if logits.rows != len(targets):
            raise ValueError(
                "Number of targets must match number of rows."
            )

        vocabulary_size = logits.columns

        for target in targets:
            if not 0 <= target < vocabulary_size:
                raise IndexError(
                    "Target index is outside the vocabulary."
                )

        data = logits.data

        shifted = (
            data
            - np.max(
                data,
                axis=1,
                keepdims=True,
            )
        )

        log_sum_exp = np.log(
            np.sum(
                np.exp(shifted),
                axis=1,
            )
        )

        losses = np.empty(
            logits.rows,
            dtype=np.float64,
        )

        for row_index, target in enumerate(targets):
            losses[row_index] = (
                -shifted[row_index, target]
                + log_sum_exp[row_index]
            )

        return float(
            np.mean(losses)
        )

    def gradient(
        self,
        logits: Matrix,
        targets: list[int],
    ) -> Matrix:
        if logits.rows != len(targets):
            raise ValueError(
                "Number of targets must match number of rows."
            )

        vocabulary_size = logits.columns

        for target in targets:
            if not 0 <= target < vocabulary_size:
                raise IndexError(
                    "Target index is outside the vocabulary."
                )

        data = logits.data

        shifted = (
            data
            - np.max(
                data,
                axis=1,
                keepdims=True,
            )
        )

        probabilities = np.exp(shifted)

        probabilities /= np.sum(
            probabilities,
            axis=1,
            keepdims=True,
        )

        for row_index, target in enumerate(targets):
            probabilities[row_index, target] -= 1.0

        probabilities /= logits.rows

        return Matrix(probabilities)
