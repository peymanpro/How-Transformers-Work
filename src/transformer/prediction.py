from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix


@dataclass(frozen=True)
class TokenPrediction:
    token_id: int
    probability: float


class NextTokenPredictor:
    def predict(
        self,
        logits: Matrix,
    ) -> list[TokenPrediction]:
        predictions: list[TokenPrediction] = []

        for row in logits.data:
            shifted = row - np.max(row)

            probabilities = np.exp(
                shifted
            )

            probabilities /= np.sum(
                probabilities
            )

            token_id = int(
                np.argmax(probabilities)
            )

            predictions.append(
                TokenPrediction(
                    token_id=token_id,
                    probability=float(
                        probabilities[token_id]
                    ),
                )
            )

        return predictions

    def probability_of(
        self,
        logits: Matrix,
        token_ids: list[int],
    ) -> list[float]:
        if logits.rows != len(token_ids):
            raise ValueError(
                "Number of targets must match logits rows."
            )

        probabilities: list[float] = []

        for row, token_id in zip(
            logits.data,
            token_ids,
        ):
            if not 0 <= token_id < logits.columns:
                raise IndexError(
                    "Target token is outside the vocabulary."
                )

            shifted = row - np.max(row)

            row_probabilities = np.exp(
                shifted
            )

            row_probabilities /= np.sum(
                row_probabilities
            )

            probabilities.append(
                float(
                    row_probabilities[token_id]
                )
            )

        return probabilities
