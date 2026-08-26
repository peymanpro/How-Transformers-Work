from __future__ import annotations

from dataclasses import dataclass

from src.transformer.language_model import (
    TinyTransformerLanguageModel,
)
from src.transformer.training_step import (
    TransformerTrainingStep,
)


@dataclass(frozen=True)
class TrainingEpochResult:
    epoch: int
    average_loss: float


class TransformerTrainer:
    def __init__(
        self,
        model: TinyTransformerLanguageModel,
        learning_rate: float,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        self._model = model

        self._step = TransformerTrainingStep(
            learning_rate=learning_rate,
        )

    def train(
        self,
        sequences: list[list[int]],
        targets: list[list[int]],
        epochs: int,
    ) -> list[TrainingEpochResult]:
        if not sequences:
            raise ValueError(
                "Training sequences cannot be empty."
            )

        if len(sequences) != len(targets):
            raise ValueError(
                "Sequences and targets must have equal length."
            )

        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        for sequence, target in zip(
            sequences,
            targets,
        ):
            if not sequence:
                raise ValueError(
                    "Training sequences cannot contain empty sequences."
                )

            if len(sequence) != len(target):
                raise ValueError(
                    "Each sequence must have a matching target sequence."
                )

        history: list[TrainingEpochResult] = []

        for epoch in range(1, epochs + 1):
            total_loss = 0.0

            for sequence, target in zip(
                sequences,
                targets,
            ):
                result = self._step.run(
                    model=self._model,
                    token_ids=sequence,
                    targets=target,
                )

                total_loss += result.loss

            average_loss = (
                total_loss
                / len(sequences)
            )

            history.append(
                TrainingEpochResult(
                    epoch=epoch,
                    average_loss=average_loss,
                )
            )

        return history
