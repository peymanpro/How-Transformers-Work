from __future__ import annotations

from dataclasses import dataclass

from src.transformer.cross_entropy import (
    CrossEntropyFromLogits,
)
from src.transformer.language_model import (
    TinyTransformerLanguageModel,
)
from src.transformer.language_model_backward import (
    TinyTransformerBackward,
)


@dataclass(frozen=True)
class TrainingStepResult:
    loss: float


class TransformerTrainingStep:
    def __init__(
        self,
        learning_rate: float,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        self._learning_rate = learning_rate
        self._loss = CrossEntropyFromLogits()

    def run(
        self,
        model: TinyTransformerLanguageModel,
        token_ids: list[int],
        targets: list[int],
    ) -> TrainingStepResult:
        forward_result = model.forward(
            token_ids
        )

        loss = self._loss.loss(
            forward_result.logits.values,
            targets,
        )

        output_gradient = self._loss.gradient(
            forward_result.logits.values,
            targets,
        )

        gradients = TinyTransformerBackward().backward(
            model=model,
            token_ids=token_ids,
            forward_result=forward_result,
            output_gradient=output_gradient,
        )

        model.apply_vocabulary_gradients(
            weight_gradient=(
                gradients.vocabulary_weights
            ),
            bias_gradient=(
                gradients.vocabulary_bias
            ),
            learning_rate=self._learning_rate,
        )

        return TrainingStepResult(
            loss=loss
        )
