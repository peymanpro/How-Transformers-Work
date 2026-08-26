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
        model.apply_gradients(
            embedding_gradient=gradients.token_embedding,
            attention_output_gradient=(
                gradients.decoder.attention.attention.output_weights
            ),
            feed_forward_weights_1_gradient=(
                gradients.decoder.feed_forward.weights_1
            ),
            feed_forward_bias_1_gradient=(
                gradients.decoder.feed_forward.bias_1
            ),
            feed_forward_weights_2_gradient=(
                gradients.decoder.feed_forward.weights_2
            ),
            feed_forward_bias_2_gradient=(
                gradients.decoder.feed_forward.bias_2
            ),
            vocabulary_weights_gradient=(
                gradients.vocabulary_weights
            ),
            vocabulary_bias_gradient=(
                gradients.vocabulary_bias
            ),
            learning_rate=self._learning_rate,
        )

        return TrainingStepResult(
            loss=loss
        )



