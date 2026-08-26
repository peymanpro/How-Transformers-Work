from __future__ import annotations

from src.transformer.language_model import (
    TinyTransformerLanguageModel,
)
from src.transformer.prediction import (
    NextTokenPredictor,
)
from src.transformer.training import (
    TransformerTrainer,
)

TOKENS = [
    "the",
    "cat",
    "drinks",
    "milk",
    "<eos>",
    "<unk>",
]


SEQUENCE = [0, 1, 2, 3]
TARGETS = [1, 2, 3, 4]


def print_predictions(
    title: str,
    model: TinyTransformerLanguageModel,
) -> None:
    predictor = NextTokenPredictor()

    result = model.forward(
        SEQUENCE
    )

    predictions = predictor.predict(
        result.logits.values
    )

    target_probabilities = predictor.probability_of(
        result.logits.values,
        TARGETS,
    )

    print(title)
    print("-" * len(title))
    print()

    for position, (
        token_id,
        target_id,
        prediction,
        target_probability,
    ) in enumerate(
        zip(
            SEQUENCE,
            TARGETS,
            predictions,
            target_probabilities,
        )
    ):
        print(
            f"{position}: "
            f"{TOKENS[token_id]:>7} → "
            f"target={TOKENS[target_id]:>7} | "
            f"predicted={TOKENS[prediction.token_id]:>7} | "
            f"target_probability="
            f"{target_probability:.6f}"
        )

    print()


def main() -> None:
    model = TinyTransformerLanguageModel(
        vocabulary_size=len(TOKENS),
        model_dimension=8,
        head_dimension=2,
        head_focuses=[0, 1, 2, 2],
        feed_forward_dimension=16,
        maximum_sequence_length=8,
        seed=42,
    )

    print("Tiny Transformer")
    print("================")
    print()

    print_predictions(
        "Before Training",
        model,
    )

    trainer = TransformerTrainer(
        model=model,
        learning_rate=0.05,
    )

    history = trainer.train(
        sequences=[SEQUENCE],
        targets=[TARGETS],
        epochs=100,
    )

    print(
        f"Initial Loss: {history[0].average_loss:.6f}"
    )

    print(
        f"Final Loss:   {history[-1].average_loss:.6f}"
    )

    print()

    print_predictions(
        "After Training",
        model,
    )


if __name__ == "__main__":
    main()
