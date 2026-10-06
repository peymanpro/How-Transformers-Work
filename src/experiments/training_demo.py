from __future__ import annotations

from src.transformer.language_model import (
    TinyTransformerLanguageModel,
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


def main() -> None:
    model = TinyTransformerLanguageModel(
        vocabulary_size=len(TOKENS),
        model_dimension=8,
        head_dimension=2,
        feed_forward_dimension=16,
        maximum_sequence_length=8,
        seed=42,
    )

    trainer = TransformerTrainer(
        model=model,
        learning_rate=0.05,
    )

    sequence = [0, 1, 2, 3]
    targets = [1, 2, 3, 4]

    initial_loss = trainer.evaluate(
        sequences=[sequence],
        targets=[targets],
    )

    history = trainer.train(
        sequences=[sequence],
        targets=[targets],
        epochs=100,
    )

    final_loss = history[-1].average_loss

    print("Tiny Transformer Training")
    print("=========================")
    print()

    print(
        "Training sequence:"
    )

    print(
        " ".join(
            TOKENS[token_id]
            for token_id in sequence
        )
    )

    print()

    print(
        f"Initial Loss: {initial_loss:.6f}"
    )

    print(
        f"Final Loss:   {final_loss:.6f}"
    )

    print(
        f"Reduction:    "
        f"{initial_loss - final_loss:.6f}"
    )

    print()

    print("Loss checkpoints:")

    for item in history:
        if (
            item.epoch == 1
            or item.epoch % 10 == 0
            or item.epoch == len(history)
        ):
            print(
                f"epoch {item.epoch:3d}: "
                f"{item.average_loss:.6f}"
            )


if __name__ == "__main__":
    main()
