from __future__ import annotations

import numpy as np

from src.transformer.language_model import (
    TinyTransformerLanguageModel,
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

    token_ids = [0, 1, 2, 3]

    result = model.forward(
        token_ids
    )

    logits = result.logits.values.data

    last_position = logits[-1]

    shifted = (
        last_position
        - np.max(last_position)
    )

    probabilities = np.exp(
        shifted
    )

    probabilities /= np.sum(
        probabilities
    )

    ranking = np.argsort(
        probabilities
    )[::-1]

    print("Tiny Transformer Language Model")
    print("===============================")
    print()
    print("Input:")
    print(
        " ".join(
            TOKENS[token_id]
            for token_id in token_ids
        )
    )
    print()
    print("Next-token probabilities:")
    print()

    for index in ranking:
        print(
            f"{TOKENS[index]:>8} : "
            f"{probabilities[index]:.6f}"
        )

    print()
    print(
        "Predicted next token:",
        TOKENS[int(ranking[0])],
    )


if __name__ == "__main__":
    main()
