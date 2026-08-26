from __future__ import annotations

import numpy as np

from src.transformer.embedding import TokenEmbedding
from src.transformer.positional_encoding import (
    SinusoidalPositionalEncoding,
)

TOKENS = [
    "the",
    "cat",
    "drinks",
    "milk",
]


def main() -> None:
    embedding = TokenEmbedding(
        vocabulary_size=len(TOKENS),
        embedding_dimension=6,
        seed=42,
    )

    positional = SinusoidalPositionalEncoding(
        maximum_sequence_length=len(TOKENS),
        embedding_dimension=6,
    )

    token_embeddings = embedding.encode(
        [0, 1, 2, 3]
    )

    positional_values = positional.encode(4)

    transformer_input = positional.add_to(
        token_embeddings
    )

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    print("HowTransformersWork")
    print("===================")
    print()
    print("Tokens:")
    print(TOKENS)
    print()
    print("Token Embeddings:")
    print(token_embeddings.data)
    print()
    print("Positional Encoding:")
    print(positional_values.data)
    print()
    print("Transformer Input:")
    print(transformer_input.data)


if __name__ == "__main__":
    main()
