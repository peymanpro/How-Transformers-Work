from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix
from src.transformer.encoder_block import (
    TransformerEncoderBlock,
)

TOKENS = [
    "the",
    "cat",
    "drinks",
    "milk",
]


def main() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            [0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.7, 0.5],
            [0.6, 0.3, 0.9, 0.2, 0.5, 0.8, 0.1, 0.4],
            [0.5, 0.4, 0.2, 0.9, 0.7, 0.3, 0.6, 0.8],
        ]
    )

    block = TransformerEncoderBlock(
        model_dimension=8,
        head_dimension=2,
        head_focuses=[0, 1, 2, 2],
        feed_forward_dimension=16,
        seed=42,
    )

    result = block.forward(
        inputs
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

    print("1. Input:")
    print(result.input.data)
    print()

    print("2. Multi-Head Attention Output:")
    print(result.attention.output.data)
    print()

    print("3. After Residual + LayerNorm:")
    print(result.after_attention_sublayer.data)
    print()

    print("4. Feed-Forward Output:")
    print(result.feed_forward_output.data)
    print()

    print("5. Final Transformer Block Output:")
    print(result.output.data)


if __name__ == "__main__":
    main()
