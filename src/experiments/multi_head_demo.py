from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix
from src.transformer.multi_head import (
    SyntheticMultiHeadAttention,
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
            [1.0, 0.0, 0.5, 0.2, 0.8, 0.1, 0.3, 0.7],
            [0.2, 0.9, 0.1, 0.4, 0.3, 0.8, 0.5, 0.6],
            [0.7, 0.3, 0.9, 0.2, 0.4, 0.6, 0.8, 0.1],
            [0.5, 0.4, 0.2, 0.9, 0.7, 0.3, 0.6, 0.8],
        ]
    )

    attention = SyntheticMultiHeadAttention(
        model_dimension=8,
        head_dimension=2,
        focuses=[0, 1, 2, 3],
        seed=42,
    )

    result = attention.forward(
        inputs
    )

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    print("Synthetic Multi-Head Self-Attention")
    print("===================================")
    print()
    print("Tokens:")
    print(TOKENS)
    print()

    for index, head_output in enumerate(
        result.head_outputs,
        start=1,
    ):
        print(f"Head {index} Output:")
        print(head_output.data)
        print()

    print("Concatenated Heads:")
    print(result.concatenated.data)
    print()

    print("After Output Projection W_O:")
    print(result.output.data)


if __name__ == "__main__":
    main()
