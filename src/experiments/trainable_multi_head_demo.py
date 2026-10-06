from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix
from src.transformer.trainable_multi_head import (
    TrainableMultiHeadAttention,
)


def main() -> None:
    inputs = Matrix.from_values(
        [
            [0.2, 0.5, 0.2, 0.8],
            [0.5, 0.1, 0.7, 0.3],
            [0.6, 0.4, 0.9, 0.2],
        ]
    )

    attention = TrainableMultiHeadAttention(
        model_dimension=4,
        head_dimension=2,
        seed=42,
        causal=True,
    )

    result = attention.forward(inputs)

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    print("Trainable Multi-Head Self-Attention")
    print("===================================")
    print()
    print(f"Number of heads: {attention.number_of_heads}")
    print(f"Head dimension:  {attention.head_dimension}")
    print(f"Causal:         {attention.causal}")
    print()

    for index, weights in enumerate(result.weights, start=1):
        print(f"Head {index} attention weights:")
        print(weights.data)
        print()

    print("Concatenated head output:")
    print(result.concatenated.data)
    print()

    print("Final output after W0:")
    print(result.output.data)


if __name__ == "__main__":
    main()
