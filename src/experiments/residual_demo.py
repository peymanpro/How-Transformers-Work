from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix
from src.transformer.multi_head import (
    SyntheticMultiHeadAttention,
)
from src.transformer.residual import add_residual


def main() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            [0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.7, 0.5],
            [0.6, 0.3, 0.9, 0.2, 0.5, 0.8, 0.1, 0.4],
        ]
    )

    attention = SyntheticMultiHeadAttention(
        model_dimension=8,
        head_dimension=2,
        focuses=[0, 1, 2, 2],
        seed=42,
    )

    attention_result = attention.forward(
        inputs
    )

    residual_output = add_residual(
        inputs,
        attention_result.output,
    )

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    print("Residual Connection")
    print("===================")
    print()
    print("Input:")
    print(inputs.data)
    print()
    print("Multi-Head Output:")
    print(attention_result.output.data)
    print()
    print("Residual Output = Input + Multi-Head Output:")
    print(residual_output.data)


if __name__ == "__main__":
    main()
