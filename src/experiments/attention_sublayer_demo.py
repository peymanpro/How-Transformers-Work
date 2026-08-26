from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix
from src.transformer.attention_sublayer import (
    AttentionSublayer,
)
from src.transformer.multi_head import (
    SyntheticMultiHeadAttention,
)


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

    sublayer = AttentionSublayer(
        dimension=8,
    )

    result = sublayer.forward(
        inputs,
        attention_result.output,
    )

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    print("Transformer Attention Sublayer")
    print("==============================")
    print()
    print("Input:")
    print(inputs.data)
    print()
    print("Multi-Head Attention Output:")
    print(attention_result.output.data)
    print()
    print("After Residual Connection:")
    print(result.residual_output.data)
    print()
    print("After Layer Normalization:")
    print(result.normalized_output.data)


if __name__ == "__main__":
    main()
