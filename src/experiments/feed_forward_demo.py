from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix
from src.transformer.feed_forward import (
    FeedForwardNetwork,
)


def main() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 0.5, 0.2, 0.8],
            [0.2, 0.9, 0.4, 0.1],
            [0.6, 0.3, 0.9, 0.2],
        ]
    )

    network = FeedForwardNetwork(
        model_dimension=4,
        hidden_dimension=8,
        seed=42,
    )

    result = network.forward(inputs)

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    print("Transformer Feed-Forward Network")
    print("================================")
    print()
    print("Input:")
    print(inputs.data)
    print()
    print("Hidden Representation:")
    print(result.hidden.data)
    print()
    print("FFN Output:")
    print(result.output.data)


if __name__ == "__main__":
    main()
