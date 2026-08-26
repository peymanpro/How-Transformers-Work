from __future__ import annotations

from src.math.matrix import Matrix


def add_residual(
    inputs: Matrix,
    sublayer_output: Matrix,
) -> Matrix:
    if inputs.shape != sublayer_output.shape:
        raise ValueError(
            "Residual inputs and sublayer output must have the same shape."
        )

    return inputs.add(sublayer_output)
