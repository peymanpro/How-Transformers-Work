from __future__ import annotations

from src.math.matrix import Matrix


def backward_residual(
    output_gradient: Matrix,
) -> tuple[Matrix, Matrix]:
    if output_gradient.rows <= 0:
        raise ValueError(
            "Output gradient cannot be empty."
        )

    input_gradient = Matrix(
        output_gradient.data
    )

    sublayer_gradient = Matrix(
        output_gradient.data
    )

    return (
        input_gradient,
        sublayer_gradient,
    )


def combine_residual_gradients(
    direct_gradient: Matrix,
    sublayer_input_gradient: Matrix,
) -> Matrix:
    if direct_gradient.shape != sublayer_input_gradient.shape:
        raise ValueError(
            "Residual gradient shapes must match."
        )

    return direct_gradient.add(
        sublayer_input_gradient
    )
