from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.synthetic_head import (
    SyntheticAttentionHead,
)
from src.math.matrix import Matrix


@dataclass(frozen=True)
class MultiHeadResult:
    head_outputs: tuple[Matrix, ...]
    concatenated: Matrix
    output: Matrix


class SyntheticMultiHeadAttention:
    def __init__(
        self,
        model_dimension: int,
        head_dimension: int,
        focuses: list[int],
        seed: int = 42,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError(
                "model_dimension must be positive."
            )

        if head_dimension <= 0:
            raise ValueError(
                "head_dimension must be positive."
            )

        if not focuses:
            raise ValueError(
                "At least one head is required."
            )

        if (
            head_dimension * len(focuses)
            != model_dimension
        ):
            raise ValueError(
                "head_dimension * number_of_heads "
                "must equal model_dimension."
            )

        self._model_dimension = model_dimension
        self._head_dimension = head_dimension

        self._heads = tuple(
            SyntheticAttentionHead(
                model_dimension=model_dimension,
                head_dimension=head_dimension,
                focus=focus,
            )
            for focus in focuses
        )

        rng = np.random.default_rng(seed)

        scale = 1.0 / np.sqrt(model_dimension)

        self._output_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(
                    model_dimension,
                    model_dimension,
                ),
            )
        )

    @property
    def number_of_heads(self) -> int:
        return len(self._heads)

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

    def forward(
        self,
        inputs: Matrix,
    ) -> MultiHeadResult:
        if inputs.columns != self._model_dimension:
            raise ValueError(
                "Input dimension does not match model dimension."
            )

        results = tuple(
            head.forward(inputs)
            for head in self._heads
        )

        concatenated_data = np.concatenate(
            [
                result.output.data
                for result in results
            ],
            axis=1,
        )

        concatenated = Matrix(
            concatenated_data
        )

        output = concatenated.multiply(
            self._output_weights
        )

        return MultiHeadResult(
            head_outputs=tuple(
                result.output
                for result in results
            ),
            concatenated=concatenated,
            output=output,
        )
