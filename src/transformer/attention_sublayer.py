from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.layer_norm import LayerNormalization
from src.transformer.residual import add_residual


@dataclass(frozen=True)
class AttentionSublayerResult:
    attention_output: Matrix
    residual_output: Matrix
    normalized_output: Matrix


class AttentionSublayer:
    def __init__(
        self,
        dimension: int,
    ) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension must be positive."
            )

        self._normalization = LayerNormalization(
            dimension=dimension,
        )

    def forward(
        self,
        inputs: Matrix,
        attention_output: Matrix,
    ) -> AttentionSublayerResult:
        residual_output = add_residual(
            inputs,
            attention_output,
        )

        normalized_output = self._normalization.forward(
            residual_output
        )

        return AttentionSublayerResult(
            attention_output=attention_output,
            residual_output=residual_output,
            normalized_output=normalized_output,
        )
