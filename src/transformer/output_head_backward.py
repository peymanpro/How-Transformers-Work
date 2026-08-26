from __future__ import annotations

from dataclasses import dataclass

from src.math.matrix import Matrix
from src.transformer.output_head import TokenLogits


@dataclass(frozen=True)
class VocabularyProjectionGradients:
    hidden_states: Matrix
    weights: Matrix
    bias: Matrix


class VocabularyProjectionBackward:
    def backward(
        self,
        hidden_states: Matrix,
        weights: Matrix,
        logits: TokenLogits,
        output_gradient: Matrix,
    ) -> VocabularyProjectionGradients:
        if hidden_states.rows != output_gradient.rows:
            raise ValueError(
                "Hidden states and output gradient must have "
                "the same number of rows."
            )

        if hidden_states.columns != weights.rows:
            raise ValueError(
                "Hidden-state dimension must match weight rows."
            )

        if weights.columns != logits.values.columns:
            raise ValueError(
                "Weight columns must match vocabulary size."
            )

        if logits.values.shape != output_gradient.shape:
            raise ValueError(
                "Logits and output gradient must have the same shape."
            )

        gradient_hidden = Matrix(
            output_gradient.data
            @ weights.data.T
        )

        gradient_weights = Matrix(
            hidden_states.data.T
            @ output_gradient.data
        )

        gradient_bias = Matrix(
            output_gradient.data.sum(
                axis=0,
                keepdims=True,
            )
        )

        return VocabularyProjectionGradients(
            hidden_states=gradient_hidden,
            weights=gradient_weights,
            bias=gradient_bias,
        )
