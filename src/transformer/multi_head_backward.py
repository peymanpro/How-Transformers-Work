from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix
from src.transformer.trainable_multi_head import (
    TrainableMultiHeadAttention,
    TrainableMultiHeadResult,
)


@dataclass(frozen=True)
class TrainableMultiHeadGradients:
    inputs: Matrix
    query_weights: tuple[Matrix, ...]
    key_weights: tuple[Matrix, ...]
    value_weights: tuple[Matrix, ...]
    output_weights: Matrix


class TrainableMultiHeadBackward:
    def backward(
        self,
        attention: TrainableMultiHeadAttention,
        forward_result: TrainableMultiHeadResult,
        inputs: Matrix,
        output_gradient: Matrix,
    ) -> TrainableMultiHeadGradients:
        if forward_result.output.shape != output_gradient.shape:
            raise ValueError(
                "Output gradient must match multi-head output."
            )

        if inputs.columns != attention.model_dimension:
            raise ValueError(
                "Input dimension does not match attention dimension."
            )

        if len(forward_result.weights) != attention.number_of_heads:
            raise ValueError(
                "Forward result head count does not match attention."
            )

        d_output_weights = Matrix(
            forward_result.concatenated.data.T
            @ output_gradient.data
        )

        d_concatenated = (
            output_gradient.data
            @ attention.output_weights.data.T
        )

        d_inputs = np.zeros_like(inputs.data)
        d_query_weights: list[Matrix] = []
        d_key_weights: list[Matrix] = []
        d_value_weights: list[Matrix] = []

        head_dimension = attention.head_dimension
        scale = np.sqrt(head_dimension)

        for head_index in range(attention.number_of_heads):
            start = head_index * head_dimension
            end = start + head_dimension

            head_output_gradient = d_concatenated[:, start:end]

            query = forward_result.queries[head_index]
            key = forward_result.keys[head_index]
            value = forward_result.values[head_index]
            weights = forward_result.weights[head_index]

            d_value = weights.data.T @ head_output_gradient
            d_weights = head_output_gradient @ value.data.T

            d_scores = weights.data * (
                d_weights
                - np.sum(
                    d_weights * weights.data,
                    axis=1,
                    keepdims=True,
                )
            )

            if attention.causal:
                future = np.triu(
                    np.ones(d_scores.shape, dtype=bool),
                    k=1,
                )
                d_scores[future] = 0.0

            d_scores /= scale

            d_query = d_scores @ key.data
            d_key = d_scores.T @ query.data

            query_weight = attention.query_weights[head_index]
            key_weight = attention.key_weights[head_index]
            value_weight = attention.value_weights[head_index]

            d_query_weights.append(
                Matrix(inputs.data.T @ d_query)
            )
            d_key_weights.append(
                Matrix(inputs.data.T @ d_key)
            )
            d_value_weights.append(
                Matrix(inputs.data.T @ d_value)
            )

            d_inputs += (
                d_query @ query_weight.data.T
                + d_key @ key_weight.data.T
                + d_value @ value_weight.data.T
            )

        return TrainableMultiHeadGradients(
            inputs=Matrix(d_inputs),
            query_weights=tuple(d_query_weights),
            key_weights=tuple(d_key_weights),
            value_weights=tuple(d_value_weights),
            output_weights=d_output_weights,
        )
