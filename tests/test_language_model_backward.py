import numpy as np
import pytest

from src.math.matrix import Matrix
from src.transformer.cross_entropy import (
    CrossEntropyFromLogits,
)
from src.transformer.language_model import (
    TinyTransformerLanguageModel,
)
from src.transformer.language_model_backward import (
    TinyTransformerBackward,
)


def create_model() -> TinyTransformerLanguageModel:
    return TinyTransformerLanguageModel(
        vocabulary_size=6,
        model_dimension=8,
        head_dimension=2,
        feed_forward_dimension=16,
        maximum_sequence_length=8,
        seed=42,
    )


def test_language_model_backward_should_return_gradient_shapes() -> None:
    model = create_model()

    token_ids = [0, 1, 2, 3]

    forward_result = model.forward(
        token_ids
    )

    targets = [1, 2, 3, 4]

    loss = CrossEntropyFromLogits()

    output_gradient = loss.gradient(
        forward_result.logits.values,
        targets,
    )

    gradients = TinyTransformerBackward().backward(
        model=model,
        token_ids=token_ids,
        forward_result=forward_result,
        output_gradient=output_gradient,
    )

    assert gradients.token_embedding.shape == (
        6,
        8,
    )

    assert gradients.vocabulary_weights.shape == (
        8,
        6,
    )

    assert gradients.vocabulary_bias.shape == (
        1,
        6,
    )

    assert gradients.decoder.input.shape == (
        4,
        8,
    )


def test_language_model_backward_should_produce_finite_gradients() -> None:
    model = create_model()

    token_ids = [0, 1, 2, 3]

    forward_result = model.forward(
        token_ids
    )

    output_gradient = (
        CrossEntropyFromLogits().gradient(
            forward_result.logits.values,
            [1, 2, 3, 4],
        )
    )

    gradients = TinyTransformerBackward().backward(
        model=model,
        token_ids=token_ids,
        forward_result=forward_result,
        output_gradient=output_gradient,
    )

    assert np.all(
        np.isfinite(
            gradients.token_embedding.data
        )
    )

    assert np.all(
        np.isfinite(
            gradients.vocabulary_weights.data
        )
    )

    assert np.all(
        np.isfinite(
            gradients.decoder.input.data
        )
    )


def test_language_model_backward_should_reject_wrong_gradient_shape() -> None:
    model = create_model()

    forward_result = model.forward(
        [0, 1, 2, 3]
    )

    with pytest.raises(ValueError):
        TinyTransformerBackward().backward(
            model=model,
            token_ids=[0, 1, 2, 3],
            forward_result=forward_result,
            output_gradient=Matrix.from_values(
                [[1.0, 2.0]]
            ),
        )
def test_language_model_vocabulary_gradient_should_match_numerical_gradient() -> None:
    model = create_model()

    token_ids = [0, 1, 2, 3]
    targets = [1, 2, 3, 4]

    forward_result = model.forward(
        token_ids
    )

    loss = CrossEntropyFromLogits()

    output_gradient = loss.gradient(
        forward_result.logits.values,
        targets,
    )

    analytic = TinyTransformerBackward().backward(
        model=model,
        token_ids=token_ids,
        forward_result=forward_result,
        output_gradient=output_gradient,
    )

    epsilon = 1e-6

    numerical = np.zeros_like(
        model.vocabulary_weights.data
    )

    def scalar_loss(
        weight_values: np.ndarray,
    ) -> float:
        original = model.vocabulary_weights

        # Update the model's vocabulary weights only
        # for the purpose of the numerical probe.
        model._vocabulary_projection._weights = Matrix(
            weight_values
        )

        current = model.forward(
            token_ids
        )

        value = loss.loss(
            current.logits.values,
            targets,
        )

        model._vocabulary_projection._weights = original

        return value

    # Probe a small representative subset rather than
    # all parameters. This keeps the numerical test fast.
    rows_to_check = min(
        model.vocabulary_weights.rows,
        2,
    )

    columns_to_check = min(
        model.vocabulary_weights.columns,
        3,
    )

    for row in range(rows_to_check):
        for column in range(columns_to_check):
            plus = model.vocabulary_weights.data.copy()
            minus = model.vocabulary_weights.data.copy()

            plus[row, column] += epsilon
            minus[row, column] -= epsilon

            numerical[row, column] = (
                scalar_loss(plus)
                - scalar_loss(minus)
            ) / (
                2.0 * epsilon
            )

    np.testing.assert_allclose(
        analytic.vocabulary_weights.data[
            :rows_to_check,
            :columns_to_check,
        ],
        numerical[
            :rows_to_check,
            :columns_to_check,
        ],
        rtol=1e-4,
        atol=1e-5,
    )
