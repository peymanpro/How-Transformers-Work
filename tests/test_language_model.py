import numpy as np

from src.transformer.language_model import (
    TinyTransformerLanguageModel,
)


def create_model() -> TinyTransformerLanguageModel:
    return TinyTransformerLanguageModel(
        vocabulary_size=6,
        model_dimension=8,
        head_dimension=2,
        head_focuses=[0, 1, 2, 2],
        feed_forward_dimension=16,
        maximum_sequence_length=8,
        seed=42,
    )


def test_language_model_should_preserve_sequence_length() -> None:
    result = create_model().forward(
        [0, 1, 2, 3]
    )

    assert result.embeddings.shape == (4, 8)
    assert result.decoder_output.shape == (4, 8)
    assert result.logits.values.shape == (4, 6)


def test_language_model_should_produce_finite_logits() -> None:
    result = create_model().forward(
        [0, 1, 2, 3]
    )

    assert np.all(
        np.isfinite(
            result.logits.values.data
        )
    )


def test_language_model_should_be_deterministic() -> None:
    first = create_model().forward(
        [0, 1, 2, 3]
    )

    second = create_model().forward(
        [0, 1, 2, 3]
    )

    np.testing.assert_allclose(
        first.logits.values.data,
        second.logits.values.data,
    )
