import pytest

from src.transformer.language_model import (
    TinyTransformerLanguageModel,
)
from src.transformer.training import (
    TransformerTrainer,
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


def test_training_should_return_one_result_per_epoch() -> None:
    trainer = TransformerTrainer(
        model=create_model(),
        learning_rate=0.05,
    )

    history = trainer.train(
        sequences=[
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        targets=[
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        epochs=5,
    )

    assert len(history) == 5
    assert [
        item.epoch
        for item in history
    ] == [1, 2, 3, 4, 5]


def test_training_should_reduce_loss() -> None:
    trainer = TransformerTrainer(
        model=create_model(),
        learning_rate=0.05,
    )

    history = trainer.train(
        sequences=[
            [0, 1, 2, 3],
        ],
        targets=[
            [1, 2, 3, 4],
        ],
        epochs=100,
    )

    assert (
        history[-1].average_loss
        < history[0].average_loss
    )


def test_training_should_reject_empty_dataset() -> None:
    trainer = TransformerTrainer(
        model=create_model(),
        learning_rate=0.05,
    )

    with pytest.raises(ValueError):
        trainer.train(
            sequences=[],
            targets=[],
            epochs=5,
        )


def test_training_should_reject_mismatched_dataset_lengths() -> None:
    trainer = TransformerTrainer(
        model=create_model(),
        learning_rate=0.05,
    )

    with pytest.raises(ValueError):
        trainer.train(
            sequences=[[0, 1]],
            targets=[[1]],
            epochs=5,
        )


def test_training_should_reject_mismatched_sequence_target_lengths() -> None:
    trainer = TransformerTrainer(
        model=create_model(),
        learning_rate=0.05,
    )

    with pytest.raises(ValueError):
        trainer.train(
            sequences=[[0, 1, 2]],
            targets=[[1, 2]],
            epochs=5,
        )
