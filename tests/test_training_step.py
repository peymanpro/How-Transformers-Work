from src.transformer.language_model import (
    TinyTransformerLanguageModel,
)
from src.transformer.training_step import (
    TransformerTrainingStep,
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


def test_training_step_should_return_loss() -> None:
    model = create_model()

    result = TransformerTrainingStep(
        learning_rate=0.05,
    ).run(
        model=model,
        token_ids=[0, 1, 2, 3],
        targets=[1, 2, 3, 4],
    )

    assert result.loss > 0.0


def test_repeated_training_steps_should_change_output() -> None:
    model = create_model()

    before = model.forward(
        [0, 1, 2, 3]
    ).logits.values.data

    TransformerTrainingStep(
        learning_rate=0.05,
    ).run(
        model=model,
        token_ids=[0, 1, 2, 3],
        targets=[1, 2, 3, 4],
    )

    after = model.forward(
        [0, 1, 2, 3]
    ).logits.values.data

    assert not (
        before == after
    ).all()


def test_multiple_training_steps_should_not_produce_non_finite_loss() -> None:
    model = create_model()

    step = TransformerTrainingStep(
        learning_rate=0.05,
    )

    for _ in range(10):
        result = step.run(
            model=model,
            token_ids=[0, 1, 2, 3],
            targets=[1, 2, 3, 4],
        )

        assert result.loss == result.loss
        assert result.loss < float("inf")
