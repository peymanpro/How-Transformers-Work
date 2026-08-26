from src.transformer.language_model import (
    TinyTransformerLanguageModel,
)
from src.transformer.prediction import (
    NextTokenPredictor,
)
from src.transformer.training import (
    TransformerTrainer,
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


def test_training_should_increase_target_probability() -> None:
    model = create_model()

    sequence = [0, 1, 2, 3]
    targets = [1, 2, 3, 4]

    predictor = NextTokenPredictor()

    before = model.forward(
        sequence
    )

    before_probabilities = (
        predictor.probability_of(
            before.logits.values,
            targets,
        )
    )

    TransformerTrainer(
        model=model,
        learning_rate=0.05,
    ).train(
        sequences=[sequence],
        targets=[targets],
        epochs=100,
    )

    after = model.forward(
        sequence
    )

    after_probabilities = (
        predictor.probability_of(
            after.logits.values,
            targets,
        )
    )

    assert sum(after_probabilities) > sum(
        before_probabilities
    )


def test_training_should_improve_mean_target_probability() -> None:
    model = create_model()

    sequence = [0, 1, 2, 3]
    targets = [1, 2, 3, 4]

    predictor = NextTokenPredictor()

    before = model.forward(
        sequence
    )

    before_probabilities = (
        predictor.probability_of(
            before.logits.values,
            targets,
        )
    )

    TransformerTrainer(
        model=model,
        learning_rate=0.05,
    ).train(
        sequences=[sequence],
        targets=[targets],
        epochs=100,
    )

    after = model.forward(
        sequence
    )

    after_probabilities = (
        predictor.probability_of(
            after.logits.values,
            targets,
        )
    )

    before_mean = sum(
        before_probabilities
    ) / len(before_probabilities)

    after_mean = sum(
        after_probabilities
    ) / len(after_probabilities)

    assert after_mean > before_mean
