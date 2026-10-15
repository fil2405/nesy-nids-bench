"""Early stopping on validation macro-F1."""
import numpy as np
import pytest

from nesybench.data.preprocess import Splits
from nesybench.models.selection import EarlyStopper, validation_macro_f1


def _splits(y_val):
    empty = np.zeros((0, 2))
    no_labels = np.zeros(0, dtype=int)
    return Splits(
        X_train=empty,
        X_val=np.arange(2 * len(y_val), dtype=float).reshape(-1, 2),
        X_test=empty,
        y_train=no_labels,
        y_val=np.asarray(y_val),
        y_test=no_labels,
        transformer=None,
        feature_names=["a", "b"],
        class_names=["A", "B", "C"],
    )


class FakeModel:


    def __init__(self, proba):
        self.proba = np.asarray(proba)
        self.seen = None

    def predict(self, X, *args, **kwargs):
        self.seen = X
        return self.proba


Y_VAL = [0, 0, 1, 1, 2, 2]


def test_perfect_probabilities_give_macro_f1_one():
    model = FakeModel(np.eye(3)[Y_VAL])
    assert validation_macro_f1(model, _splits(Y_VAL)) == pytest.approx(1.0)


def test_macro_f1_counts_every_class_equally():
    # everything predicted as class 0: F1 is 0.5 for A and 0 for B and C
    model = FakeModel(np.eye(3)[[0] * 6])
    assert validation_macro_f1(model, _splits(Y_VAL)) == pytest.approx(0.5 / 3)


def test_the_model_sees_the_validation_part():
    splits = _splits(Y_VAL)
    model = FakeModel(np.eye(3)[Y_VAL])
    validation_macro_f1(model, splits)
    assert np.array_equal(model.seen, splits.X_val)


def test_best_epoch_and_weights_are_those_of_the_best_score():
    stopper = EarlyStopper(patience=10)
    for epoch, score in enumerate((0.1, 0.5, 0.4, 0.3), start=1):
        stopper.update(score, f"weights{epoch}")
    assert stopper.best_epoch == 2  # epochs are counted from 1
    assert stopper.best_score == 0.5
    assert stopper.best_weights == "weights2"


def test_stops_after_patience_epochs_without_improvement():
    stopper = EarlyStopper(patience=2)
    assert [stopper.update(score, "w") for score in (0.5, 0.4, 0.3)] == [False, False, True]


def test_an_improvement_resets_the_wait():
    stopper = EarlyStopper(patience=2)
    assert [stopper.update(score, "w") for score in (0.5, 0.4, 0.6, 0.5)] == [False] * 4
    assert stopper.best_epoch == 3


def test_a_tie_is_not_an_improvement():
    stopper = EarlyStopper(patience=5)
    stopper.update(0.5, "first")
    stopper.update(0.5, "second")
    assert stopper.best_epoch == 1
    assert stopper.best_weights == "first"


def test_window_one_is_the_raw_score():
    stopper = EarlyStopper(patience=5, window=1)
    for score in (0.1, 0.9, 0.2):
        stopper.update(score, "w")
    assert stopper.best_epoch == 2
    assert stopper.best_score == stopper.best_raw_score == 0.9


def test_a_lucky_epoch_is_not_picked_with_a_window():
    # raw scores peak at epoch 3 (0.9), but epochs 5-6 are steadily better
    scores = (0.5, 0.5, 0.9, 0.5, 0.8, 0.8)
    raw = EarlyStopper(patience=10, window=1)
    smooth = EarlyStopper(patience=10, window=2)
    for epoch, score in enumerate(scores, start=1):
        raw.update(score, f"w{epoch}")
        smooth.update(score, f"w{epoch}")
    assert raw.best_epoch == 3
    assert smooth.best_epoch == 6
    assert smooth.best_score == pytest.approx(0.8)
    assert smooth.best_raw_score == 0.8
    assert smooth.best_weights == "w6"


def test_a_lucky_first_epoch_is_not_a_candidate():
    # seed 6 of the first 10-seed run: epoch 1 scored 0.857, later epochs ~0.82
    stopper = EarlyStopper(patience=10, window=3)
    for epoch, score in enumerate((0.857, 0.80, 0.83, 0.82, 0.84), start=1):
        stopper.update(score, f"w{epoch}")
    assert stopper.best_epoch >= 3


def test_patience_starts_when_the_window_is_full():
    stopper = EarlyStopper(patience=1, window=3)
    assert [stopper.update(s, "w") for s in (0.9, 0.1, 0.5, 0.4)] == [False, False, False, True]
    assert stopper.best_epoch == 3


def test_close_keeps_the_last_epoch_if_the_window_never_filled():
    stopper = EarlyStopper(patience=5, window=5)
    stopper.update(0.4, "w1")
    stopper.update(0.6, "w2")
    stopper.close("w2")
    assert stopper.best_epoch == 2
    assert stopper.best_weights == "w2"
    assert stopper.best_raw_score == 0.6
    assert stopper.best_score == pytest.approx(0.5)


def test_close_does_nothing_once_an_epoch_was_chosen():
    stopper = EarlyStopper(patience=5, window=1)
    stopper.update(0.9, "w1")
    stopper.update(0.1, "w2")
    stopper.close("w2")
    assert stopper.best_epoch == 1
    assert stopper.best_weights == "w1"


def test_window_below_one_is_an_error():
    with pytest.raises(ValueError, match="window"):
        EarlyStopper(patience=5, window=0)
