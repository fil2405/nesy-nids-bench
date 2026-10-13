"""Specification of nesybench/models/selection.py.

This is the model-selection rule shared by MLP-CE, LTN-sup and LTN-full: early
stopping on the macro-F1 of the validation part. The tests fail until the
module is written.
"""
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
    """Stands in for a Keras model: returns fixed probabilities."""

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
