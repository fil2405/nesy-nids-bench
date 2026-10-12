import numpy as np

from nesybench.metrics import compute_metrics

NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 0, 1])
    m = compute_metrics(y, y, NAMES)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_missing_class_keeps_shape():
    m = compute_metrics(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]), NAMES)
    assert len(m["f1"]) == 5
    assert m["confusion_matrix"].shape == (5, 5)


def test_accuracy_hides_rare_class():
    y_true = np.array([0] * 95 + [4] * 5)
    y_pred = np.zeros(100, dtype=int)
    m = compute_metrics(y_true, y_pred, NAMES)
    assert m["accuracy"] == 0.95
    assert m["macro_f1"] < 0.25


def test_confusion_matrix_rows_are_true_classes():
    m = compute_metrics(np.array([0, 1]), np.array([1, 1]), NAMES)
    assert m["confusion_matrix"][0, 1] == 1


def test_number_of_classes_comes_from_names():
    m = compute_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]), ["a", "b", "c"])
    assert m["confusion_matrix"].shape == (3, 3)
    assert m["class_names"] == ["a", "b", "c"]
