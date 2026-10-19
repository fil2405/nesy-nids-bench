import numpy as np

from nesybench.metrics import compute_metrics, pr_auc_per_class, operator_metrics

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


def test_pr_auc_is_one_for_a_perfect_ranking():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert np.allclose(pr_auc_per_class(y, np.eye(3)[y], ["a", "b", "c"]), 1.0)


def test_pr_auc_ignores_the_decision_threshold():
    y = np.array([0, 0, 1, 1])
    proba = np.array([[.9, .1], [.6, .4], [.55, .45], [.2, .8]])
    assert proba.argmax(axis=1)[2] != y[2]  # argmax gets one wrong
    assert np.allclose(pr_auc_per_class(y, proba, ["a", "b"]), 1.0)


def test_pr_auc_of_a_class_without_examples_is_nan():
    y = np.array([0, 1, 0, 1])
    out = pr_auc_per_class(y, np.full((4, 3), 1 / 3), ["a", "b", "c"])
    assert np.isnan(out[2])
    assert not np.isnan(out[:2]).any()


def _scored(attack, normal):
    y = np.array([1] * len(attack) + [0] * len(normal))
    s = np.array(attack + normal)
    return y, np.column_stack([1 - s, s])


def test_operator_metrics_on_a_hand_computed_ranking():
    y, proba = _scored([.9, .8, .7, .3], [.6, .2, .1, .05])
    m = operator_metrics(y, proba, tpr_levels=(0.75, 1.0), fpr_levels=(0.2, 0.25))
    assert m["fpr_at_tpr"] == {"0.75": 0.0, "1": 0.25}
    assert m["tpr_at_fpr"] == {"0.2": 0.75, "0.25": 1.0}


def test_operator_metrics_ignore_which_attack_class():
    y = np.array([0, 0, 1, 2])
    proba = np.array([[.9, .05, .05], [.8, .1, .1], [.1, .1, .8], [.1, .8, .1]])
    m = operator_metrics(y, proba, tpr_levels=(1.0,), fpr_levels=(0.0,))
    assert m["fpr_at_tpr"]["1"] == 0.0
    assert m["tpr_at_fpr"]["0"] == 1.0


def test_operator_metrics_are_nan_without_both_sides():
    m = operator_metrics(np.array([1, 1]), np.array([[.2, .8], [.1, .9]]))
    assert all(np.isnan(v) for v in m["fpr_at_tpr"].values())
    assert all(np.isnan(v) for v in m["tpr_at_fpr"].values())

def test_operator_metrics_keep_the_order_of_tiny_benign_probabilities():
    y = np.array([1, 1, 0, 0])
    p_normal = np.array([1e-11, 1e-12, 1e-9, 0.5], dtype=np.float32)
    proba = np.column_stack([p_normal, 1 - p_normal]).astype(np.float32)
    m = operator_metrics(y, proba, tpr_levels=(1.0,), fpr_levels=(0.5,))
    assert m["fpr_at_tpr"]["1"] == 0.0
