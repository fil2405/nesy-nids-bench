from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    average_precision_score,
    roc_curve
)
import numpy as np


def compute_metrics(y_true, y_pred, class_names) -> dict:
    """Per-class arrays follow class_names."""
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "class_names": list(class_names),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def pr_auc_per_class(y_true, proba, class_names):
    """One-vs-rest average precision; nan for a class with no examples."""
    y_true = np.asarray(y_true)
    out = np.full(len(class_names), np.nan)
    for c in range(len(class_names)):
        positive = y_true == c
        if positive.any():
            out[c] = average_precision_score(positive, proba[:, c])
    return out


def operator_metrics(y_true, proba, benign=0,
                     tpr_levels=(0.90, 0.95, 0.99), fpr_levels=(1e-3, 1e-2)):
    """Attack vs benign, scored by -P(benign): FPR at fixed TPR, TPR at fixed FPR.

    1 - P(benign) rounds to 1.0 in float32 once P(benign) < ~6e-8 and creates ties.
    """
    y_attack = np.asarray(y_true) != benign
    if y_attack.all() or not y_attack.any():
        return {"fpr_at_tpr": {f"{t:g}": float("nan") for t in tpr_levels},
                "tpr_at_fpr": {f"{f:g}": float("nan") for f in fpr_levels}}
    fpr, tpr, _ = roc_curve(y_attack, -proba[:, benign])
    return {
        "fpr_at_tpr": {f"{t:g}": float(fpr[np.searchsorted(tpr, t)]) for t in tpr_levels},
        "tpr_at_fpr": {f"{f:g}": float(tpr[np.searchsorted(fpr, f, side="right") - 1])
                       for f in fpr_levels},
    }