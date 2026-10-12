from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, class_names) -> dict:
    """Detection metrics; per-class arrays follow the order of class_names."""
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
