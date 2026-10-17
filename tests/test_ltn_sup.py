import numpy as np
import pytest

from nesybench.data.preprocess import Splits
from nesybench.metrics import compute_metrics
from nesybench.models import get_model

CLASS_NAMES = ["A", "B", "C"]
CENTERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _part(rng, counts, spread):
    y = np.repeat(np.arange(3), counts)
    X = CENTERS[y] * spread + rng.normal(size=(len(y), 2))
    return X.astype("float32"), y


def _splits(spread=6.0, train_counts=(200, 200, 200), test=True):
    rng = np.random.default_rng(0)
    X_train, y_train = _part(rng, train_counts, spread)
    X_val, y_val = _part(rng, (100, 100, 100), spread)
    X_test, y_test = _part(rng, (100, 100, 100), spread) if test else (None, None)
    return Splits(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train,
                  y_val=y_val, y_test=y_test, transformer=None,
                  feature_names=["a", "b"], class_names=CLASS_NAMES)


def _config(**training):
    return {
        "seed": 0,
        "network": {"hidden_units": [16], "dropout": 0.0, "activation": "relu"},
        "training": {"optimizer": "adam", "learning_rate": 1e-2, "batch_size": 32,
                     "max_epochs": 30, "early_stopping_patience": 5,
                     "selection_window": 1, "sampling_power": 0.0,
                     "min_per_class": 2, **training},
    }


def test_registered():
    assert get_model("ltn_sup").__name__ == "LTNSup"


def test_learns_and_reports():
    splits = _splits()
    model = get_model("ltn_sup")()
    info = model.fit(splits, _config())
    n = info["epochs_run"]
    assert 1 <= info["best_epoch"] <= n
    assert len(info["val_macro_f1_history"]) == n
    assert len(info["train_loss_history"]) == n
    assert len(info["epoch_seconds"]) == n
    proba = model.predict_proba(splits, "test")
    assert proba.shape == (len(splits.y_test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba.argmax(axis=1) == splits.y_test).mean() > 0.9


def test_weights_of_the_best_epoch_are_restored():
    splits = _splits(spread=1.0)
    model = get_model("ltn_sup")()
    info = model.fit(splits, _config(max_epochs=12, early_stopping_patience=100))
    proba = model.predict_proba(splits, "val")
    score = compute_metrics(splits.y_val, proba.argmax(axis=1), CLASS_NAMES)["macro_f1"]
    assert score == pytest.approx(info["best_val_macro_f1"])


def test_a_rare_class_gives_finite_losses():
    splits = _splits(train_counts=(300, 300, 3))
    info = get_model("ltn_sup")().fit(splits, _config(max_epochs=5))
    assert np.all(np.isfinite(info["train_loss_history"]))


def test_class_weights_are_refused():
    with pytest.raises(ValueError, match="class weights"):
        get_model("ltn_sup")().fit(_splits(), _config(imbalance="class_weight"))


def test_fit_never_reads_the_test_part():
    get_model("ltn_sup")().fit(_splits(test=False), _config(max_epochs=3))


def test_same_seed_same_history():
    a = get_model("ltn_sup")().fit(_splits(), _config(max_epochs=6))
    b = get_model("ltn_sup")().fit(_splits(), _config(max_epochs=6))
    assert a["train_loss_history"] == pytest.approx(b["train_loss_history"], rel=1e-4)
    assert a["val_macro_f1_history"] == pytest.approx(b["val_macro_f1_history"], abs=1e-3)