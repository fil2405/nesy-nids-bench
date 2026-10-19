"""MLP-CE on synthetic Gaussian blobs."""
import numpy as np
import pytest

import nesybench  # sets TF_USE_LEGACY_KERAS before TensorFlow is imported

pytest.importorskip("tensorflow")

from nesybench.data.preprocess import Splits  # noqa: E402
from nesybench.metrics import compute_metrics  # noqa: E402
from nesybench.models import get_model  # noqa: E402

CLASS_NAMES = ["A", "B", "C"]
CENTERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _part(rng, n_per_class, spread):
    y = np.repeat(np.arange(3), n_per_class)
    X = CENTERS[y] * spread + rng.normal(size=(len(y), 2))
    return X.astype("float32"), y


def _splits(spread, test=True):
    rng = np.random.default_rng(0)
    X_train, y_train = _part(rng, 200, spread)
    X_val, y_val = _part(rng, 100, spread)
    X_test, y_test = _part(rng, 100, spread) if test else (None, None)
    return Splits(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train,
                  y_val=y_val, y_test=y_test, transformer=None,
                  feature_names=["a", "b"], class_names=CLASS_NAMES)


def _config(max_epochs, patience):
    return {
        "seed": 0,
        "network": {"hidden_units": [8], "dropout": 0.0, "activation": "relu"},
        "training": {"optimizer": "adam", "learning_rate": 0.05, "batch_size": 32,
                     "max_epochs": max_epochs, "early_stopping_patience": patience},
    }


def _val_macro_f1(model, splits):
    proba = model.predict_proba(splits, "val")
    return compute_metrics(splits.y_val, proba.argmax(axis=1), CLASS_NAMES)["macro_f1"]


def test_fit_reports_what_the_result_file_needs():
    splits = _splits(spread=6.0)
    model = get_model("mlp_ce")()
    info = model.fit(splits, _config(max_epochs=30, patience=3))
    n = info["epochs_run"]
    assert info["n_parameters"] == (2 * 8 + 8) + (8 * 3 + 3)
    assert 1 <= info["best_epoch"] <= n
    assert len(info["val_macro_f1_history"]) == n
    assert len(info["epoch_seconds"]) == n
    assert len(info["val_score_seconds"]) == n
    assert info["best_val_macro_f1"] == max(info["val_macro_f1_history"])


def test_it_learns_and_stops_early_on_easy_data():
    splits = _splits(spread=6.0)
    model = get_model("mlp_ce")()
    info = model.fit(splits, _config(max_epochs=40, patience=2))
    assert info["epochs_run"] < 40
    proba = model.predict_proba(splits, "test")
    assert proba.shape == (len(splits.y_test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba.argmax(axis=1) == splits.y_test).mean() > 0.9


def test_weights_of_the_best_epoch_are_restored_after_early_stop():
    # overlapping classes: the score moves around, so the last epoch is not the best
    splits = _splits(spread=1.0)
    model = get_model("mlp_ce")()
    info = model.fit(splits, _config(max_epochs=40, patience=1))
    assert _val_macro_f1(model, splits) == pytest.approx(info["best_val_macro_f1"])


def test_weights_of_the_best_epoch_are_restored_when_max_epochs_is_reached():
    splits = _splits(spread=1.0)
    model = get_model("mlp_ce")()
    info = model.fit(splits, _config(max_epochs=6, patience=100))
    assert info["epochs_run"] == 6
    assert _val_macro_f1(model, splits) == pytest.approx(info["best_val_macro_f1"])


def test_fit_never_reads_the_test_part():
    splits = _splits(spread=6.0, test=False)  # X_test and y_test are None
    get_model("mlp_ce")().fit(splits, _config(max_epochs=3, patience=3))


def test_unknown_part_is_an_error():
    splits = _splits(spread=6.0)
    model = get_model("mlp_ce")()
    model.fit(splits, _config(max_epochs=2, patience=2))
    with pytest.raises(ValueError, match="dev"):
        model.predict_proba(splits, "dev")


def test_fit_reports_selection_with_a_window():
    splits = _splits(spread=1.0)
    config = _config(max_epochs=12, patience=100)
    config["training"]["selection_window"] = 3
    info = get_model("mlp_ce")().fit(splits, config)
    history = info["val_macro_f1_history"]
    smoothed = [np.mean(history[i - 2:i + 1]) for i in range(2, len(history))]
    best = 2 + int(np.argmax(smoothed))  # the first two epochs are not candidates
    assert info["selection_window"] == 3
    assert info["best_epoch"] == best + 1
    assert info["selection_score"] == pytest.approx(max(smoothed))
    assert info["best_val_macro_f1"] == pytest.approx(history[best])


def test_training_shorter_than_the_window_keeps_the_last_epoch():
    splits = _splits(spread=1.0)
    config = _config(max_epochs=2, patience=100)
    config["training"]["selection_window"] = 5
    model = get_model("mlp_ce")()
    info = model.fit(splits, config)
    assert info["best_epoch"] == 2
    assert _val_macro_f1(model, splits) == pytest.approx(info["val_macro_f1_history"][-1])


def test_class_weight_power_is_used_and_reported():
    splits = _splits(spread=6.0)
    config = _config(max_epochs=2, patience=2)
    config["training"].update({"imbalance": "class_weight", "class_weight_power": 0.5})
    info = get_model("mlp_ce")().fit(splits, config)
    assert info["class_weight_power"] == 0.5
    assert len(info["class_weights"]) == 3


def test_no_imbalance_reports_no_weights():
    info = get_model("mlp_ce")().fit(_splits(spread=6.0), _config(max_epochs=2, patience=2))
    assert info["class_weights"] is None
    assert info["class_weight_power"] is None


def test_class_weight_power_defaults_to_balanced():
    splits = _splits(spread=6.0)
    config = _config(max_epochs=2, patience=2)
    config["training"]["imbalance"] = "class_weight"
    info = get_model("mlp_ce")().fit(splits, config)
    assert info["class_weight_power"] == 1.0
    assert list(info["class_weights"].values()) == pytest.approx([1.0, 1.0, 1.0])


def test_predict_batches_matches_predict():
    from nesybench.models.selection import predict_batches

    splits = _splits(spread=2.0)
    model = get_model("mlp_ce")()
    model.fit(splits, {"seed": 0, "network": {"hidden_units": [8], "dropout": 0.0, "activation": "relu"},
                       "training": {"optimizer": "adam", "learning_rate": 1e-2, "batch_size": 32,
                                    "max_epochs": 2, "early_stopping_patience": 5, "selection_window": 1}})
    X = splits.X_val
    expected = model.model.predict(X, batch_size=64, verbose=0)
    assert np.allclose(predict_batches(model.model, X, batch_size=64), expected, atol=1e-6)
    assert np.allclose(predict_batches(model.model, X, batch_size=7), expected, atol=1e-6)
