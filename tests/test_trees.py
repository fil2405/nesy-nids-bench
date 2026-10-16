"""Random Forest and XGBoost on synthetic Gaussian blobs."""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from nesybench.data.preprocess import Splits
from nesybench.metrics import compute_metrics
from nesybench.models import get_model
from nesybench.models.weights import class_weights

CLASS_NAMES = ["A", "B", "C"]
CENTERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _part(rng, counts, spread):
    y = np.repeat(np.arange(3), counts)
    X = CENTERS[y] * spread + rng.normal(size=(len(y), 2))
    return X.astype("float32"), y


def _splits(spread, train_counts=(200, 200, 200), test=True):
    rng = np.random.default_rng(0)
    X_train, y_train = _part(rng, train_counts, spread)
    X_val, y_val = _part(rng, (100, 100, 100), spread)
    X_test, y_test = _part(rng, (100, 100, 100), spread) if test else (None, None)
    return Splits(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train,
                  y_val=y_val, y_test=y_test, transformer=None,
                  feature_names=["a", "b"], class_names=CLASS_NAMES)


def _rf_config(**training):
    return {"seed": 0, "params": {"n_estimators": 20}, "training": training}


def _xgb_config(max_rounds, patience, **training):
    return {
        "seed": 0,
        "params": {"max_depth": 3},
        "training": {"max_rounds": max_rounds, "early_stopping_patience": patience,
                     **training},
    }


def _val_macro_f1(model, splits):
    proba = model.predict_proba(splits, "val")
    return compute_metrics(splits.y_val, proba.argmax(axis=1), CLASS_NAMES)["macro_f1"]


def _spy_rf_fit(monkeypatch):
    seen = {}
    original = RandomForestClassifier.fit

    def fit(self, X, y, sample_weight=None):
        seen["sample_weight"] = sample_weight
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(RandomForestClassifier, "fit", fit)
    return seen


def test_both_models_are_registered():
    assert get_model("rf").__name__ == "RandomForest"
    assert get_model("xgb").__name__ == "XGBoost"


# Random Forest

def test_rf_learns_and_reports():
    splits = _splits(spread=6.0)
    model = get_model("rf")()
    info = model.fit(splits, _rf_config())
    assert info["n_trees"] == 20
    assert info["n_nodes"] >= 20
    assert info["fit_seconds"] > 0
    proba = model.predict_proba(splits, "test")
    assert proba.shape == (len(splits.y_test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba.argmax(axis=1) == splits.y_test).mean() > 0.9


def test_rf_same_seed_same_probabilities():
    splits = _splits(spread=1.0)
    first, second = get_model("rf")(), get_model("rf")()
    first.fit(splits, _rf_config())
    second.fit(splits, _rf_config())
    assert np.array_equal(first.predict_proba(splits, "val"),
                          second.predict_proba(splits, "val"))


def test_rf_class_weights_go_in_as_sample_weight(monkeypatch):
    counts = (300, 100, 20)
    splits = _splits(spread=1.0, train_counts=counts)
    seen = _spy_rf_fit(monkeypatch)
    info = get_model("rf")().fit(
        splits, _rf_config(imbalance="class_weight", class_weight_power=0.5))
    per_class = class_weights(counts, 0.5)
    assert seen["sample_weight"] == pytest.approx(per_class[splits.y_train])
    assert info["class_weight_power"] == 0.5
    assert list(info["class_weights"].values()) == pytest.approx(list(per_class))


def test_rf_class_weight_power_defaults_to_balanced():
    splits = _splits(spread=1.0, train_counts=(300, 100, 20))
    info = get_model("rf")().fit(splits, _rf_config(imbalance="class_weight"))
    assert info["class_weight_power"] == 1.0


def test_rf_without_imbalance_has_no_weights(monkeypatch):
    seen = _spy_rf_fit(monkeypatch)
    info = get_model("rf")().fit(_splits(spread=6.0), _rf_config())
    assert seen["sample_weight"] is None
    assert info["class_weights"] is None
    assert info["class_weight_power"] is None


def test_rf_unsupported_imbalance_is_an_error():
    with pytest.raises(ValueError, match="imbalance"):
        get_model("rf")().fit(_splits(spread=6.0), _rf_config(imbalance="smote"))


def test_rf_class_without_training_examples_keeps_its_column():
    splits = _splits(spread=6.0, train_counts=(200, 200, 0))
    model = get_model("rf")()
    model.fit(splits, _rf_config())
    proba = model.predict_proba(splits, "val")
    assert proba.shape == (len(splits.y_val), 3)
    assert np.all(proba[:, 2] == 0)


def test_rf_class_weights_reject_a_class_without_examples():
    splits = _splits(spread=6.0, train_counts=(200, 200, 0))
    with pytest.raises(ValueError, match="no training examples"):
        get_model("rf")().fit(splits, _rf_config(imbalance="class_weight"))


def test_rf_fit_never_reads_the_test_part():
    splits = _splits(spread=6.0, test=False)
    get_model("rf")().fit(splits, _rf_config())


def test_rf_unknown_part_is_an_error():
    splits = _splits(spread=6.0)
    model = get_model("rf")()
    model.fit(splits, _rf_config())
    with pytest.raises(ValueError, match="dev"):
        model.predict_proba(splits, "dev")


# XGBoost

def test_xgb_learns_and_reports():
    pytest.importorskip("xgboost")
    splits = _splits(spread=6.0)
    model = get_model("xgb")()
    info = model.fit(splits, _xgb_config(max_rounds=30, patience=3))
    n = info["rounds_run"]
    assert 1 <= info["best_round"] <= n <= 30
    assert info["n_trees"] == info["best_round"] * 3
    assert len(info["val_macro_f1_history"]) == n
    assert len(info["round_seconds"]) == n
    assert len(info["val_score_seconds"]) == n
    assert info["fit_seconds"] > 0
    assert info["best_val_macro_f1"] == max(info["val_macro_f1_history"])
    proba = model.predict_proba(splits, "test")
    assert proba.shape == (len(splits.y_test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba.argmax(axis=1) == splits.y_test).mean() > 0.9


def test_xgb_stops_early_on_easy_data():
    pytest.importorskip("xgboost")
    splits = _splits(spread=6.0)
    info = get_model("xgb")().fit(splits, _xgb_config(max_rounds=200, patience=2))
    assert info["rounds_run"] < 200


def test_xgb_predicts_with_the_chosen_round_only():
    pytest.importorskip("xgboost")
    splits = _splits(spread=1.0)  # overlapping classes: the last round is not the best
    model = get_model("xgb")()
    info = model.fit(splits, _xgb_config(max_rounds=40, patience=1))
    assert _val_macro_f1(model, splits) == pytest.approx(info["best_val_macro_f1"])


def test_xgb_chosen_round_is_kept_when_max_rounds_is_reached():
    pytest.importorskip("xgboost")
    splits = _splits(spread=1.0)
    model = get_model("xgb")()
    info = model.fit(splits, _xgb_config(max_rounds=6, patience=100))
    assert info["rounds_run"] == 6
    assert _val_macro_f1(model, splits) == pytest.approx(info["best_val_macro_f1"])


def test_xgb_selection_uses_the_window():
    pytest.importorskip("xgboost")
    splits = _splits(spread=1.0)
    config = _xgb_config(max_rounds=12, patience=100, selection_window=3)
    info = get_model("xgb")().fit(splits, config)
    history = info["val_macro_f1_history"]
    smoothed = [np.mean(history[i - 2:i + 1]) for i in range(2, len(history))]
    best = 2 + int(np.argmax(smoothed))  # the first two rounds are not candidates
    assert info["selection_window"] == 3
    assert info["best_round"] == best + 1
    assert info["selection_score"] == pytest.approx(max(smoothed))
    assert info["best_val_macro_f1"] == pytest.approx(history[best])


def test_xgb_training_shorter_than_the_window_keeps_the_last_round():
    pytest.importorskip("xgboost")
    splits = _splits(spread=1.0)
    config = _xgb_config(max_rounds=2, patience=100, selection_window=5)
    model = get_model("xgb")()
    info = model.fit(splits, config)
    assert info["best_round"] == 2
    assert _val_macro_f1(model, splits) == pytest.approx(info["val_macro_f1_history"][-1])


def test_xgb_class_weights_go_in_as_weights(monkeypatch):
    xgb = pytest.importorskip("xgboost")
    counts = (300, 100, 20)
    splits = _splits(spread=1.0, train_counts=counts)
    weights = []
    original = xgb.DMatrix

    def spy(*args, **kwargs):
        weights.append(kwargs.get("weight"))
        return original(*args, **kwargs)

    monkeypatch.setattr(xgb, "DMatrix", spy)
    config = _xgb_config(max_rounds=2, patience=2,
                         imbalance="class_weight", class_weight_power=0.5)
    info = get_model("xgb")().fit(splits, config)
    per_class = class_weights(counts, 0.5)
    assert weights[0] == pytest.approx(per_class[splits.y_train])  # train
    assert weights[1] is None                                      # validation
    assert info["class_weight_power"] == 0.5


def test_xgb_without_imbalance_has_no_weights():
    pytest.importorskip("xgboost")
    info = get_model("xgb")().fit(_splits(spread=6.0), _xgb_config(max_rounds=2, patience=2))
    assert info["class_weights"] is None
    assert info["class_weight_power"] is None


def test_xgb_fit_never_reads_the_test_part():
    pytest.importorskip("xgboost")
    splits = _splits(spread=6.0, test=False)
    get_model("xgb")().fit(splits, _xgb_config(max_rounds=3, patience=3))
