"""Saved models give back the same probabilities after loading."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

import nesybench  # sets TF_USE_LEGACY_KERAS before TensorFlow is imported
from nesybench.data.base import DatasetSpec
from nesybench.data.preprocess import Splits, preprocess
from nesybench.export import artifact_dir, file_sizes, save_preprocessing
from nesybench.models import get_model

CLASS_NAMES = ["A", "B", "C"]
CENTERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _part(rng, n_per_class, spread=6.0):
    y = np.repeat(np.arange(3), n_per_class)
    X = CENTERS[y] * spread + rng.normal(size=(len(y), 2))
    return X.astype("float32"), y


def _splits():
    rng = np.random.default_rng(0)
    X_train, y_train = _part(rng, 100)
    X_val, y_val = _part(rng, 50)
    X_test, y_test = _part(rng, 50)
    return Splits(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train,
                  y_val=y_val, y_test=y_test, transformer=None,
                  feature_names=["a", "b"], class_names=CLASS_NAMES)


@pytest.fixture
def folder():
    return Path(tempfile.mkdtemp())


def test_artifact_dir_names_run_and_seed(folder):
    assert artifact_dir(folder, "r", 3) == folder / "r_seed3"


def test_file_sizes_lists_files_with_their_bytes(folder):
    (folder / "a.bin").write_bytes(b"123")
    (folder / "sub").mkdir()
    (folder / "sub" / "b.bin").write_bytes(b"12")
    assert file_sizes(folder) == {"a.bin": 3, str(Path("sub") / "b.bin"): 2}


def test_preprocessing_loads_and_transforms_like_the_original(folder):
    spec = DatasetSpec(name="toy", classes=("A", "B"), categorical=("proto",),
                       non_features=("label", "class"), fixed_test=True)
    rng = np.random.default_rng(0)

    def frame(n):
        labels = np.resize(["A", "B"], n)
        return pd.DataFrame({"x": rng.normal(size=n), "proto": np.resize(["tcp", "udp"], n),
                             "label": labels, "class": labels})

    train, val, test = frame(40), frame(20), frame(20)
    splits = preprocess(train, val, test, spec)
    save_preprocessing(splits, folder)

    loaded = joblib.load(folder / "preprocessing.joblib")
    assert np.allclose(loaded.transform(val.drop(columns=["label", "class"])), splits.X_val)
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["feature_names"] == splits.feature_names
    assert meta["class_names"] == ["A", "B"]


def test_rf_saved_and_loaded(folder):
    splits = _splits()
    model = get_model("rf")()
    model.fit(splits, {"seed": 0, "params": {"n_estimators": 10}, "training": {}})
    model.save(folder)
    loaded = joblib.load(folder / "model.joblib")
    assert np.array_equal(loaded.predict_proba(splits.X_val), model.model.predict_proba(splits.X_val))


def test_xgb_saved_and_loaded(folder):
    import xgboost as xgb

    splits = _splits()
    model = get_model("xgb")()
    model.fit(splits, {"seed": 0, "params": {"max_depth": 3},
                       "training": {"max_rounds": 20, "early_stopping_patience": 3}})
    model.save(folder)
    booster = xgb.Booster()
    booster.load_model(str(folder / "model.json"))
    best = int(booster.attr("best_round"))
    assert best == model.best_round
    proba = booster.predict(xgb.DMatrix(splits.X_val), iteration_range=(0, best))
    assert np.allclose(proba, model.predict_proba(splits, "val"))


def test_rule_only_saves_its_kb(folder):
    kb = folder / "kb_in.yaml"
    kb.write_text(
        "version: t_v1\n"
        "axioms:\n"
        "  - {id: A1, group: dos, origin: mechanism, target: B,\n"
        "     all: [{col: flag, op: eq, value: S0}]}\n")
    out = folder / "out"
    out.mkdir()
    df = pd.DataFrame({"flag": ["S0", "SF"]})
    splits = SimpleNamespace(class_names=CLASS_NAMES, raw=lambda part: df)
    model = get_model("rule_only")()
    model.fit(splits, {"kb": str(kb)})
    model.save(out)
    rules = json.loads((out / "rules.json").read_text())
    assert rules == {"kb_version": "t_v1", "default_class": 0, "targets": [1]}
    assert (out / "kb.yaml").read_text() == kb.read_text()


def _keras_config():
    return {
        "seed": 0,
        "network": {"hidden_units": [8], "dropout": 0.0, "activation": "relu"},
        "training": {"optimizer": "adam", "learning_rate": 0.05, "batch_size": 32,
                     "max_epochs": 3, "early_stopping_patience": 3},
    }


@pytest.mark.parametrize("name", ["mlp_ce", "ltn_sup"])
def test_keras_arms_saved_and_loaded(folder, name):
    pytest.importorskip("tensorflow")
    import tf_keras

    splits = _splits()
    model = get_model(name)()
    model.fit(splits, _keras_config())
    model.save(folder)
    loaded = tf_keras.models.load_model(str(folder / "model.keras"))
    assert np.allclose(np.asarray(loaded(splits.X_val, training=False)),
                       model.predict_proba(splits, "val"), atol=1e-6)
