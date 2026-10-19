import numpy as np
import pandas as pd
import pytest

from nesybench.data.preprocess import Splits
from nesybench.models import get_model

CLASS_NAMES = ["A", "B", "C"]
CENTERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

KB_YAML = """
version: toy_v1
axioms:
  - id: T1
    group: dataset
    origin: mechanism
    target: B
    all:
      - {col: f1, op: gt, tau: 3, s: 1}
"""


def _part(rng, counts, spread):
    y = np.repeat(np.arange(3), counts)
    X = CENTERS[y] * spread + rng.normal(size=(len(y), 2))
    return X.astype("float32"), y


def _splits(spread=6.0, train_counts=(200, 200, 200), test=True, raw=True):
    rng = np.random.default_rng(0)
    X_train, y_train = _part(rng, train_counts, spread)
    X_val, y_val = _part(rng, (100, 100, 100), spread)
    X_test, y_test = _part(rng, (100, 100, 100), spread) if test else (None, None)
    frame = (lambda X: pd.DataFrame({"f1": X[:, 0]})) if raw else (lambda X: None)
    return Splits(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train,
                  y_val=y_val, y_test=y_test, transformer=None,
                  feature_names=["a", "b"], class_names=CLASS_NAMES,
                  raw_train=frame(X_train), raw_val=frame(X_val),
                  raw_test=frame(X_test) if test else None)


@pytest.fixture
def kb_path(tmp_path):
    path = tmp_path / "kb.yaml"
    path.write_text(KB_YAML)
    return str(path)


def _config(kb_path, **training):
    return {
        "seed": 0,
        "kb": kb_path,
        "network": {"hidden_units": [16], "dropout": 0.0, "activation": "relu"},
        "training": {"optimizer": "adam", "learning_rate": 1e-2, "batch_size": 32,
                     "max_epochs": 30, "early_stopping_patience": 5,
                     "selection_window": 1, "sampling_power": 0.0,
                     "min_per_class": 2, **training},
    }


def test_registered():
    assert get_model("ltn_full").__name__ == "LTNFull"


def test_learns_and_reports_the_kb(kb_path):
    splits = _splits()
    model = get_model("ltn_full")()
    info = model.fit(splits, _config(kb_path))
    assert info["kb_version"] == "toy_v1"
    assert info["kb_ids"] == ["T1"]
    assert info["domain_weight"] == 0.25 and info["crisp"] is False
    assert 0.0 < info["antecedent_mean_train"][0] < 1.0
    assert len(info["train_loss_history"]) == info["epochs_run"]
    proba = model.predict_proba(splits, "test")
    assert proba.shape == (len(splits.y_test), 3)
    assert (proba.argmax(axis=1) == splits.y_test).mean() > 0.9


def test_zero_weight_is_ltn_sup(kb_path):
    splits = _splits()
    config = _config(kb_path, max_epochs=6, domain_weight=0.0)
    full, sup = get_model("ltn_full")(), get_model("ltn_sup")()
    a, b = full.fit(splits, config), sup.fit(splits, config)
    assert a["train_loss_history"] == pytest.approx(b["train_loss_history"], rel=1e-4)
    assert np.allclose(full.predict_proba(splits, "val"),
                       sup.predict_proba(splits, "val"), atol=1e-4)


def test_domain_weight_changes_the_loss(kb_path):
    splits = _splits()
    low = get_model("ltn_full")().fit(splits, _config(kb_path, max_epochs=3,
                                                       domain_weight=0.0))
    high = get_model("ltn_full")().fit(splits, _config(kb_path, max_epochs=3,
                                                        domain_weight=0.5))
    assert low["train_loss_history"] != pytest.approx(high["train_loss_history"],
                                                      rel=1e-3)


def test_crisp_runs_without_s(tmp_path):
    path = tmp_path / "kb.yaml"
    path.write_text(KB_YAML.replace(", s: 1", ""))
    info = get_model("ltn_full")().fit(_splits(), _config(str(path), max_epochs=3,
                                                          crisp=True))
    assert info["crisp"] is True
    assert 0.0 < info["antecedent_mean_train"][0] < 1.0


def test_graded_without_s_is_refused(tmp_path):
    path = tmp_path / "kb.yaml"
    path.write_text(KB_YAML.replace(", s: 1", ""))
    with pytest.raises(ValueError, match="s not set"):
        get_model("ltn_full")().fit(_splits(), _config(str(path), max_epochs=2))


def test_needs_a_kb(kb_path):
    config = _config(kb_path)
    config["kb"] = None
    with pytest.raises(ValueError, match="kb"):
        get_model("ltn_full")().fit(_splits(), config)


def test_needs_raw_rows(kb_path):
    with pytest.raises(ValueError, match="raw rows"):
        get_model("ltn_full")().fit(_splits(raw=False), _config(kb_path, max_epochs=2))


def test_weight_out_of_range_is_refused(kb_path):
    with pytest.raises(ValueError, match="domain_weight"):
        get_model("ltn_full")().fit(_splits(), _config(kb_path, domain_weight=1.5))


def test_class_weights_are_refused(kb_path):
    with pytest.raises(ValueError, match="class weights"):
        get_model("ltn_full")().fit(_splits(), _config(kb_path, imbalance="class_weight"))


def test_a_rare_class_gives_finite_losses(kb_path):
    splits = _splits(train_counts=(300, 300, 3))
    info = get_model("ltn_full")().fit(splits, _config(kb_path, max_epochs=5))
    assert np.all(np.isfinite(info["train_loss_history"]))


def test_fit_never_reads_the_test_part(kb_path):
    get_model("ltn_full")().fit(_splits(test=False), _config(kb_path, max_epochs=3))


def test_p_and_s_scale_are_reported_and_change_the_loss(kb_path):
    splits = _splits()
    base = get_model("ltn_full")().fit(splits, _config(kb_path, max_epochs=3))
    other = get_model("ltn_full")().fit(
        splits, _config(kb_path, max_epochs=3, p=4, s_scale=3.0))
    assert (base["p"], base["s_scale"]) == (2, 1.0)
    assert (other["p"], other["s_scale"]) == (4, 3.0)
    assert base["train_loss_history"] != pytest.approx(other["train_loss_history"], rel=1e-3)


def test_s_scale_moves_the_antecedents(kb_path):
    splits = _splits()
    base = get_model("ltn_full")().fit(splits, _config(kb_path, max_epochs=1))
    wide = get_model("ltn_full")().fit(splits, _config(kb_path, max_epochs=1, s_scale=4.0))
    assert base["antecedent_mean_train"] != pytest.approx(wide["antecedent_mean_train"], rel=1e-3)


def test_s_scale_must_be_positive(kb_path):
    with pytest.raises(ValueError, match="s_scale"):
        get_model("ltn_full")().fit(_splits(), _config(kb_path, s_scale=0.0))
