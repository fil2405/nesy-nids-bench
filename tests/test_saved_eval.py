import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from nesybench import saved_eval
from nesybench.latency import load_predictor
from nesybench.logic.kb import load_kb

NAMES = ["Normal", "DoS", "Probe"]
KB_YAML = """
version: test_v1
axioms:
  - {id: A1, group: dos, origin: mechanism, target: DoS,
     all: [{col: rate, op: gt, tau: 0.5, s: 0.1}]}
"""


@pytest.fixture
def kb(tmp_path):
    path = tmp_path / "kb.yaml"
    path.write_text(KB_YAML)
    return load_kb(path)


def test_score_reports_before_and_after_each_mode(kb):
    raw = pd.DataFrame({"rate": [0.9, 0.0, 0.9, 0.0]})
    y = np.array([1, 0, 1, 0])
    proba = np.array([[.8, .1, .1], [.8, .1, .1], [.1, .8, .1], [.8, .1, .1]])
    out = saved_eval.score(proba, y, raw, kb, NAMES, ("annotate", "enforce"))
    assert out["postcheck"]["annotate"]["macro_f1"] == out["macro_f1"]
    assert out["postcheck"]["enforce"]["macro_f1"] > out["macro_f1"]
    assert out["postcheck"]["enforce"]["n_changed"] == 1


def test_forest_missing_a_class_keeps_the_output_width(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    forest = RandomForestClassifier(n_estimators=3, random_state=0).fit(x, rng.integers(0, 2, 40))  # classes 0, 1
    joblib.dump(forest, tmp_path / "model.joblib")
    predict, _ = load_predictor(tmp_path, n_classes=3)
    proba = predict(x.astype(np.float32))
    assert proba.shape == (40, 3) and (proba[:, 2] == 0).all()


def test_predict_all_batches_the_rows():
    calls = []
    x = np.zeros((10, 2))
    out = saved_eval.predict_all(lambda b: (calls.append(len(b)), np.ones((len(b), 3)))[1], x, size=4)
    assert calls == [4, 4, 2] and out.shape == (10, 3)


def test_summarize_means_over_seeds(kb):
    rows = {s: saved_eval.score(np.array([[.8, .1, .1], [.1, .8, .1]]), np.array([0, 1]),
                                pd.DataFrame({"rate": [0.0, 0.9]}), kb, NAMES, ("enforce",)) for s in (0, 1)}
    summary = saved_eval.summarize(rows, ("enforce",))
    assert summary["seeds"] == 2 and summary["enforce"]["macro_f1"]["mean"] == pytest.approx(rows[0]["postcheck"]["enforce"]["macro_f1"])
