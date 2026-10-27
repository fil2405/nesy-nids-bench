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


def test_noise_is_seeded_relative_and_inside_the_train_range():
    from nesybench import perturb
    train = pd.DataFrame({"bytes": [0, 10, 1000], "rate": [0.0, 0.5, 1.0], "const": [7, 7, 7]})
    raw = pd.DataFrame({"bytes": [0, 100, 900] * 50, "rate": [0.0, 0.5, 0.99] * 50,
                        "const": [7] * 150})
    bounds = perturb.column_bounds(train, ["bytes", "rate", "const"])
    assert bounds["bytes"] == (0.0, 1000.0, True) and bounds["rate"][2] is False
    one = perturb.noise(raw, bounds, 0.5, seed=1)
    two = perturb.noise(raw, bounds, 0.5, seed=1)
    assert one.equals(two) and not one["bytes"].equals(raw["bytes"])
    assert (one["const"] == 7).all()                   # constant column untouched
    assert (one["bytes"][raw["bytes"] == 0] == 0).all() and (one["rate"][raw["rate"] == 0] == 0).all()
    assert one["bytes"].between(0, 1000).all() and one["rate"].between(0, 1).all()
    assert (one["bytes"] == one["bytes"].round()).all()
    assert raw["bytes"].tolist()[:3] == [0, 100, 900]  # input not modified


def test_evasion_moves_only_firing_rows_of_the_target_class(kb):
    from nesybench import perturb
    raw = pd.DataFrame({"rate": [0.9, 0.9, 0.0]})
    y = np.array([1, 0, 1])                            # rows 0 and 2 are DoS; only row 0 fires A1
    out, moved = perturb.evade(raw, y, kb, NAMES, 0.2)
    assert moved == 1
    assert out["rate"].tolist() == pytest.approx([0.5 - 0.1, 0.9, 0.0])
    assert raw["rate"].tolist() == [0.9, 0.9, 0.0]


@pytest.mark.parametrize("spec", ["noise", "noise:", "shake:0.1", "noise:0", "evade:-1"])
def test_bad_perturbation_specs_are_refused(spec):
    from nesybench import perturb
    with pytest.raises(ValueError):
        perturb.parse(spec)


def test_summary_has_a_block_per_perturbation(kb):
    base = saved_eval.score(np.array([[.8, .1, .1], [.1, .8, .1]]), np.array([0, 1]),
                            pd.DataFrame({"rate": [0.0, 0.9]}), kb, NAMES, ("enforce",))
    base["perturbed"] = {"noise:0.1": base}
    summary = saved_eval.summarize({0: base}, ("enforce",))
    assert set(summary["perturbed"]) == {"noise:0.1"}
    assert summary["perturbed"]["noise:0.1"]["macro_f1"]["mean"] == pytest.approx(base["macro_f1"])
