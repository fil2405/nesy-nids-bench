import json
import sys
from pathlib import Path

import numpy as np
import pytest

from nesybench import operating

sys.path.insert(0, str(Path(__file__).parents[1] / "analysis"))
import saved_tables as st  # noqa: E402


def test_threshold_allows_floor_fpr_false_alarms():
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 0.85])
    attack = np.array([False] * 5 + [True] * 2)
    t = operating.threshold_at(score, attack, 0.2)  # one false alarm out of five
    assert t == 0.8
    tpr, fpr = operating.rates(score, attack, t)
    assert (tpr, fpr) == (1.0, 0.2)
    assert operating.threshold_at(score, attack, 0.0) == 0.9
    assert operating.threshold_at(score, attack, 1.0) == -np.inf


def test_ties_never_exceed_the_budget():
    score = np.array([0.0] * 10 + [0.0, 1.0])
    attack = np.array([False] * 10 + [True, True])
    t = operating.threshold_at(score, attack, 0.5)
    assert operating.rates(score, attack, t) == (0.5, 0.0)


def test_cost_envelope_perfect_and_random():
    attack = np.array([False, False, True, True])
    assert np.allclose(operating.cost_envelope(np.array([0., .1, .8, .9]), attack), 0)
    env = operating.cost_envelope(np.zeros(4), attack)  # trivial classifiers only
    assert env[50] == pytest.approx(0.5) and env[0] == 0 and env[-1] == 0
    assert np.allclose(env, np.minimum(operating.GRID, 1 - operating.GRID))


def test_seed_report_threshold_from_validation_and_unseen():
    rng = np.random.default_rng(0)
    val_a = np.r_[np.zeros(1000, bool), np.ones(200, bool)]
    val_s = np.r_[rng.normal(0, 1, 1000), rng.normal(3, 1, 200)]
    test_a = np.r_[np.zeros(500, bool), np.ones(100, bool)]
    test_s = np.r_[rng.normal(0.5, 1, 500), rng.normal(3, 1, 100)]  # benign shifted on test
    labels = np.array(["normal"] * 500 + ["neptune"] * 50 + ["new"] * 50)
    rep = operating.seed_report(val_s, val_a, test_s, test_a, labels, {"new"}, test_s > 2)
    row = rep["threshold"]["0.01"]
    assert row["val_fpr"] <= 0.01 and row["test_fpr"] > row["val_fpr"] and row["gap"] == row["test_fpr"] - 0.01
    assert row["unseen_detected"] == row["by_label"]["new"]
    assert rep["argmax"]["test_tpr"] > 0.5 and 0 < rep["cost"]["area"] < 0.25


@pytest.fixture
def folder(tmp_path):
    for run_id, shift in (("nslkdd_ltn_full_tuned", 0.0), ("nslkdd_mlp_weights_tuned", 0.1)):
        per_seed = {}
        for s in range(3):
            env = np.minimum(operating.GRID, 1 - operating.GRID) * (0.2 + shift)
            env[:50] *= (1 - 4 * shift)  # mlp better at low PC
            per_seed[str(s)] = {"threshold": {"0.001": {"test_tpr": 0.8 - shift, "test_fpr": 0.002, "val_fpr": 0.001,
                                                        "unseen_detected": 0.3}},
                                "cost": {"envelope": env.tolist(), "area": float(env.mean())}}
        (tmp_path / f"{run_id}.json").write_text(json.dumps(
            {"run_id": run_id, "budgets": [0.001], "grid": operating.GRID.tolist(), "per_seed": per_seed}))
    return tmp_path


def test_threshold_rows(folder):
    rows = st.threshold_rows(st.load_reports(folder, "nslkdd"))
    arm, b, tpr, fa, fa_val, unseen = rows[0]
    assert (arm, b) == ("mlp_weights", 0.001)
    assert tpr[0] == pytest.approx(0.7) and fa[0] == pytest.approx(200) and fa_val[0] == pytest.approx(100)


def test_cost_best_ranges_and_figure(folder, tmp_path):
    grid, curves, areas = st.cost_curves(st.load_reports(folder, "nslkdd"))
    ranges = st.best_ranges(grid, curves)
    assert [r[2] for r in ranges][:3] == ["mlp_weights", "ltn_full", "mlp_weights"]  # PC 1: all zero, tie
    st.plot_costs(tmp_path / "c.pdf", grid, curves, "nslkdd")
    assert (tmp_path / "c.pdf").stat().st_size > 0
