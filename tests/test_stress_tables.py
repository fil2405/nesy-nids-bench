import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "analysis"))
import stress_tables as st  # noqa: E402

HELD = ["DoS Slowloris", "DoS Slowhttptest"]


def e9_rec(run_id, seed, det, macro=0.8):
    by = {"DoS Slowloris": {"n": 300, "recall": det / 2, "detected": det},
          "DoS Slowhttptest": {"n": 100, "recall": 0.0, "detected": 0.0},
          "DoS Slowloris - Attempted": {"n": 50, "recall": 0.9, "detected": 0.1},
          "BENIGN": {"n": 1000, "recall": 1.0, "detected": 0.0}}
    return {"seed": seed,
            "config": {"run_id": run_id, "split_params": {"holdout_labels": HELD}},
            "metrics": {"test": {"macro_f1": macro, "by_label": by}}}


def e4_rec(run_id, seed, macro, training=None):
    return {"seed": seed, "config": {"run_id": run_id, "training": training or {}},
            "metrics": {"test": {"macro_f1": macro}}}


@pytest.fixture
def e9_runs():
    full = "cicids2017_ltn_full_tuned_e9_slow"
    sup = "cicids2017_ltn_sup_tuned_e9_slow"
    return {full: {s: e9_rec(full, s, 0.8 + 0.01 * s) for s in range(3)},
            sup: {s: e9_rec(sup, s, 0.4 + 0.01 * s) for s in range(3)},
            "cicids2017_ltn_full_tuned": {0: e9_rec("x", 0, 0.9)}}


def test_by_variant_groups_arms(e9_runs):
    v = st.by_variant(e9_runs, "cicids2017", "e9")
    assert v == {"slow": {"ltn_full": "cicids2017_ltn_full_tuned_e9_slow",
                          "ltn_sup": "cicids2017_ltn_sup_tuned_e9_slow"}}


def test_pooled_is_row_weighted_over_the_held_labels(e9_runs):
    rec = e9_runs["cicids2017_ltn_full_tuned_e9_slow"][0]
    assert st.pooled(rec, HELD, "detected") == pytest.approx(300 * 0.8 / 400)
    assert np.isnan(st.pooled(rec, ["missing"], "detected"))


def test_e9_rows_report_alarms_on_attempted(e9_runs):
    rows = st.e9_rows(e9_runs, st.by_variant(e9_runs, "cicids2017", "e9"))
    var, arm, n, labels, mf1, det, rec, att = rows[0]
    assert (var, arm, n, labels) == ("slow", "ltn_sup", 3, HELD)
    assert att[0] == pytest.approx(0.1)


def test_e9_effect_is_full_minus_sup_on_detection(e9_runs):
    (var, r), = st.e9_effect(e9_runs, st.by_variant(e9_runs, "cicids2017", "e9"))
    assert var == "slow" and r.median == pytest.approx(0.4 * 300 / 400) and r.wins == 3


def test_e4_rows_diff_from_finals():
    runs = {"nslkdd_ltn_full_tuned": {s: e4_rec("f", s, 0.57, {"p": 6, "s_scale": 1.29}) for s in range(3)},
            "nslkdd_ltn_full_tuned_e4_p1": {s: e4_rec("p1", s, 0.55) for s in range(3)},
            "nslkdd_ltn_sup_tuned_e4_p1": {s: e4_rec("q", s, 0.56) for s in range(2)}}
    variants = st.by_variant(runs, "nslkdd", "e4", st.LTN_ARMS)
    rows = {(r[0], r[1]): r for r in st.e4_rows(runs, "nslkdd", variants, "test", "macro_f1")}
    assert rows[("ltn_full", "p1")][5].median == pytest.approx(-0.02)
    assert rows[("ltn_sup", "p1")][5] is None
    assert st.base_setting(runs, "nslkdd", "ltn_full") == {"p": 6, "s_scale": 1.29}
    assert "ltn-full & p1" in st.tex_e4(list(rows.values()), "macro_f1")


def test_unseen_rows_read_the_finals(e9_runs):
    runs = {"cicids2017_ltn_full_tuned": e9_runs["cicids2017_ltn_full_tuned_e9_slow"],
            "cicids2017_rules": {0: e9_rec("r", 0, 0.2)}}
    rows = {r[0]: r for r in st.unseen_rows(runs, "cicids2017", ["DoS Slowloris"])}
    assert set(rows) == {"ltn_full", "rules"}
    assert rows["rules"][2][0] == pytest.approx(0.2)
