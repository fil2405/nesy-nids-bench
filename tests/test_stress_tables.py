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


def test_saved_eval_reports_read_as_runs(tmp_path):
    import json
    rep = {"run_id": "cicddos2019_xgb_tuned", "part": "test",
           "per_seed": {"0": {"macro_f1": 0.7, "by_label": {"Portmap": {"n": 10, "recall": 0.5, "detected": 0.6}}}}}
    (tmp_path / "x.json").write_text(json.dumps(rep))
    runs = st.saved_eval_runs(tmp_path)
    rows = st.unseen_rows(runs, "cicddos2019", ["Portmap"])
    assert rows[0][0] == "xgb" and rows[0][2][0] == pytest.approx(0.6)


def test_e3_rows_against_the_base_and_the_weak_tag():
    runs = {"nslkdd_ltn_full_tuned": {s: e4_rec("f", s, 0.57) for s in range(3)},
            "nslkdd_ltn_full_tuned_e3_no_A1": {s: e4_rec("a", s, 0.56) for s in range(3)},
            "nslkdd_ltn_full_tuned_weak": {s: e4_rec("w", s, 0.58) for s in range(3)},
            "nslkdd_ltn_full_tuned_weak_e3_no_R1": {s: e4_rec("r", s, 0.55) for s in range(3)}}
    v = st.by_variant(runs, "nslkdd", "e3", ("ltn_full",))
    assert list(v) == ["no_A1"]
    (row,) = st.e4_rows(runs, "nslkdd", v, "test", "macro_f1", ("ltn_full",))
    assert row[5].median == pytest.approx(-0.01)
    w = st.by_variant(runs, "nslkdd", "e3", ("ltn_full",), "_weak")
    (row,) = st.e4_rows(runs, "nslkdd", w, "test", "macro_f1", ("ltn_full",), "_weak")
    assert row[1] == "no_R1" and row[5].median == pytest.approx(-0.03)


def xfer_rec(run_id, seed, macro, det, fa):
    return {"seed": seed, "config": {"run_id": run_id},
            "metrics": {"test": {"macro_f1": macro, "recall": [1 - fa, det]}}}


@pytest.fixture
def xfer_runs():
    spec = {"cic_transfer_ltn_sup_tuned_e9_xfer": (0.32, 0.08, 0.3),
            "cic_transfer_ltn_full_tuned_e9_xfer": (0.35, 0.02, 0.1),
            "cic_transfer_ltn_full_tuned_e9_xfer_mech": (0.46, 0.12, 0.07),
            "cic_transfer_rules_e9_xfer_mech": (0.92, 0.83, 0.01),
            "cicids2017_ltn_full_tuned": (0.9, 0.9, 0.0)}
    return {rid: {s: xfer_rec(rid, s, m + 0.01 * s, d, f) for s in range(3)} for rid, (m, d, f) in spec.items()}


def test_transfer_rows_and_effects(xfer_runs):
    assert st.transfer_runs(xfer_runs)[("rules", "xfer_mech")] == "cic_transfer_rules_e9_xfer_mech"
    rows = st.transfer_rows(xfer_runs)
    assert [(r[0], r[1]) for r in rows][:2] == [("rules", "xfer_mech"), ("ltn_full", "xfer_mech")]
    arm, base, n, mf1, det, fa = rows[0]
    assert n == 3 and det[0] == pytest.approx(0.83) and fa[0] == pytest.approx(0.01)
    eff = {(a, b): r for a, b, r in st.transfer_effects(xfer_runs)}
    assert eff[(("ltn_full", "xfer"), ("ltn_full", "xfer_mech"))].median == pytest.approx(0.11)
    assert (("mlp_weights", "xfer"), ("ltn_full", "xfer_mech")) not in eff
    assert "\\texttt{xfer\\_mech}" in st.tex_transfer(rows)
