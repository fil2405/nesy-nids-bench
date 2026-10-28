import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "analysis"))
import stress_curves as sc  # noqa: E402
import summarize_runs  # noqa: E402

CLASSES = ["Normal", "DoS", "R2L"]
E7_5 = {"train_fraction": 0.05, "min_keep_per_class": 5}
E7_50 = {"train_fraction": 0.5, "min_keep_per_class": 5}


def rec(run_id, seed, macro, train=None, sp=None):
    r = {"seed": seed, "config": {"run_id": run_id, "split_params": sp or {}},
         "metrics": {"test": {"class_names": CLASSES, "macro_f1": macro, "f1": [macro] * 3}}}
    if train:
        r["split_sizes"] = {"train": train, "val": [1, 1, 1], "test": [1, 1, 1]}
    return r


def runs_for(spec):
    """spec: {run_id: ([macro per seed], train sizes or None, split_params or None)}"""
    return {run_id: {s: rec(run_id, s, m, train, sp) for s, m in enumerate(scores)}
            for run_id, (scores, train, sp) in spec.items()}


@pytest.fixture
def runs():
    return runs_for({
        "nslkdd_ltn_full_tuned_e7_f5": ([0.60, 0.62, 0.61], [500, 300, 50], E7_5),
        "nslkdd_ltn_sup_tuned_e7_f5": ([0.55, 0.57, 0.56], [500, 300, 50], E7_5),
        "nslkdd_ltn_full_tuned_e7_f50": ([0.70, 0.71, 0.72], [5000, 3000, 500], E7_50),
        "nslkdd_ltn_sup_tuned_e7_f50": ([0.69, 0.70, 0.70], [5000, 3000, 500], E7_50),
        "nslkdd_ltn_full_tuned": ([0.80, 0.81, 0.79], None, None),
        "nslkdd_ltn_sup_tuned": ([0.79, 0.80, 0.79], None, None),
        "nslkdd_xgb_tuned_e8_f10": ([0.5, 0.5, 0.5], [10000, 6000, 100],
                                    {"train_class_fractions": {"R2L": 0.1}, "min_keep_per_class": 5}),
        "cicids2017_xgb_tuned_e7_f1": ([0.5, 0.5, 0.5], [1, 1, 1], None),
    })


def test_collect_picks_dataset_exp_and_finals(runs):
    found = sc.collect(runs, "nslkdd", "e7")
    assert found["ltn_full"] == {0.05: "nslkdd_ltn_full_tuned_e7_f5",
                                 0.5: "nslkdd_ltn_full_tuned_e7_f50",
                                 1.0: "nslkdd_ltn_full_tuned"}
    assert found["xgb"] == {}


def test_level_value_decimal_tag():
    assert sc.level_value("5") == pytest.approx(0.05)
    assert sc.level_value("0p1") == pytest.approx(0.001)


def test_full_sizes_e7_scale_every_class(runs):
    found = sc.collect(runs, "nslkdd", "e7")
    assert sc.full_sizes(runs, found, "test") == pytest.approx([10000, 6000, 1000])


def test_full_sizes_e8_scale_only_the_thinned_class(runs):
    found = sc.collect(runs, "nslkdd", "e8")
    assert sc.full_sizes(runs, found, "test") == pytest.approx([10000, 6000, 1000])


def test_full_sizes_skip_classes_at_the_floor():
    runs = runs_for({"nslkdd_xgb_tuned_e8_f10": ([0.5], [100, 60, 5],
                     {"train_class_fractions": {"R2L": 0.1}, "min_keep_per_class": 5})})
    est = sc.full_sizes(runs, sc.collect(runs, "nslkdd", "e8"), "test")
    assert est[:2] == pytest.approx([100, 60]) and np.isnan(est[2])


def test_table_rows_and_class_rows(runs):
    found = sc.collect(runs, "nslkdd", "e7")
    rows = sc.table(runs, found, "test", "macro_f1", "R2L")
    by = {(r[0], r[1]): r for r in rows}
    arm, level, n, total, crows, mean, std = by[("ltn_full", 0.05)]
    assert (n, total, crows) == (3, 850, 50)
    assert mean == pytest.approx(0.61) and std == pytest.approx(0.01)
    assert by[("ltn_full", 1.0)][3] == pytest.approx(17000)
    assert by[("ltn_full", 1.0)][4] == pytest.approx(1000)


def test_knowledge_effect_is_full_minus_sup(runs):
    found = sc.collect(runs, "nslkdd", "e7")
    effect = dict(sc.knowledge_effect(runs, found, "test", "macro_f1"))
    assert set(effect) == {0.05, 0.5, 1.0}
    assert effect[0.05].median == pytest.approx(0.05)
    assert effect[0.05].wins == 3


def test_tex_tables_have_one_row_per_arm_and_level(runs):
    found = sc.collect(runs, "nslkdd", "e7")
    rows = sc.table(runs, found, "test", "macro_f1", None)
    text = sc.tex_scores(rows, "macro_f1")
    assert "ltn-full" in text and "ltn-sup" in text and "xgb" not in text
    effect = sc.knowledge_effect(runs, found, "test", "macro_f1")
    assert sc.tex_effect(effect, "macro_f1").count("\\\\") == 1 + len(effect)


def test_load_runs_skips_files_that_are_not_runs(tmp_path):
    (tmp_path / "e12.json").write_text(json.dumps({"batches": {}}))
    (tmp_path / "r_seed0.json").write_text(json.dumps(rec("r", 0, 0.5)))
    assert list(summarize_runs.load_runs(tmp_path)) == ["r"]


def test_collect_ltn_full_edit_tag(runs):
    extra = runs_for({
        "nslkdd_ltn_full_tuned_e7_no_A3_f5": ([0.58, 0.6, 0.59], [500, 300, 50], E7_5),
        "nslkdd_ltn_full_tuned_e3_no_A3": ([0.78, 0.8, 0.79], None, None),
    })
    both = {**runs, **extra}
    assert sc.collect(both, "nslkdd", "e7")["ltn_full"][0.05] == "nslkdd_ltn_full_tuned_e7_f5"
    found = sc.collect(both, "nslkdd", "e7", "no_A3")
    assert found["ltn_full"] == {0.05: "nslkdd_ltn_full_tuned_e7_no_A3_f5", 1.0: "nslkdd_ltn_full_tuned_e3_no_A3"}
    assert found["ltn_sup"][0.05] == "nslkdd_ltn_sup_tuned_e7_f5"
