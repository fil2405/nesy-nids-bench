import numpy as np
import pandas as pd
import pytest

from nesybench.logic.edit import dump_kb, edit_kb, load_run_kb
from nesybench.logic.kb import load_kb

KB_YAML = """
version: test_v1
axioms:
  - {id: A1, group: ddos, origin: mechanism, target: DoS,
     all: [{col: flag, op: eq, value: S0}, {col: count, op: gt, tau: 250, s: 5}]}
  - {id: A2, group: dataset, origin: data, target: Probe,
     all: [{col: rate, op: le, tau: 0.5, s: 0.1}]}
  - {id: A3, group: dataset, origin: data, target: Probe,
     all: [{col: rate, op: gt, tau: 0.9, s: 0.1}]}
"""


@pytest.fixture
def kb(tmp_path):
    path = tmp_path / "kb.yaml"
    path.write_text(KB_YAML)
    return load_kb(path)


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"flag": rng.choice(["S0", "SF"], 2000),
                         "count": rng.uniform(0, 500, 2000),
                         "rate": rng.uniform(0, 1, 2000)})


def tau(axiom, col):
    return next(c.tau for c in axiom.conditions if c.col == col)


def test_exclude_by_id_and_group(kb):
    assert edit_kb(kb, {"exclude": ["A2"]}).ids == ["A1", "A3"]
    assert edit_kb(kb, {"exclude": ["group:dataset"]}).ids == ["A1"]


def test_exclude_unknown_or_everything(kb):
    with pytest.raises(ValueError, match="matches no axiom"):
        edit_kb(kb, {"exclude": ["A9"]})
    with pytest.raises(ValueError, match="every axiom"):
        edit_kb(kb, {"exclude": ["group:ddos", "group:dataset"]})


def test_unknown_key(kb):
    with pytest.raises(ValueError, match="unknown keys"):
        edit_kb(kb, {"drop": ["A1"]})


def test_swap_target(kb):
    out = edit_kb(kb, {"swap_target": {"A2": "DoS"}})
    assert [a.target for a in out.axioms] == ["DoS", "DoS", "Probe"]


def test_tau_shift_is_in_percentile_points(kb, raw):
    out = edit_kb(kb, {"tau_shift": {"axioms": ["A2"], "by": 0.1}}, raw)
    new = tau(out.axioms[1], "rate")
    assert np.mean(raw["rate"] <= new) == pytest.approx(0.6, abs=0.01)
    assert tau(out.axioms[0], "count") == 250        # other axioms untouched
    assert out.axioms[1].conditions[0].s == 0.1      # width kept


def test_tau_shift_leaves_eq_and_clips(kb, raw):
    out = edit_kb(kb, {"tau_shift": {"axioms": ["A1"], "by": 0.9}}, raw)
    flag, count = out.axioms[0].conditions
    assert flag.value == "S0" and flag.tau is None
    assert count.tau == pytest.approx(raw["count"].max())   # rank clipped at 1


def test_tau_shift_all_and_needs_data(kb, raw):
    out = edit_kb(kb, {"tau_shift": {"by": -0.1}}, raw)
    assert all(tau(a, c) < o for a, c, o in
               [(out.axioms[1], "rate", 0.5), (out.axioms[2], "rate", 0.9)])
    with pytest.raises(ValueError, match="raw train"):
        edit_kb(kb, {"tau_shift": {"by": 0.1}})


def test_copies_have_distinct_thresholds(kb, raw):
    out = edit_kb(kb, {"copies": {"axiom": "A1", "n": 5, "spread": 0.2}}, raw)
    assert out.ids == ["A1", "A2", "A3", "A1#1", "A1#2", "A1#3", "A1#4", "A1#5"]
    taus = [tau(a, "count") for a in out.axioms if a.id.startswith("A1")]
    assert len(set(taus)) == 6
    assert {a.target for a in out.axioms if a.id.startswith("A1")} == {"DoS"}


def test_order_exclude_then_copies(kb, raw):
    out = edit_kb(kb, {"exclude": ["A3"], "copies": {"axiom": "A2", "n": 2}}, raw)
    assert out.ids == ["A1", "A2", "A2#1", "A2#2"]
    with pytest.raises(ValueError, match="not in the KB"):
        edit_kb(kb, {"exclude": ["A2"], "copies": {"axiom": "A2", "n": 2}}, raw)


def test_original_kb_is_not_changed(kb, raw):
    edit_kb(kb, {"tau_shift": {"by": 0.2}, "exclude": ["A3"]}, raw)
    assert kb.ids == ["A1", "A2", "A3"] and tau(kb.axioms[1], "rate") == 0.5


def test_load_run_kb(tmp_path, raw):
    path = tmp_path / "kb.yaml"
    path.write_text(KB_YAML)
    assert load_run_kb({"kb": str(path)}).ids == ["A1", "A2", "A3"]
    assert load_run_kb({"kb": str(path), "kb_edit": {"exclude": ["A1"]}}).ids == ["A2", "A3"]


def test_dump_round_trip(tmp_path, kb, raw):
    out = edit_kb(kb, {"tau_shift": {"by": 0.1}, "copies": {"axiom": "A2", "n": 2}}, raw)
    dump_kb(out, tmp_path / "out.yaml")
    back = load_kb(tmp_path / "out.yaml")
    assert back.ids == out.ids
    assert back.axioms == out.axioms
