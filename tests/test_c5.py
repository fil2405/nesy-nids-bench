import numpy as np
import pandas as pd
import pytest

from nesybench.logic.c5 import axiom_report
from nesybench.logic.kb import load_kb

NAMES = ["Normal", "DoS", "Probe"]

KB_YAML = """
version: test_v1
axioms:
  - id: A1
    group: ddos
    origin: mechanism
    target: DoS
    all:
      - {col: flag, op: eq, value: S0}
  - id: A2
    group: dataset
    origin: data
    target: Probe
    all:
      - {col: rate, op: gt, tau: 0.5, s: 0.1}
"""


@pytest.fixture
def kb(tmp_path):
    path = tmp_path / "kb.yaml"
    path.write_text(KB_YAML)
    return load_kb(path)


def test_violation_rate_and_satisfaction_by_hand(kb):
    df = pd.DataFrame({"flag": ["S0", "S0", "SF", "SF"], "rate": [0.0] * 4})
    proba = np.array([[.1, .8, .1], [.6, .3, .1], [.2, .7, .1], [.8, .1, .1]])
    r = axiom_report(kb, df, proba, NAMES)["axioms"]["A1"]
    assert r["n_covered"] == 2
    assert r["coverage"] == 0.5
    assert r["n_violations"] == 1
    assert r["violation_rate"] == 0.5
    expected = 1 - np.sqrt(np.mean([(1 - .8) ** 2, (1 - .3) ** 2]))
    assert r["satisfaction"] == pytest.approx(expected)


def test_argmax_decides_not_the_target_probability(kb):
    df = pd.DataFrame({"flag": ["S0"], "rate": [0.0]})
    proba = np.array([[.3, .4, .3]])
    r = axiom_report(kb, df, proba, NAMES)["axioms"]["A1"]
    assert r["violation_rate"] == 0.0
    assert r["satisfaction"] == pytest.approx(0.4)


def test_axiom_without_covered_rows_is_nan_not_zero(kb):
    df = pd.DataFrame({"flag": ["SF", "SF"], "rate": [0.0, 0.0]})
    proba = np.full((2, 3), 1 / 3)
    rep = axiom_report(kb, df, proba, NAMES)
    assert np.isnan(rep["axioms"]["A1"]["violation_rate"])
    assert np.isnan(rep["axioms"]["A1"]["satisfaction"])
    assert rep["axioms"]["A1"]["n_covered"] == 0


def test_means_skip_undefined_axioms(kb):
    df = pd.DataFrame({"flag": ["S0", "S0"], "rate": [0.0, 0.0]})
    proba = np.array([[.1, .8, .1], [.8, .1, .1]])
    rep = axiom_report(kb, df, proba, NAMES)
    assert np.isnan(rep["axioms"]["A2"]["violation_rate"])
    assert rep["mean_violation_rate"] == 0.5


def test_graded_antecedent_below_threshold_is_not_covered(kb):
    df = pd.DataFrame({"flag": ["SF"] * 3, "rate": [0.4, 0.5, 0.9]})
    proba = np.array([[.9, .05, .05]] * 3)
    r = axiom_report(kb, df, proba, NAMES)["axioms"]["A2"]
    assert r["n_covered"] == 2
    assert r["violation_rate"] == 1.0


def test_rule_precision_is_the_share_of_covered_rows_with_the_target_class(kb):
    df = pd.DataFrame({"flag": ["S0", "S0", "S0", "SF"], "rate": [0.0] * 4})
    proba = np.array([[.1, .8, .1]] * 4)
    y = np.array([1, 1, 0, 2])
    r = axiom_report(kb, df, proba, NAMES, y_true=y)["axioms"]["A1"]
    assert r["rule_precision"] == pytest.approx(2 / 3)
    assert r["violation_rate"] == 0.0


def test_rule_precision_is_nan_without_labels(kb):
    df = pd.DataFrame({"flag": ["S0"], "rate": [0.0]})
    r = axiom_report(kb, df, np.array([[.1, .8, .1]]), NAMES)["axioms"]["A1"]
    assert np.isnan(r["rule_precision"])
