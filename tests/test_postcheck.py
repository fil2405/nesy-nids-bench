import numpy as np
import pandas as pd
import pytest

from nesybench import run
from nesybench.logic.kb import load_kb
from nesybench.logic.postcheck import enforce, explanation_metrics, run_postcheck

NAMES = ["Normal", "DoS", "Probe"]

KB_YAML = """
version: test_v1
axioms:
  - {id: A1, group: dos, origin: mechanism, target: DoS,
     all: [{col: flag, op: eq, value: S0}]}
  - {id: A2, group: dataset, origin: data, target: Probe,
     all: [{col: rate, op: gt, tau: 0.5, s: 0.1}]}
"""


@pytest.fixture
def kb(tmp_path):
    path = tmp_path / "kb.yaml"
    path.write_text(KB_YAML)
    return load_kb(path)


# rows: A1 only, A2 only, both, none, none
DF = pd.DataFrame({"flag": ["S0", "SF", "S0", "SF", "SF"], "rate": [0.0, 0.9, 0.9, 0.0, 0.0]})
PROBA = np.array([[.1, .8, .1], [.8, .1, .1], [.1, .1, .8], [.1, .8, .1], [.8, .1, .1]])


def test_annotate_keeps_predictions_and_reports_explanations(kb):
    after, rep = run_postcheck(kb, DF, PROBA, NAMES, "annotate")
    assert after.tolist() == [1, 0, 2, 1, 0]
    assert rep["n_changed"] == 0 and rep["n_rows_with_rule"] == 3
    ex = rep["explanations"]
    # alerts: rows 0, 2, 3; rule fires on 0 and 2; row 0 agrees (A1 -> DoS), row 2 agrees (A2 -> Probe)
    assert ex["n_alerts"] == 3
    assert ex["coverage"] == pytest.approx(2 / 3)
    assert ex["fidelity"] == 1.0
    assert ex["conciseness"] == pytest.approx(1.5)


def test_enforce_overrides_where_a_rule_fires_and_first_axiom_wins(kb):
    after, rep = run_postcheck(kb, DF, PROBA, NAMES, "enforce")
    # row 0 -> DoS (A1), row 1 -> Probe (A2), row 2 -> DoS (A1 first), rows 3, 4 untouched
    assert after.tolist() == [1, 2, 1, 1, 0]
    assert rep["n_changed"] == 2 and rep["rows"] == 5
    assert rep["seconds"] >= 0


def test_enforce_helper_on_matrices():
    fired = np.array([[True, True], [False, True], [False, False]])
    assert enforce(np.array([0, 0, 0]), fired, np.array([1, 2])).tolist() == [1, 2, 0]


def test_fidelity_counts_a_rule_that_points_elsewhere():
    fired = np.array([[True, False], [True, False]])
    ex = explanation_metrics(np.array([2, 1]), fired, np.array([1, 2]))
    assert ex["coverage"] == 1.0 and ex["fidelity"] == 0.5


def test_no_alerts_gives_nan_not_zero(kb):
    _, rep = run_postcheck(kb, DF, np.tile([.8, .1, .1], (5, 1)), NAMES, "annotate")
    ex = rep["explanations"]
    assert ex["n_alerts"] == 0 and np.isnan(ex["coverage"]) and np.isnan(ex["fidelity"])


def test_unknown_mode_is_refused(kb):
    with pytest.raises(ValueError, match="postcheck mode"):
        run_postcheck(kb, DF, PROBA, NAMES, "block")


def test_config_modes_are_checked_before_any_run(kb):
    assert run._postcheck_modes({"postcheck": "enforce"}, kb) == ("enforce",)
    assert run._postcheck_modes({"postcheck": ["annotate", "enforce"]}, kb) == ("annotate", "enforce")
    assert run._postcheck_modes({}, None) == ()
    with pytest.raises(ValueError, match="needs a kb"):
        run._postcheck_modes({"postcheck": "annotate"}, None)
    with pytest.raises(ValueError, match="unknown modes"):
        run._postcheck_modes({"postcheck": ["x"]}, kb)


class _Splits:
    class_names = NAMES

    def __init__(self):
        self.y = np.array([1, 0, 2, 1, 0])

    def part(self, name):
        return None, self.y

    def raw(self, name):
        return DF


class _Model:
    def predict_proba(self, splits, part):
        return PROBA


def test_evaluate_adds_the_postcheck_block(kb):
    metrics = run._evaluate(_Model(), _Splits(), "val", {}, kb, ("annotate", "enforce"))
    block = metrics["postcheck"]
    assert set(block) == {"annotate", "enforce"}
    assert block["annotate"]["macro_f1"] == block["annotate"]["macro_f1_before"] == metrics["macro_f1"]
    # the toy model is perfect, so overriding it with the rules can only cost
    assert block["enforce"]["macro_f1_before"] == 1.0
    assert block["enforce"]["macro_f1"] < 1.0
    assert "postcheck" not in run._evaluate(_Model(), _Splits(), "val", {}, kb)
