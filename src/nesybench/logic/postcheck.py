"""Symbolic post-check on the predictions of any model (E12, E17).

annotate: predictions stay, every alert gets the rules that fire on its row.
enforce: where a rule fires the prediction becomes its target; the first axiom
in KB order wins a conflict (same rule as the rule-only arm).
"""
from time import perf_counter

import numpy as np

from nesybench.logic.c5 import THRESHOLD
from nesybench.logic.kb import KB, antecedents, target_index

MODES = ("annotate", "enforce")
BENIGN = 0


def fired_rules(kb: KB, raw_df, threshold: float = THRESHOLD) -> np.ndarray:
    """Rows x axioms, True where the crisp antecedent holds."""
    return antecedents(kb, raw_df, crisp=True) >= threshold


def enforce(pred, fired, targets) -> np.ndarray:
    out = np.array(pred, copy=True)
    for j in reversed(range(len(targets))):   # last write is the first axiom
        out[fired[:, j]] = targets[j]
    return out


def explanation_metrics(pred, fired, targets) -> dict:
    """Over the alerts (prediction not benign): coverage, fidelity, conciseness.

    coverage: share of alerts with at least one rule firing.
    fidelity: among those, share where a firing rule has the predicted class as target.
    conciseness: mean number of firing rules per covered alert.
    """
    pred = np.asarray(pred)
    alert = pred != BENIGN
    n_alerts = int(alert.sum())
    out = {"n_alerts": n_alerts, "coverage": float("nan"), "fidelity": float("nan"),
           "conciseness": float("nan")}
    if n_alerts == 0:
        return out
    covered = alert & fired.any(axis=1)
    n_covered = int(covered.sum())
    out["coverage"] = n_covered / n_alerts
    if n_covered:
        agree = (fired & (np.asarray(targets)[None, :] == pred[:, None])).any(axis=1)
        out["fidelity"] = float(agree[covered].mean())
        out["conciseness"] = float(fired[covered].sum(axis=1).mean())
    return out


def run_postcheck(kb: KB, raw_df, proba, class_names, mode: str,
                  threshold: float = THRESHOLD) -> tuple[np.ndarray, dict]:
    """Predictions after the check and a report. The timer covers rules and override only."""
    if mode not in MODES:
        raise ValueError(f"postcheck mode {mode!r}, known: {list(MODES)}")
    pred = np.asarray(proba).argmax(axis=1)
    targets = target_index(kb, class_names)
    start = perf_counter()
    fired = fired_rules(kb, raw_df, threshold)
    after = enforce(pred, fired, targets) if mode == "enforce" else pred
    seconds = perf_counter() - start
    report = {
        "mode": mode,
        "kb_version": kb.version,
        "rows": int(len(pred)),
        "seconds": seconds,
        "rows_per_second": len(pred) / seconds if seconds > 0 else float("nan"),
        "n_rows_with_rule": int(fired.any(axis=1).sum()),
        "n_changed": int((after != pred).sum()),
        "explanations": explanation_metrics(pred, fired, targets),
    }
    return after, report
