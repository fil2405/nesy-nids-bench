"""C5: axiom-level satisfaction and violation rate for any model."""
import numpy as np

from nesybench.logic.kb import KB, antecedents, target_index

THRESHOLD = 0.5  # sigmoid midpoint: the antecedent holds at or above it


def axiom_report(kb: KB, raw_df, proba, class_names, crisp: bool = False,
                 threshold: float = THRESHOLD) -> dict:
    """Per axiom, on the rows where the antecedent holds.

    violation: argmax differs from the target class. satisfaction: p-mean-error
    (p=2) of the Reichenbach implication degree. Both nan if no row is covered.
    """
    proba = np.asarray(proba, dtype=float)
    a = antecedents(kb, raw_df, crisp)
    targets = target_index(kb, class_names)
    pred = proba.argmax(axis=1)
    axioms = {}
    for j, axiom in enumerate(kb.axioms):
        covered = a[:, j] >= threshold
        n = int(covered.sum())
        if n == 0:
            axioms[axiom.id] = {"n_covered": 0, "coverage": 0.0, "n_violations": 0,
                                "violation_rate": float("nan"),
                                "satisfaction": float("nan")}
            continue
        aj = a[covered, j]
        implied = 1.0 - aj + aj * proba[covered, targets[j]]
        n_viol = int((pred[covered] != targets[j]).sum())
        axioms[axiom.id] = {
            "n_covered": n,
            "coverage": n / len(proba),
            "n_violations": n_viol,
            "violation_rate": n_viol / n,
            "satisfaction": float(1.0 - np.sqrt(np.mean((1.0 - implied) ** 2))),
        }
    return {
        "kb_version": kb.version,
        "threshold": threshold,
        "axioms": axioms,
        "mean_violation_rate": _mean(axioms, "violation_rate"),
        "mean_satisfaction": _mean(axioms, "satisfaction"),
    }


def _mean(axioms: dict, key: str) -> float:
    values = [v[key] for v in axioms.values() if not np.isnan(v[key])]
    return float(np.mean(values)) if values else float("nan")
