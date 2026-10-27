"""Simple perturbations of a raw test part (E10): feature noise and rule-aware evasion.

noise:S    every numeric value is scaled by (1 + S * N(0, 1)), clipped to the train range of its
           column and rounded where the train column is integer: zeros and constant columns
           stay as they are, bytes and counts get a relative error, not a heavy-tail std.
evade:D    on the rows of an axiom's target class where it fires, its first numeric
           condition is moved D*|tau| past tau (D*1 when tau is 0), so the rule stops firing.
"""
import numpy as np

from nesybench.logic.kb import target_index
from nesybench.logic.postcheck import fired_rules

KINDS = ("noise", "evade")


def parse(spec: str) -> tuple[str, float]:
    kind, _, level = spec.partition(":")
    if kind not in KINDS or not level:
        raise ValueError(f"perturbation {spec!r}, expected noise:S or evade:D")
    value = float(level)
    if value <= 0:
        raise ValueError(f"perturbation {spec!r}: the level must be > 0")
    return kind, value


def column_bounds(train, columns) -> dict:
    """Per column: (min, max, integer) on the train part."""
    out = {}
    for col in columns:
        v = train[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        out[col] = (float(v.min()), float(v.max()), bool(np.all(v == np.round(v))))
    return out


def noise(raw, bounds: dict, sigma: float, seed: int):
    rng = np.random.default_rng(seed)
    out = raw.copy()
    for col, (low, high, integer) in bounds.items():
        v = out[col].to_numpy(dtype=float)
        v = np.clip(v * (1.0 + sigma * rng.standard_normal(len(v))), low, high)
        out[col] = np.round(v) if integer else v
    return out


def evade(raw, y, kb, class_names, delta: float):
    """Perturbed copy and the number of rows moved."""
    out = raw.copy()
    targets = target_index(kb, class_names)
    fired = fired_rules(kb, raw)
    y = np.asarray(y)
    moved = 0
    for j, axiom in enumerate(kb.axioms):
        cond = next((c for c in axiom.conditions if c.op in ("gt", "le")), None)
        if cond is None:
            continue
        rows = fired[:, j] & (y == targets[j])
        if not rows.any():
            continue
        step = delta * (abs(cond.tau) or 1.0)
        out[cond.col] = out[cond.col].astype(float)
        out.loc[rows, cond.col] = cond.tau - step if cond.op == "gt" else cond.tau + step
        moved += int(rows.sum())
    return out, moved
