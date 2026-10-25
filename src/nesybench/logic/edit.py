"""Edits of a KB for the stress runs: ablation, degradation, copies.

Config, applied to the KB a model trains on; C5 keeps reading the original file.

    kb_edit:
      exclude: [A1, group:dos]         # ids or group:<name>
      swap_target: {A2: DoS}           # id -> new target class
      tau_shift: {axioms: [A1], by: 0.05}   # fraction of the train column, 0.05 = 5 percentile points
      copies: {axiom: A1, n: 6, spread: 0.1}

Order: exclude, swap_target, tau_shift, copies.
"""
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from nesybench.data.base import REPO_ROOT
from nesybench.logic.kb import KB, Axiom, Condition, load_kb

KEYS = ("exclude", "swap_target", "tau_shift", "copies")


def _pick(kb: KB, names, where: str) -> set[str]:
    ids = set()
    for name in names:
        if str(name).startswith("group:"):
            group = name.split(":", 1)[1]
            hit = {a.id for a in kb.axioms if a.group == group}
        else:
            hit = {a.id for a in kb.axioms if a.id == name}
        if not hit:
            raise ValueError(f"{where}: {name!r} matches no axiom of {kb.version}")
        ids |= hit
    return ids


def _moved(cond: Condition, shift: float, column) -> Condition:
    """Threshold moved by `shift` as a fraction of the column (0.05 = 5 percentile points); eq stays."""
    if cond.op == "eq":
        return cond
    values = np.asarray(column, dtype=float)
    values = values[np.isfinite(values)]
    rank = float(np.mean(values <= cond.tau))
    tau = float(np.quantile(values, min(1.0, max(0.0, rank + shift))))
    return replace(cond, tau=tau)


def _shifted(axiom: Axiom, shift: float, raw_train, new_id: str | None = None) -> Axiom:
    conds = tuple(_moved(c, shift, raw_train[c.col]) for c in axiom.conditions)
    return replace(axiom, id=new_id or axiom.id, conditions=conds)


def edit_kb(kb: KB, edit: dict, raw_train=None) -> KB:
    unknown = sorted(set(edit) - set(KEYS))
    if unknown:
        raise ValueError(f"kb_edit: unknown keys {unknown}, known: {list(KEYS)}")
    axioms = list(kb.axioms)

    if edit.get("exclude"):
        gone = _pick(kb, edit["exclude"], "kb_edit.exclude")
        axioms = [a for a in axioms if a.id not in gone]
        if not axioms:
            raise ValueError("kb_edit.exclude removes every axiom")

    for ident, target in (edit.get("swap_target") or {}).items():
        _pick(kb, [ident], "kb_edit.swap_target")
        axioms = [replace(a, target=target) if a.id == ident else a for a in axioms]

    shift = edit.get("tau_shift")
    copies = edit.get("copies")
    if (shift or copies) and raw_train is None:
        raise ValueError("kb_edit: tau_shift and copies need the raw train table")

    if shift:
        names = shift.get("axioms", "all")
        ids = {a.id for a in axioms} if names == "all" else _pick(kb, names, "kb_edit.tau_shift")
        axioms = [_shifted(a, float(shift["by"]), raw_train) if a.id in ids else a for a in axioms]

    if copies:
        base = next((a for a in axioms if a.id == copies["axiom"]), None)
        if base is None:
            raise ValueError(f"kb_edit.copies: {copies['axiom']!r} not in the KB after the edits")
        n = int(copies["n"])
        spread = float(copies.get("spread", 0.1))
        for k in range(1, n + 1):
            by = spread * k / n * (-1) ** k   # alternating sides, growing
            axioms.append(_shifted(base, by, raw_train, f"{base.id}#{k}"))

    return KB(kb.version, tuple(axioms))


def kb_file(config: dict) -> Path:
    path = Path(config["kb"])
    return path if path.is_absolute() else REPO_ROOT / path


def load_run_kb(config: dict, raw_train=None) -> KB:
    """The KB a model trains on: the file of `kb`, edited if the config asks."""
    kb = load_kb(kb_file(config))
    edit = config.get("kb_edit")
    return edit_kb(kb, edit, raw_train) if edit else kb


def kb_to_dict(kb: KB) -> dict:
    """YAML-ready form, to save the KB a run actually used."""
    def cond(c: Condition) -> dict:
        if c.op == "eq":
            return {"col": c.col, "op": c.op, "value": c.value}
        out = {"col": c.col, "op": c.op, "tau": c.tau}
        if c.s is not None:
            out["s"] = c.s
        return out

    axioms = [{"id": a.id, "group": a.group, "origin": a.origin, "target": a.target,
               "train": copy.deepcopy(a.train), "all": [cond(c) for c in a.conditions]}
              for a in kb.axioms]
    return {"version": kb.version, "axioms": axioms}


def dump_kb(kb: KB, path) -> None:
    Path(path).write_text(yaml.safe_dump(kb_to_dict(kb), sort_keys=False))
