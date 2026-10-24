"""Knowledge base: YAML axioms -> degrees of truth on raw columns."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.special import expit

OPS = ("eq", "gt", "le")
GROUPS = ("domain", "dos", "dataset")
ORIGINS = ("mechanism", "data", "internship")


@dataclass(frozen=True)
class Condition:
    col: str
    op: str
    tau: float | None = None
    s: float | None = None
    value: str | None = None


@dataclass(frozen=True)
class Axiom:
    id: str
    group: str
    origin: str
    target: str
    conditions: tuple[Condition, ...]
    train: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class KB:
    version: str
    axioms: tuple[Axiom, ...]

    @property
    def ids(self) -> list[str]:
        return [a.id for a in self.axioms]


def _condition(raw: dict, where: str) -> Condition:
    op = raw.get("op")
    if op not in OPS:
        raise ValueError(f"{where}: op {op!r}, known: {list(OPS)}")
    if "col" not in raw:
        raise ValueError(f"{where}: missing col")
    if op == "eq":
        if raw.get("value") is None:
            raise ValueError(f"{where}: eq needs value")
        return Condition(raw["col"], op, value=str(raw["value"]))
    if raw.get("tau") is None:
        raise ValueError(f"{where}: {op} needs tau")
    s = raw.get("s")
    if s is not None and s <= 0:
        raise ValueError(f"{where}: s must be > 0, got {s}")
    return Condition(raw["col"], op, tau=float(raw["tau"]),
                     s=None if s is None else float(s))


def _axiom(raw: dict) -> Axiom:
    ident = raw.get("id")
    if not ident:
        raise ValueError(f"axiom without id: {raw}")
    for key, known in (("group", GROUPS), ("origin", ORIGINS)):
        if raw.get(key) not in known:
            raise ValueError(f"{ident}: {key} {raw.get(key)!r}, known: {list(known)}")
    if not raw.get("target"):
        raise ValueError(f"{ident}: missing target")
    if not raw.get("all"):
        raise ValueError(f"{ident}: empty condition list")
    conditions = tuple(_condition(c, f"{ident}[{i}]") for i, c in enumerate(raw["all"]))
    return Axiom(ident, raw["group"], raw["origin"], raw["target"], conditions,
                 raw.get("train") or {})


def load_kb(path) -> KB:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "version" not in raw or not raw.get("axioms"):
        raise ValueError(f"{path}: needs version and a non-empty axioms list")
    axioms = tuple(_axiom(a) for a in raw["axioms"])
    ids = [a.id for a in axioms]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"{path}: duplicate ids {dup}")
    return KB(str(raw["version"]), axioms)


def condition_degree(df, cond: Condition, crisp: bool = False,
                     s_scale: float = 1.0) -> np.ndarray:
    """Degree in [0, 1] per row. Graded: sigmoid of the distance to tau over s * s_scale."""
    x = df[cond.col]
    if cond.op == "eq":
        return (x == cond.value).to_numpy(dtype=float)
    v = x.to_numpy(dtype=float)
    if crisp:
        return (v > cond.tau if cond.op == "gt" else v <= cond.tau).astype(float)
    if cond.s is None:
        raise ValueError(f"{cond.col} {cond.op} {cond.tau}: s not set, "
                         "set it in the KB or ask for crisp")
    d = v - cond.tau if cond.op == "gt" else cond.tau - v
    return expit(d / (cond.s * s_scale))


def antecedents(kb: KB, df, crisp: bool = False, s_scale: float = 1.0) -> np.ndarray:
    """(n_rows, n_axioms): product of the condition degrees of each axiom."""
    if s_scale <= 0:
        raise ValueError(f"s_scale must be > 0, got {s_scale}")
    cols = {c.col for a in kb.axioms for c in a.conditions}
    missing = sorted(cols - set(df.columns))
    if missing:
        raise KeyError(f"columns not in the data: {missing}")
    out = np.empty((len(df), len(kb.axioms)))
    for j, axiom in enumerate(kb.axioms):
        out[:, j] = np.prod([condition_degree(df, c, crisp, s_scale) for c in axiom.conditions],
                            axis=0)
    return out


def target_index(kb: KB, class_names) -> np.ndarray:
    """Class index of each axiom's consequent."""
    names = list(class_names)
    unknown = sorted({a.target for a in kb.axioms} - set(names))
    if unknown:
        raise ValueError(f"targets not in the classes: {unknown}, classes: {names}")
    return np.array([names.index(a.target) for a in kb.axioms])
