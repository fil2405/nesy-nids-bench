"""Loaders return raw features plus LABEL_COL (fine label) and CLASS_COL (target class)."""
import os
from dataclasses import dataclass
from pathlib import Path

LABEL_COL = "label"
CLASS_COL = "class"

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("NESYBENCH_DATA", REPO_ROOT / "data" / "raw"))


@dataclass(frozen=True)
class DatasetSpec:
    """`classes` order defines the label index everywhere.
    `non_features` are never model inputs (targets, metadata, addresses, time).
    `fixed_test`: the dataset ships an official test file."""

    name: str
    classes: tuple[str, ...]
    categorical: tuple[str, ...]
    non_features: tuple[str, ...]
    fixed_test: bool


def class_index(df, spec: DatasetSpec):
    """CLASS_COL as indices into spec.classes."""
    unknown = sorted(set(df[CLASS_COL]) - set(spec.classes))
    if unknown:
        raise ValueError(f"classes not in {spec.name} spec: {unknown}")
    return df[CLASS_COL].map({c: i for i, c in enumerate(spec.classes)})
