"""What the rest of the code may know about a dataset.

Every loader returns a DataFrame with the raw features plus two columns:
LABEL_COL (fine label, e.g. "neptune") and CLASS_COL (the class the models
predict, e.g. "DoS"). Everything else a model needs comes from DatasetSpec.
"""
import os
from dataclasses import dataclass
from pathlib import Path

LABEL_COL = "label"
CLASS_COL = "class"

# src/nesybench/data/base.py -> repository root is three levels above `data/`.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("NESYBENCH_DATA", REPO_ROOT / "data" / "raw"))


@dataclass(frozen=True)
class DatasetSpec:
    """Fixed description of a dataset.

    classes:      class names in a fixed order; class i is index i in y,
                  in the metrics and in every table.
    categorical:  columns that are one-hot encoded.
    non_features: columns that are never network inputs (the answer and
                  metadata such as difficulty, addresses or timestamps).
    fixed_test:   True if the dataset ships an official test file,
                  False if the test set has to be cut from the data.
    """

    name: str
    classes: tuple[str, ...]
    categorical: tuple[str, ...]
    non_features: tuple[str, ...]
    fixed_test: bool


def class_index(df, spec: DatasetSpec):
    """CLASS_COL as integers in the order of spec.classes; unknown names are an error."""
    unknown = sorted(set(df[CLASS_COL]) - set(spec.classes))
    if unknown:
        raise ValueError(f"classes not in {spec.name} spec: {unknown}")
    return df[CLASS_COL].map({c: i for i, c in enumerate(spec.classes)})
