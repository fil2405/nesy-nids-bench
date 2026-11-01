from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .base import DatasetSpec, class_index

PARTS = ("train", "val", "test")


class Splits(NamedTuple):
    """Build by keyword: new fields are appended."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    transformer: ColumnTransformer
    feature_names: list[str]
    class_names: list[str]
    raw_train: pd.DataFrame | None = None  # untransformed rows, same order as X
    raw_val: pd.DataFrame | None = None
    raw_test: pd.DataFrame | None = None

    def part(self, name: str):
        """(X, y) for "train", "val" or "test"."""
        if name not in PARTS:
            raise ValueError(f"unknown part {name!r}, known: {list(PARTS)}")
        return getattr(self, f"X_{name}"), getattr(self, f"y_{name}")

    def raw(self, name: str) -> pd.DataFrame:
        """Raw rows of a part, for the rules and C5."""
        if name not in PARTS:
            raise ValueError(f"unknown part {name!r}, known: {list(PARTS)}")
        frame = getattr(self, f"raw_{name}")
        if frame is None:
            raise ValueError("these splits were built without raw rows")
        return frame


def _split_xy(df, spec: DatasetSpec):
    X = df.drop(columns=[c for c in spec.non_features if c in df.columns])
    return X, class_index(df, spec).to_numpy()


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


NUMERIC_TRANSFORMS = {None: None, "signed_log1p": signed_log1p}


def _numeric_step(numeric_transform):
    if numeric_transform not in NUMERIC_TRANSFORMS:
        raise ValueError(f"unknown numeric_transform {numeric_transform!r}, "
                         f"known: {[k for k in NUMERIC_TRANSFORMS if k]}")
    func = NUMERIC_TRANSFORMS[numeric_transform]
    if func is None:
        return StandardScaler()
    return Pipeline([("log", FunctionTransformer(func, feature_names_out="one-to-one")),
                     ("scale", StandardScaler())])


def preprocess(train_df, val_df, test_df, spec: DatasetSpec, numeric_transform=None):
    """Transformer fitted on train only. numeric_transform: None or "signed_log1p", before the scaler."""
    X_train, y_train = _split_xy(train_df, spec)
    X_val, y_val = _split_xy(val_df, spec)
    X_test, y_test = _split_xy(test_df, spec)

    cat_cols = list(spec.categorical)
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    transformer = ColumnTransformer(
        [
            ("num", _numeric_step(numeric_transform), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    transformer.fit(X_train)

    return Splits(
        X_train=np.asarray(transformer.transform(X_train)),
        X_val=np.asarray(transformer.transform(X_val)),
        X_test=np.asarray(transformer.transform(X_test)),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        transformer=transformer,
        feature_names=list(transformer.get_feature_names_out()),
        class_names=list(spec.classes),
        raw_train=train_df.reset_index(drop=True),
        raw_val=val_df.reset_index(drop=True),
        raw_test=test_df.reset_index(drop=True),
    )