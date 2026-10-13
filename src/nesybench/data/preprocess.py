from typing import NamedTuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    def part(self, name: str):
        """(X, y) for "train", "val" or "test"."""
        if name not in PARTS:
            raise ValueError(f"unknown part {name!r}, known: {list(PARTS)}")
        return getattr(self, f"X_{name}"), getattr(self, f"y_{name}")


def _split_xy(df, spec: DatasetSpec):
    X = df.drop(columns=[c for c in spec.non_features if c in df.columns])
    return X, class_index(df, spec).to_numpy()


def preprocess(train_df, val_df, test_df, spec: DatasetSpec):
    """Transformer fitted on train only."""
    X_train, y_train = _split_xy(train_df, spec)
    X_val, y_val = _split_xy(val_df, spec)
    X_test, y_test = _split_xy(test_df, spec)

    cat_cols = list(spec.categorical)
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
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
    )
