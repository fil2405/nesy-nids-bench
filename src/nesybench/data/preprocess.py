from typing import NamedTuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CAT_COLS = ["protocol_type", "service", "flag"]
DROP_COLS = ["label", "class", "difficulty"]


class Splits(NamedTuple):
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    transformer: ColumnTransformer


def _split_xy(df):
    X = df.drop(columns=DROP_COLS)
    y = df["class"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return X, y


def preprocess(train_df, test_df, seed, val_fraction=0.15):
    """Split train into train/validation, and fit the transformer."""
    X, y = _split_xy(train_df)
    X_test, y_test = _split_xy(test_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )

    num_cols = [c for c in X.columns if c not in CAT_COLS]
    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ],
        sparse_threshold=0,
    )
    transformer.fit(X_train)

    return Splits(
        transformer.transform(X_train),
        transformer.transform(X_val),
        transformer.transform(X_test),
        y_train,
        y_val,
        y_test,
        transformer,
    )