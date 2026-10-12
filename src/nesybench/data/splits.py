"""Split strategies.

A strategy takes the raw frames and returns (train_df, val_df, test_df), still
raw: the rules and the C5 metrics read raw columns, the network reads the
transformed ones. The config names the strategy (`split:` in configs/).
"""
from sklearn.model_selection import train_test_split

from .base import class_index


def official_test_with_stratified_val(train_df, test_df, spec, seed, val_fraction=0.15):
    """Official test file as test; validation cut from train, stratified by class."""
    if test_df is None:
        raise ValueError(f"{spec.name} has no official test file")
    train_part, val_part = train_test_split(
        train_df,
        test_size=val_fraction,
        stratify=class_index(train_df, spec),
        random_state=seed,
    )
    return train_part, val_part, test_df


STRATEGIES = {
    "official_test_with_stratified_val": official_test_with_stratified_val,
}


def split_data(strategy, train_df, test_df, spec, seed, **params):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown split {strategy!r}, known: {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](train_df, test_df, spec, seed, **params)
