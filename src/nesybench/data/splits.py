"""Split strategies, selected by name from the config.

They return raw frames: rules and C5 metrics need the untransformed columns.
"""
from sklearn.model_selection import train_test_split

from .base import class_index


def official_test_with_stratified_val(train_df, test_df, spec, seed, val_fraction=0.15):
    """Official test file; validation stratified from train."""
    if test_df is None:
        raise ValueError(f"{spec.name} has no official test file")
    train_part, val_part = train_test_split(
        train_df,
        test_size=val_fraction,
        stratify=class_index(train_df, spec),
        random_state=seed,
    )
    return train_part, val_part, test_df


def stratified_random(train_df, test_df, spec, seed, val_fraction=0.2,
                      test_fraction=0.2, test_seed=0):
    """One frame cut into train, validation and test, stratified by class.

    The test cut uses test_seed, so every run sees the same test; only the
    train/validation cut follows the run seed (as with the official test file).
    """
    if test_df is not None:
        raise ValueError(f"{spec.name} has an official test file, use another split")
    rest, test = train_test_split(
        train_df,
        test_size=test_fraction,
        stratify=class_index(train_df, spec),
        random_state=test_seed,
    )
    train, val = train_test_split(
        rest,
        test_size=val_fraction / (1 - test_fraction),
        stratify=class_index(rest, spec),
        random_state=seed,
    )
    return train, val, test


STRATEGIES = {
    "official_test_with_stratified_val": official_test_with_stratified_val,
    "stratified_random": stratified_random,
}


def split_data(strategy, train_df, test_df, spec, seed, **params):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown split {strategy!r}, known: {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](train_df, test_df, spec, seed, **params)
