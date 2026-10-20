"""Split strategies, selected by name from the config.

They return raw frames: rules and C5 metrics need the untransformed columns.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from .base import class_index


def _keep_train_fraction(train, spec, fraction, seed):
    """Stratified subsample of the train part only; validation and test stay whole."""
    if not 0 < fraction <= 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return train
    kept, _ = train_test_split(
        train, train_size=fraction, stratify=class_index(train, spec), random_state=seed
    )
    return kept


def _thin_classes(train, spec, fractions, seed):
    """Random subsample of some classes of the train part only (e.g. Normal for the tuning)."""
    if not fractions:
        return train
    unknown = set(fractions) - set(spec.classes)
    if unknown:
        raise ValueError(f"train_class_fractions: unknown classes {sorted(unknown)}")
    labels = np.asarray(class_index(train, spec))
    keep = np.ones(len(train), dtype=bool)
    rng = np.random.default_rng(seed)
    for name, fraction in fractions.items():
        if not 0 < fraction <= 1:
            raise ValueError(f"train_class_fractions[{name}] must be in (0, 1], got {fraction}")
        rows = np.flatnonzero(labels == spec.classes.index(name))
        n_drop = len(rows) - max(1, round(fraction * len(rows))) if len(rows) else 0
        keep[rng.choice(rows, n_drop, replace=False)] = False
    return train[keep]


def _shrink_train(train, spec, seed, train_fraction, train_class_fractions):
    train = _thin_classes(train, spec, train_class_fractions, seed)
    return _keep_train_fraction(train, spec, train_fraction, seed)


def official_test_with_stratified_val(train_df, test_df, spec, seed, val_fraction=0.15,
                                      train_fraction=1.0, train_class_fractions=None):
    """Official test file; validation stratified from train."""
    if test_df is None:
        raise ValueError(f"{spec.name} has no official test file")
    train_part, val_part = train_test_split(
        train_df,
        test_size=val_fraction,
        stratify=class_index(train_df, spec),
        random_state=seed,
    )
    return (_shrink_train(train_part, spec, seed, train_fraction, train_class_fractions),
            val_part, test_df)


def stratified_random(train_df, test_df, spec, seed, val_fraction=0.2,
                      test_fraction=0.2, test_seed=0, train_fraction=1.0,
                      train_class_fractions=None):
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
    return _shrink_train(train, spec, seed, train_fraction, train_class_fractions), val, test


STRATEGIES = {
    "official_test_with_stratified_val": official_test_with_stratified_val,
    "stratified_random": stratified_random,
}


def split_data(strategy, train_df, test_df, spec, seed, **params):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown split {strategy!r}, known: {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](train_df, test_df, spec, seed, **params)
