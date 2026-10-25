"""Split strategies, selected by name from the config.

They return raw frames: rules and C5 metrics need the untransformed columns.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import CLASS_COL, LABEL_COL, class_index


def _top_up(kept, train, spec, minimum, seed):
    """Put back random dropped rows until every class has `minimum` rows (or all it had)."""
    if not train.index.is_unique:
        raise ValueError("train index must be unique")
    labels = np.asarray(class_index(train, spec))
    keep = train.index.isin(kept.index)
    rng = np.random.default_rng(seed)
    for c in range(len(spec.classes)):
        rows = np.flatnonzero(labels == c)
        missing = min(minimum, len(rows)) - int(keep[rows].sum())
        if missing > 0:
            keep[rng.choice(rows[~keep[rows]], missing, replace=False)] = True
    return train[keep]


def _keep_train_fraction(train, spec, fraction, seed, min_keep_per_class=0):
    """Stratified subsample of the train part only; validation and test stay whole.

    min_keep_per_class: floor per class, so rare classes survive small fractions.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
    if min_keep_per_class < 0:
        raise ValueError(f"min_keep_per_class must be >= 0, got {min_keep_per_class}")
    if fraction == 1:
        return train
    kept, _ = train_test_split(
        train, train_size=fraction, stratify=class_index(train, spec), random_state=seed
    )
    if min_keep_per_class:
        kept = _top_up(kept, train, spec, min_keep_per_class, seed)
    return kept


def _thin_classes(train, spec, fractions, seed, min_keep_per_class=0):
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
        n_keep = min(len(rows), max(1, min_keep_per_class, round(fraction * len(rows))))
        n_drop = len(rows) - n_keep
        keep[rng.choice(rows, n_drop, replace=False)] = False
    return train[keep]


def _shrink_train(train, spec, seed, train_fraction, train_class_fractions,
                  min_keep_per_class=0):
    train = _thin_classes(train, spec, train_class_fractions, seed, min_keep_per_class)
    return _keep_train_fraction(train, spec, train_fraction, seed, min_keep_per_class)


def official_test_with_stratified_val(train_df, test_df, spec, seed, val_fraction=0.15,
                                      train_fraction=1.0, train_class_fractions=None,
                                      min_keep_per_class=0):
    """Official test file; validation stratified from train."""
    if test_df is None:
        raise ValueError(f"{spec.name} has no official test file")
    train_part, val_part = train_test_split(
        train_df,
        test_size=val_fraction,
        stratify=class_index(train_df, spec),
        random_state=seed,
    )
    return (_shrink_train(train_part, spec, seed, train_fraction, train_class_fractions,
                          min_keep_per_class), val_part, test_df)


def _hold_out(train_df, labels):
    """Rows of the given fine labels, apart; every class keeps at least one other label."""
    labels = list(labels)
    unknown = sorted(set(labels) - set(train_df[LABEL_COL]))
    if unknown:
        raise ValueError(f"holdout_labels not in the data: {unknown}")
    mask = train_df[LABEL_COL].isin(labels)
    emptied = sorted(set(train_df.loc[mask, CLASS_COL]) - set(train_df.loc[~mask, CLASS_COL]))
    if emptied:
        raise ValueError(f"holdout_labels remove every row of the classes {emptied}")
    return train_df[~mask], train_df[mask]


def stratified_random(train_df, test_df, spec, seed, val_fraction=0.2,
                      test_fraction=0.2, test_seed=0, train_fraction=1.0,
                      train_class_fractions=None, holdout_labels=None, min_keep_per_class=0):
    """One frame cut into train, validation and test, stratified by class.

    The test cut uses test_seed, so every run sees the same test; only the
    train/validation cut follows the run seed (as with the official test file).
    holdout_labels: fine labels never seen in train or validation; all their rows
    are added to the test part (E9, attack variants).
    """
    if test_df is not None:
        raise ValueError(f"{spec.name} has an official test file, use another split")
    held = None
    if holdout_labels:
        train_df, held = _hold_out(train_df, holdout_labels)
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
    if held is not None:
        test = pd.concat([test, held])
    return (_shrink_train(train, spec, seed, train_fraction, train_class_fractions,
                          min_keep_per_class), val, test)


STRATEGIES = {
    "official_test_with_stratified_val": official_test_with_stratified_val,
    "stratified_random": stratified_random,
}


def split_data(strategy, train_df, test_df, spec, seed, **params):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown split {strategy!r}, known: {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](train_df, test_df, spec, seed, **params)
