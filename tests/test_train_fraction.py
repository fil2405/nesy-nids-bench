"""train_fraction keeps a stratified share of the train part and nothing else."""
import numpy as np
import pandas as pd
import pytest

from nesybench.data import get_spec
from nesybench.data.splits import split_data

CASES = [
    ("nslkdd", "official_test_with_stratified_val", True),
    ("cicids2017", "stratified_random", False),
]


def _frame(spec, n_per_class=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.random(n_per_class * len(spec.classes)),
                         "class": np.repeat(spec.classes, n_per_class)})


def _split(name, strategy, official, seed=0, **params):
    spec = get_spec(name)
    df = _frame(spec)
    return spec, split_data(strategy, df, _frame(spec) if official else None, spec,
                            seed, **params)


@pytest.mark.parametrize("name, strategy, official", CASES)
def test_default_is_the_whole_train_part(name, strategy, official):
    _, whole = _split(name, strategy, official)
    _, one = _split(name, strategy, official, train_fraction=1.0)
    for a, b in zip(whole, one):
        assert a.index.equals(b.index)


@pytest.mark.parametrize("name, strategy, official", CASES)
def test_train_shrinks_and_stays_stratified(name, strategy, official):
    spec, (whole, *_) = _split(name, strategy, official)
    _, (train, *_) = _split(name, strategy, official, train_fraction=0.25)
    assert len(train) == pytest.approx(0.25 * len(whole), abs=len(spec.classes))
    assert set(train.index) <= set(whole.index)
    counts = train["class"].value_counts()
    assert set(counts.index) == set(spec.classes)
    assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize("name, strategy, official", CASES)
def test_validation_and_test_do_not_change(name, strategy, official):
    _, (_, val, test) = _split(name, strategy, official)
    _, (_, val_f, test_f) = _split(name, strategy, official, train_fraction=0.1)
    assert val.index.equals(val_f.index) and test.index.equals(test_f.index)


@pytest.mark.parametrize("name, strategy, official", CASES)
def test_subsample_follows_the_seed(name, strategy, official):
    _, (a, *_) = _split(name, strategy, official, seed=1, train_fraction=0.5)
    _, (b, *_) = _split(name, strategy, official, seed=1, train_fraction=0.5)
    _, (c, *_) = _split(name, strategy, official, seed=2, train_fraction=0.5)
    assert a.index.equals(b.index)
    assert not set(a.index) == set(c.index)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5])
def test_bad_fractions_are_refused(fraction):
    with pytest.raises(ValueError, match="train_fraction"):
        _split("cicids2017", "stratified_random", False, train_fraction=fraction)
