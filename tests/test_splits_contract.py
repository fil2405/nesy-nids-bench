"""Splits contract, on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from nesybench.data import get_spec
from nesybench.data.base import class_index
from nesybench.data.preprocess import preprocess
from nesybench.data.splits import split_data

SPEC = get_spec("nslkdd")
CLASSES = list(SPEC.classes)
SPLIT = "official_test_with_stratified_val"


def _toy(n_per_class=20):
    n = n_per_class * len(CLASSES)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "duration": rng.random(n),
            "src_bytes": rng.integers(0, 1000, n),
            "protocol_type": np.resize(["tcp", "udp", "icmp"], n),
            "service": np.resize(["http", "ftp"], n),
            "flag": np.resize(["SF", "S0"], n),
            "label": "x",
            "class": np.repeat(CLASSES, n_per_class),
            "difficulty": 21,
        }
    )


def _run(train_df, test_df, seed=0):
    return preprocess(*split_data(SPLIT, train_df, test_df, SPEC, seed), SPEC)


def test_feature_names_match_columns():
    s = _run(_toy(), _toy())
    assert len(s.feature_names) == s.X_train.shape[1]
    assert "duration" in s.feature_names
    assert "protocol_type_tcp" in s.feature_names


def test_class_names_match_labels():
    s = _run(_toy(), _toy())
    assert s.class_names == CLASSES
    assert set(np.unique(s.y_train)) == set(range(len(s.class_names)))


def test_unknown_class_is_an_error():
    bad = _toy()
    bad.loc[0, "class"] = "Botnet"
    with pytest.raises(ValueError, match="Botnet"):
        _run(bad, _toy())


def test_unknown_dataset_is_an_error():
    with pytest.raises(ValueError, match="nslkdd"):
        get_spec("kdd99")


def test_unknown_split_is_an_error():
    with pytest.raises(ValueError, match="official_test"):
        split_data("by_day", _toy(), _toy(), SPEC, seed=0)


def test_split_keeps_raw_rows():
    train, val, test = split_data(SPLIT, _toy(), _toy(), SPEC, seed=0)
    assert len(train) + len(val) == len(_toy())
    assert set(train.index).isdisjoint(val.index)
    assert "label" in val.columns


def test_part_gives_aligned_x_and_y():
    s = _run(_toy(), _toy())
    for name in ("train", "val", "test"):
        X, y = s.part(name)
        assert len(X) == len(y)
    assert s.part("val")[0] is s.X_val
    assert s.part("test")[1] is s.y_test


def test_unknown_part_is_an_error():
    with pytest.raises(ValueError, match="train"):
        _run(_toy(), _toy()).part("dev")


def test_raw_rows_align_with_x_and_y():
    s = _run(_toy(), _toy())
    for name in ("train", "val", "test"):
        X, y = s.part(name)
        raw = s.raw(name)
        assert len(raw) == len(X)
        assert (raw.index == np.arange(len(raw))).all()
        assert np.array_equal(class_index(raw, SPEC).to_numpy(), y)


def test_raw_rows_are_not_transformed():
    s = _run(_toy(), _toy())
    assert s.raw("train")["src_bytes"].max() > 10      # a scaled column is near 0
    assert set(s.raw("train")["flag"]) == {"SF", "S0"}  # not one-hot encoded
    assert "label" in s.raw("val").columns              # non-features are kept


def test_raw_follows_the_split():
    train, val, test = split_data(SPLIT, _toy(), _toy(), SPEC, seed=0)
    s = preprocess(train, val, test, SPEC)
    assert s.raw("val")["duration"].tolist() == val["duration"].tolist()


def test_raw_missing_or_unknown_part_is_an_error():
    s = _run(_toy(), _toy())
    with pytest.raises(ValueError, match="train"):
        s.raw("dev")
    bare = s._replace(raw_val=None)
    with pytest.raises(ValueError, match="raw rows"):
        bare.raw("val")