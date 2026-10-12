import numpy as np
import pytest

from nesybench.data.nslkdd import DATA_DIR, FILES, SPEC, load_nslkdd
from nesybench.data.preprocess import preprocess
from nesybench.data.splits import split_data

N_NUM = 38
SPLIT = "official_test_with_stratified_val"


def _run(frames, seed):
    return preprocess(*split_data(SPLIT, *frames, SPEC, seed), SPEC)


@pytest.fixture(scope="module")
def frames():
    for name in FILES.values():
        if not (DATA_DIR / name).exists():
            pytest.skip(f"dataset not found in {DATA_DIR}")
    return load_nslkdd("train"), load_nslkdd("test")


@pytest.fixture(scope="module")
def splits(frames):
    return _run(frames, seed=0)


def test_shapes(splits):
    assert splits.X_train.shape[1] == 122
    assert splits.X_val.shape[1] == 122
    assert splits.X_test.shape == (22544, 122)
    assert len(splits.y_train) + len(splits.y_val) == 125973


def test_val_has_all_classes(splits):
    assert set(splits.y_val.tolist()) == {0, 1, 2, 3, 4}
    assert (splits.y_val == 4).sum() >= 7


def test_scaler_fitted_on_train_only(splits):
    assert np.allclose(splits.X_train[:, :N_NUM].mean(axis=0), 0, atol=1e-6)
    assert not np.allclose(splits.X_val[:, :N_NUM].mean(axis=0), 0, atol=1e-6)


def test_same_seed_same_split(frames, splits):
    again = _run(frames, seed=0)
    assert np.array_equal(splits.X_val, again.X_val)
    assert np.array_equal(splits.y_val, again.y_val)


def test_different_seed_different_split(frames, splits):
    other = _run(frames, seed=1)
    assert not np.array_equal(splits.y_val, other.y_val)
