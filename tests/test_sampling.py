import numpy as np
import pytest

from nesybench.models.sampling import class_quotas, sample_batches


def _y():
    return np.repeat(np.arange(5), [600, 300, 80, 15, 5])


def test_every_batch_has_every_class():
    y = _y()
    for batch in sample_batches(y, 5, 64, power=0.0, min_per_class=2):
        assert np.all(np.bincount(y[batch], minlength=5) >= 2)


def test_epoch_has_ceil_n_over_batch_batches():
    assert len(list(sample_batches(_y(), 5, 64))) == 16


def test_power_zero_follows_the_data():
    q = class_quotas([600, 300, 80, 15, 5], 100, 0.0, 1)
    assert q[0] > q[1] > q[2] > q[3] >= q[4]
    assert q.sum() == 100


def test_power_one_is_uniform():
    q = class_quotas([600, 300, 80, 15, 5], 100, 1.0, 1)
    assert set(q) == {20}


def test_same_seed_same_batches():
    a = list(sample_batches(_y(), 5, 64, seed=3))
    b = list(sample_batches(_y(), 5, 64, seed=3))
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_rare_class_is_repeated_when_the_quota_exceeds_it():
    y = _y()
    batch = next(sample_batches(y, 5, 100, power=1.0))
    assert np.bincount(y[batch], minlength=5)[4] == 20


def test_class_without_examples_is_an_error():
    with pytest.raises(ValueError):
        next(sample_batches(np.array([0, 0, 1, 1]), 3, 8))