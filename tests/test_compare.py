import numpy as np
import pytest

from nesybench.compare import NEMENYI_Q05, friedman_nemenyi, holm, paired


def test_paired_all_positive_ten_seeds():
    a = np.zeros(10)
    b = np.arange(1, 11) / 100
    r = paired(a, b)
    assert r.n == 10 and r.wins == 10 and r.ties == 0
    assert r.p == pytest.approx(2 / 1024)          # exact two-sided, all signs equal
    assert r.low <= r.median <= r.high and r.median == pytest.approx(0.055)


def test_paired_identical_runs():
    r = paired(np.ones(5), np.ones(5))
    assert r.p == 1.0 and r.median == 0.0 and r.wins == 0 and r.ties == 5


def test_paired_sign_and_nan_dropped():
    r = paired([0.9, 0.8, 0.7, np.nan], [0.8, 0.7, 0.6, 0.5])
    assert r.n == 3 and r.median < 0 and r.wins == 0


def test_paired_checks_shapes():
    with pytest.raises(ValueError, match="same length"):
        paired([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="two seeds"):
        paired([1], [2])


def test_paired_is_reproducible():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=10), rng.normal(size=10)
    assert paired(a, b) == paired(a, b)


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_holm_is_monotone_and_capped():
    out = holm([0.04, 0.2, 0.5, 0.001])
    assert max(out) <= 1.0 and out[3] == pytest.approx(0.004)
    assert out[0] >= out[3]


def test_friedman_ranks_and_cd():
    rng = np.random.default_rng(0)
    base = rng.normal(size=10) * 0.001
    scores = {"a": base + 0.9, "b": base + 0.8, "c": base + 0.7}
    r = friedman_nemenyi(scores)
    assert r.mean_ranks == [1.0, 2.0, 3.0]
    assert r.friedman_p < 0.001
    assert r.cd == pytest.approx(NEMENYI_Q05[1] * np.sqrt(3 * 4 / (6 * 10)))


def test_friedman_ties_share_the_rank():
    r = friedman_nemenyi({"a": [1, 1, 1], "b": [1, 1, 1], "c": [0, 0, 0]})
    assert r.mean_ranks == [1.5, 1.5, 3.0]


def test_friedman_needs_three_arms():
    with pytest.raises(ValueError, match="3 to"):
        friedman_nemenyi({"a": [1, 2], "b": [2, 1]})
