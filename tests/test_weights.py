"""Class weights for the cross-entropy baseline."""
import numpy as np
import pytest

from nesybench.models.weights import class_weights

COUNTS = [900, 90, 9, 1]


def test_power_one_is_the_balanced_weight():
    n, k = sum(COUNTS), len(COUNTS)
    assert class_weights(COUNTS, 1.0) == pytest.approx(n / (k * np.array(COUNTS)))


def test_power_zero_is_no_weighting():
    assert class_weights(COUNTS, 0.0) == pytest.approx(np.ones(4))


@pytest.mark.parametrize("power", [0.0, 0.25, 0.5, 1.0, 2.0])
def test_mean_weight_over_examples_is_one(power):
    weights = class_weights(COUNTS, power)
    assert (weights * COUNTS).sum() / sum(COUNTS) == pytest.approx(1.0)


def test_rarer_classes_never_weigh_less():
    weights = class_weights(COUNTS, 0.5)
    assert all(a < b for a, b in zip(weights, weights[1:]))


def test_a_smaller_power_gives_milder_weights():
    assert class_weights(COUNTS, 0.5)[-1] < class_weights(COUNTS, 1.0)[-1]


def test_a_class_without_examples_is_an_error():
    with pytest.raises(ValueError, match=r"\[2\]"):
        class_weights([10, 5, 0], 1.0)


def test_negative_power_is_an_error():
    with pytest.raises(ValueError, match="class_weight_power"):
        class_weights(COUNTS, -0.5)
