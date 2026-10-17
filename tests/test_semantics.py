import numpy as np
import pytest
import tensorflow as tf

from nesybench.logic.semantics import axiom_degrees, class_degree, sat_agg


def _batch():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(12, 5))
    P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 4])
    return P.astype("float32"), y


def test_ltn_degree_matches_the_hand_formula():
    P, y = _batch()
    degrees = axiom_degrees(tf.constant(P), y, 5)
    for c in range(5):
        assert float(degrees[c].tensor) == pytest.approx(class_degree(P, y, c), abs=1e-3)


def test_gradient_reaches_the_probabilities():
    P, y = _batch()
    P = tf.Variable(P)
    with tf.GradientTape() as tape:
        loss = 1 - sat_agg(axiom_degrees(P, y, 5)).tensor
    assert tape.gradient(loss, P) is not None


def test_empty_class_is_nan_and_poisons_sat_agg():
    P, y = _batch()
    y = np.where(y == 4, 0, y)  # class 4 has no examples
    degrees = axiom_degrees(tf.constant(P), y, 5)
    values = [float(d.tensor) for d in degrees]
    assert np.isnan(values[4])
    assert np.all(np.isfinite(values[:4]))
    assert np.isnan(float(sat_agg(degrees).tensor))  # why the sampler keeps every class in every batch