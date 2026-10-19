import numpy as np
import pytest
import tensorflow as tf

from nesybench.logic.semantics import (axiom_degrees, class_degree,
                                       domain_degrees, sat_agg)


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


def test_p_changes_the_degree_and_matches_the_hand_formula():
    P, y = _batch()
    for p in (1, 4):
        degrees = axiom_degrees(tf.constant(P), y, 5, p)
        for c in range(5):
            assert float(degrees[c].tensor) == pytest.approx(class_degree(P, y, c, p), abs=1e-3)
    d2 = float(axiom_degrees(tf.constant(P), y, 5)[0].tensor)
    d4 = float(axiom_degrees(tf.constant(P), y, 5, 4)[0].tensor)
    assert d2 != pytest.approx(d4, abs=1e-3)


def test_p_below_one_is_refused():
    P, y = _batch()
    with pytest.raises(ValueError, match="p of the aggregator"):
        axiom_degrees(tf.constant(P), y, 5, 0.5)
    with pytest.raises(ValueError, match="p of the aggregator"):
        sat_agg([], 0)


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


def _rule_batch():
    P, _ = _batch()
    a = np.linspace(0.0, 1.0, len(P)).astype("float32")[:, None]
    return P, a, np.array([2])


def test_domain_degree_matches_the_hand_formula():
    P, a, targets = _rule_batch()
    degree = domain_degrees(tf.constant(P), tf.constant(a), targets)[0]
    err = a[:, 0] * (1 - P[:, 2])  # 1 - Reichenbach(a, p)
    assert float(degree.tensor) == pytest.approx(1 - np.sqrt(np.mean(err**2)), abs=1e-3)


def test_domain_degree_is_one_when_the_antecedent_never_holds():
    P, a, targets = _rule_batch()
    degree = domain_degrees(tf.constant(P), tf.constant(a * 0), targets)[0]
    assert float(degree.tensor) == pytest.approx(1.0, abs=1e-3)


def test_domain_degree_gradient_is_finite_and_pushes_the_consequent_up():
    P, a, targets = _rule_batch()
    P = tf.Variable(P)
    with tf.GradientTape() as tape:
        loss = 1 - sat_agg(domain_degrees(P, tf.constant(a), targets)).tensor
    grad = tape.gradient(loss, P).numpy()
    assert np.all(np.isfinite(grad))
    assert np.all(grad[:, 2] <= 0)  # more P(target) lowers the loss
    assert grad[0, 2] == pytest.approx(0.0, abs=1e-6)  # antecedent 0: no push

