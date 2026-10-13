"""build_mlp, shared by MLP-CE, LTN-sup and LTN-full."""
import numpy as np
import pytest

import nesybench  # sets TF_USE_LEGACY_KERAS before TensorFlow is imported

pytest.importorskip("tensorflow")

from nesybench.models.mlp import build_mlp  # noqa: E402


def _build(**changes):
    args = dict(input_dim=7, n_classes=4, hidden_units=(8, 4), dropout=0.2,
                activation="relu", seed=0)
    args.update(changes)
    return build_mlp(**args)


def test_sizes_come_from_the_arguments():
    out = _build().predict(np.zeros((5, 7), dtype="float32"), verbose=0)
    assert out.shape == (5, 4)


def test_output_is_a_probability_distribution():
    X = np.random.default_rng(0).normal(size=(20, 7)).astype("float32")
    out = _build().predict(X, verbose=0)
    assert (out >= 0).all()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_parameter_count_follows_from_the_arguments():
    # (7*8 + 8) + (8*4 + 4) + (4*4 + 4) = 64 + 36 + 20
    assert _build().count_params() == 120


def test_dropout_follows_every_hidden_layer():
    rates = [layer.rate for layer in _build(dropout=0.3).layers
             if layer.__class__.__name__ == "Dropout"]
    assert rates == [0.3, 0.3]


def test_network_is_not_compiled():
    assert getattr(_build(), "optimizer", None) is None


def test_same_seed_gives_the_same_initial_weights():
    a, b = _build(seed=3), _build(seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a.get_weights(), b.get_weights()))


def test_different_seeds_give_different_initial_weights():
    assert not np.array_equal(_build(seed=3).get_weights()[0], _build(seed=4).get_weights()[0])
