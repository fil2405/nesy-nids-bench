import numpy as np


def class_weights(counts, power=1.0):
    """Per-class loss weights proportional to count**-power.

    Scaled so the mean weight over training examples is 1, which keeps the
    loss on the same scale for every power. power=0 gives no weighting and
    power=1 gives N / (K * count), the 'balanced' weights.
    """
    counts = np.asarray(counts, dtype=float)
    if power < 0:
        raise ValueError(f"class_weight_power must be >= 0, got {power}")
    empty = np.flatnonzero(counts == 0)
    if len(empty):
        raise ValueError(f"no training examples for classes {empty.tolist()}")
    raw = counts ** -power
    return raw * counts.sum() / (counts * raw).sum()
