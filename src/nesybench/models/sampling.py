import math

import numpy as np


def class_quotas(counts, batch_size, power, min_per_class):
    w = np.asarray(counts, dtype=float) ** (1.0 - power)
    q = np.maximum(np.round(batch_size * w / w.sum()).astype(int), min_per_class)
    q[np.argmax(q)] += batch_size - q.sum()
    return q


def sample_batches(y, n_classes, batch_size, power=0.0, min_per_class=2, seed=0):
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == c) for c in range(n_classes)]
    counts = [len(i) for i in by_class]
    if min(counts) == 0:
        raise ValueError("class without training examples")
    if min_per_class < 1 or batch_size < n_classes * min_per_class:
        raise ValueError("batch too small for min_per_class")
    q = class_quotas(counts, batch_size, power, min_per_class)
    for _ in range(math.ceil(len(y) / batch_size)):
        parts = [rng.choice(idx, size=k, replace=k > len(idx))
                 for idx, k in zip(by_class, q)]
        batch = np.concatenate(parts)
        rng.shuffle(batch)
        yield batch