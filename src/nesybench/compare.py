"""Paired comparison of runs over seeds: Wilcoxon, Holm, bootstrap, Friedman and Nemenyi."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

# Nemenyi critical values q(0.05) for k = 2..10 (Demsar 2006, Table 5)
NEMENYI_Q05 = (1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164)


@dataclass
class Paired:
    n: int
    median: float
    mean: float
    low: float
    high: float
    wins: int
    ties: int
    p: float


def paired(a, b, n_boot: int = 10_000, seed: int = 0) -> Paired:
    """b minus a over the same seeds: median with bootstrap interval, exact-ish Wilcoxon."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"need two 1-d arrays of the same length, got {a.shape} and {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    d = (b - a)[keep]
    if len(d) < 2:
        raise ValueError("need at least two seeds with finite values")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(d, size=(n_boot, len(d)), replace=True), axis=1)
    p = 1.0 if np.all(d == 0) else float(wilcoxon(d, method="auto").pvalue)
    return Paired(len(d), float(np.median(d)), float(d.mean()),
                  float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)),
                  int((d > 0).sum()), int((d == 0).sum()), p)


def holm(pvalues) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


@dataclass
class Ranking:
    names: list[str]
    mean_ranks: list[float]     # 1 is best
    n: int
    friedman_p: float
    cd: float                   # Nemenyi critical difference at 0.05


def friedman_nemenyi(scores: dict) -> Ranking:
    """scores: name -> values over seeds (higher is better); seeds are the blocks."""
    names = list(scores)
    k = len(names)
    if not 3 <= k <= len(NEMENYI_Q05) + 1:
        raise ValueError(f"need 3 to {len(NEMENYI_Q05) + 1} arms, got {k}")
    matrix = np.column_stack([np.asarray(scores[n], dtype=float) for n in names])
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    n = len(matrix)
    if n < 2:
        raise ValueError("need at least two complete seeds")
    ranks = np.vstack([rankdata(-row) for row in matrix])
    p = float(friedmanchisquare(*matrix.T).pvalue) if np.ptp(matrix, axis=1).any() else 1.0
    cd = NEMENYI_Q05[k - 2] * np.sqrt(k * (k + 1) / (6 * n))
    return Ranking(names, ranks.mean(axis=0).tolist(), n, p, float(cd))
