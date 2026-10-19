from functools import lru_cache

import numpy as np
import tensorflow as tf
import ltn
from ltn.core import Formula

def class_degree(P, y, c, p=2):
    p_c = P[y == c, c]
    return 1 - np.mean((1 - p_c) ** p) ** (1 / p)


def check_p(p):
    if p < 1:
        raise ValueError(f"p of the aggregator must be >= 1, got {p}")
    return p


@lru_cache(maxsize=None)
def _forall(p):
    aggregator = ltn.fuzzy_ops.Aggreg_pMeanError(p=check_p(p))
    return ltn.Wrapper_Quantifier(aggregator, semantics="forall")


def axiom_degrees(P, y, n_classes, p=2):
    forall = _forall(p)
    degrees = []
    for c in range(n_classes):
        keep = y == c
        p_c = tf.boolean_mask(P[:, c], keep)
        x = ltn.Variable("x", tf.boolean_mask(P, keep))
        degrees.append(forall(x, Formula(p_c, ["x"])))
    return degrees


def domain_degrees(P, antecedents, targets, p=2):
    """forall x (antecedent_j(x) -> class targets[j](x)), Reichenbach, over every row."""
    forall = _forall(p)
    x = ltn.Variable("x", P)
    degrees = []
    for j, c in enumerate(targets):
        a = antecedents[:, j]
        implied = 1.0 - a + a * P[:, int(c)]
        degrees.append(forall(x, Formula(implied, ["x"])))
    return degrees


def sat_agg(degrees, p=2):
    agg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=check_p(p)))
    return agg(degrees)