import numpy as np
import tensorflow as tf
import ltn
from ltn.core import Formula

def class_degree(P, y, c):
    p_c = P[y == c, c]
    return 1 - np.sqrt(np.mean((1-p_c)**2)) 

_forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(), semantics="forall")

def axiom_degrees(P, y, n_classes):
    degrees = []
    for c in range(n_classes):
        keep = y == c
        p_c = tf.boolean_mask(P[:, c], keep)
        x = ltn.Variable("x", tf.boolean_mask(P, keep))
        degrees.append(_forall(x, Formula(p_c, ["x"])))
    return degrees


def domain_degrees(P, antecedents, targets):
    """forall x (antecedent_j(x) -> class targets[j](x)), Reichenbach, over every row."""
    x = ltn.Variable("x", P)
    degrees = []
    for j, c in enumerate(targets):
        a = antecedents[:, j]
        implied = 1.0 - a + a * P[:, int(c)]
        degrees.append(_forall(x, Formula(implied, ["x"])))
    return degrees


def sat_agg(degrees):
    agg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())
    return agg(degrees)