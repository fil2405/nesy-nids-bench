"""Shallow tree on KDDTrain+ to read candidate conjunctions (test file never read).

The tree is fit on one half of the train and each leaf is re-counted on the
other half, so a leaf that only fits noise shows up as a drop in confidence.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from axiom_candidates_nslkdd import cond_expr, feature_table
from nesybench.data import get_spec, load_dataset
from nesybench.data.base import CLASS_COL


def leaf_rules(tree, names):
    t = tree.tree_
    out = {}

    def walk(node, conds):
        if t.children_left[node] == -1:
            out[node] = conds
            return
        c, cut = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [cond_expr(c, "<=", cut)])
        walk(t.children_right[node], conds + [cond_expr(c, ">", cut)])

    walk(0, [])
    return out


def leaf_counts(leaf, node, is_t, normal):
    m = leaf == node
    n, h_t = int(m.sum()), int((m & is_t).sum())
    conf = h_t / n if n else float("nan")
    return n, h_t, int((m & normal).sum()), conf, h_t / max(is_t.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="a family: DoS, Probe, R2L, U2R")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--min-leaf", type=int, default=20)
    ap.add_argument("--min-hits", type=int, default=8,
                    help="target rows a leaf needs on the fit half")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, spec = load_dataset("nslkdd", "train"), get_spec("nslkdd")
    if args.target not in set(df[CLASS_COL]):
        raise SystemExit(f"{args.target!r} is not a class: {sorted(set(df[CLASS_COL]))}")
    X = feature_table(df, spec)
    is_t = (df[CLASS_COL] == args.target).to_numpy()
    normal = (df[CLASS_COL] == "Normal").to_numpy()
    a, b = train_test_split(np.arange(len(df)), test_size=0.5, random_state=args.seed,
                            stratify=df[CLASS_COL])
    # balanced only steers the splits; leaf counts below are plain
    tree = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=args.min_leaf,
                                  class_weight="balanced", random_state=args.seed)
    tree.fit(X.iloc[a].to_numpy(), is_t[a])
    leaf = tree.apply(X.to_numpy())
    rules = leaf_rules(tree, list(X.columns))

    rows = []
    for node, conds in rules.items():
        na, ha, nora, ca, sa = leaf_counts(leaf[a], node, is_t[a], normal[a])
        nb, hb, norb, cb, sb = leaf_counts(leaf[b], node, is_t[b], normal[b])
        if ha < args.min_hits:
            continue
        rows.append((ha, nora, ca, sa, hb, norb, cb, sb, " and ".join(conds)))
    out = pd.DataFrame(rows, columns=["hits_fit", "normal_fit", "conf_fit", "supp_fit",
                                      "hits_chk", "normal_chk", "conf_chk", "supp_chk",
                                      "rule"])
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 220
    pd.options.display.max_colwidth = None
    print(f"KDDTrain+  n={len(df)}  target={args.target}  depth={args.depth}  "
          f"fit {len(a)} rows, check {len(b)} rows")
    if out.empty:
        print("no leaf with enough target rows")
        return
    print(out.sort_values("conf_fit", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
