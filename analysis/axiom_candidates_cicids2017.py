"""Candidate axiom conditions on CIC-IDS2017, on train + validation only (the test cut is dropped).

    python analysis/axiom_candidates_cicids2017.py --columns
    python analysis/axiom_candidates_cicids2017.py --target DDoS --cond "<expr>"
    python analysis/axiom_candidates_cicids2017.py --target Probe --rules     # single-feature scan
    python analysis/axiom_candidates_cicids2017.py --target DoS --tree        # depth-3 tree, fit/check halves
    ... --exclude "<expr>"   # work only on the rows the expression does NOT cover

Columns with spaces go in backticks in --cond. Addresses, ports and timestamps are not model
inputs and are not in the table. The split is the one of the configs (test_seed 0, 20% test).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from nesybench.data import get_spec, load_dataset
from nesybench.data.base import CLASS_COL, LABEL_COL
from nesybench.data.splits import split_data

DATASET = "cicids2017"
NORMAL = "Normal"


def working_frame(spec):
    """Train + validation of seed 0; the test cut is never kept."""
    full = load_dataset(DATASET, "train")
    train, val, _test = split_data("stratified_random", full, None, spec, seed=0,
                                   val_fraction=0.2, test_fraction=0.2, test_seed=0)
    del _test
    return pd.concat([train, val]).reset_index(drop=True)


def feature_table(df, spec):
    cols = [c for c in df.columns if c not in spec.non_features and c not in spec.categorical]
    X = df[cols].astype(float)
    for v in sorted(df["Protocol"].unique()):
        X[f"Protocol={v}"] = (df["Protocol"] == v).astype(float)
    return X


def cond_expr(column, op, cut):
    if column.startswith("Protocol="):
        value = column.split("=")[1]
        return f"Protocol {'==' if op == '>' else '!='} '{value}'"
    return f"`{column}` {op} {cut:g}"


def stats(df, expr, target):
    hit = df.eval(expr, engine="python").astype(bool).to_numpy()
    is_t = (df[CLASS_COL] == target).to_numpy()
    n_hit = int(hit.sum())
    normal = (df[CLASS_COL] == NORMAL).to_numpy()
    return hit, {"hits": n_hit, "support": (hit & is_t).sum() / max(is_t.sum(), 1),
                 "confidence": (hit & is_t).sum() / n_hit if n_hit else float("nan"),
                 "normal_hits": int((hit & normal).sum()),
                 "other_attack_hits": int((hit & ~is_t & ~normal).sum())}


def by_label(df, hit):
    tab = pd.DataFrame({"class": df[CLASS_COL], "label": df[LABEL_COL], "hit": hit}).groupby(
        ["class", "label"]).agg(rows=("hit", "size"), holds=("hit", "sum"))
    tab["share"] = tab["holds"] / tab["rows"]
    return tab[tab["holds"] > 0].sort_values("holds", ascending=False)


def rule_scan(df, X, target, min_support, top):
    is_t = (df[CLASS_COL] == target).to_numpy()
    normal = (df[CLASS_COL] == NORMAL).to_numpy()
    rows = {}
    for c in X.columns:
        v = X[c].to_numpy()
        for cut in np.unique(np.quantile(v[is_t], np.linspace(0, 1, 11), method="nearest")):
            for op, hit in ((">", v > cut), ("<=", v <= cut)):
                n_hit = int(hit.sum())
                h_t = int((hit & is_t).sum())
                if n_hit and h_t / is_t.sum() >= min_support:
                    rows[cond_expr(c, op, cut)] = (c, h_t / is_t.sum(), h_t / n_hit, n_hit,
                                                   int((hit & normal).sum()))
    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=["feature", "support", "confidence", "hits", "normal_hits"])
    if out.empty:
        print("no condition reaches the minimum support")
        return
    out = out.sort_values(["confidence", "support"], ascending=False).groupby("feature", sort=False).head(1)
    print(out.head(top).reset_index(names="condition").drop(columns="feature").to_string(index=False))


def leaf_rules(tree, names):
    t, out = tree.tree_, {}

    def walk(node, conds):
        if t.children_left[node] == -1:
            out[node] = conds
            return
        c, cut = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [cond_expr(c, "<=", cut)])
        walk(t.children_right[node], conds + [cond_expr(c, ">", cut)])

    walk(0, [])
    return out


def tree_rules(df, X, target, depth, min_leaf, min_hits, seed):
    is_t = (df[CLASS_COL] == target).to_numpy()
    a, b = train_test_split(np.arange(len(df)), test_size=0.5, random_state=seed,
                            stratify=df[CLASS_COL])
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf,
                                  class_weight="balanced", random_state=seed)
    tree.fit(X.iloc[a].to_numpy(), is_t[a])  # balanced only steers the splits; counts below are plain
    leaf = tree.apply(X.to_numpy())
    rows = []
    for node, conds in leaf_rules(tree, list(X.columns)).items():
        res = []
        for part in (a, b):
            m = leaf[part] == node
            n, h = int(m.sum()), int((m & is_t[part]).sum())
            res += [h, n - h, h / n if n else float("nan")]
        if res[0] >= min_hits:
            rows.append((*res, " and ".join(conds)))
    out = pd.DataFrame(rows, columns=["hits_fit", "other_fit", "conf_fit", "hits_chk", "other_chk",
                                      "conf_chk", "rule"])
    pd.options.display.max_colwidth = None
    print(out.sort_values("conf_fit", ascending=False).to_string(index=False)
          if len(out) else "no leaf with enough target rows")


def main():
    spec = get_spec(DATASET)
    attacks = [c for c in spec.classes if c != NORMAL]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default="DDoS", choices=attacks)
    ap.add_argument("--columns", action="store_true", help="list the input columns and exit")
    ap.add_argument("--cond", help="pandas expression, columns with spaces in backticks")
    ap.add_argument("--exclude", help="drop the rows this expression covers before anything else")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-support", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--min-hits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = working_frame(spec)
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 220
    print(f"CIC-IDS2017 train+validation (seed 0)  n={len(df)}  {df[CLASS_COL].value_counts().to_dict()}")
    if args.columns:
        print(", ".join(feature_table(df, spec).columns))
        return
    if args.exclude:
        df = df[~df.eval(args.exclude, engine="python").astype(bool).to_numpy()].reset_index(drop=True)
        print(f"after excluding {args.exclude}: n={len(df)}  {df[CLASS_COL].value_counts().to_dict()}")
    print(f"target {args.target}: {int((df[CLASS_COL] == args.target).sum())} rows")
    if args.cond:
        hit, s = stats(df, args.cond, args.target)
        print(f"{args.cond}\n{s}")
        print(by_label(df, hit).head(20).to_string())
    elif args.rules:
        rule_scan(df, feature_table(df, spec), args.target, args.min_support, args.top)
    elif args.tree:
        tree_rules(df, feature_table(df, spec), args.target, args.depth, args.min_leaf,
                   args.min_hits, args.seed)


if __name__ == "__main__":
    main()
