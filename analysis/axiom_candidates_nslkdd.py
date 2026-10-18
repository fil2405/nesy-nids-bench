"""Candidate axiom conditions on KDDTrain+ (the test file is never read)."""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nesybench.data import get_spec, load_dataset
from nesybench.data.base import CLASS_COL, LABEL_COL

ATTACKS = ("DoS", "Probe", "R2L", "U2R")
INDICATORS = ("protocol_type", "flag")  # service has 70 values, use --cond


def feature_table(df, spec):
    cols = [c for c in df.columns
            if c not in spec.non_features and c not in spec.categorical]
    X = df[cols].astype(float)
    for c in INDICATORS:
        for v in sorted(df[c].unique()):
            X[f"{c}={v}"] = (df[c] == v).astype(float)
    return X


def separation(df, X, col, target, top):
    """Each column alone as a score for target vs Normal."""
    keep = df[CLASS_COL].isin([target, "Normal"]) if col == CLASS_COL else \
        df[col].isin([target]) | (df[CLASS_COL] == "Normal")
    y = (df.loc[keep, col] == target).to_numpy()
    rows = []
    for c in X.columns:
        v = X.loc[keep, c].to_numpy()
        if np.ptp(v) == 0:
            continue
        rows.append((c, roc_auc_score(y, v), np.median(v[y]), np.median(v[~y]),
                     (v[y] != 0).mean(), (v[~y] != 0).mean()))
    out = pd.DataFrame(rows, columns=["feature", "auc", "median_target",
                                      "median_normal", "nonzero_target",
                                      "nonzero_normal"])
    order = (out["auc"] - 0.5).abs().sort_values(ascending=False).index
    return out.loc[order].head(top)


def condition_stats(df, expr, col, target):
    hit = df.eval(expr, engine="python").astype(bool)
    tab = (pd.DataFrame({"group": df[col], "hit": hit})
           .groupby("group").agg(rows=("hit", "size"), holds=("hit", "sum"))
           .reset_index())
    tab["support"] = tab["holds"] / tab["rows"]
    if col == LABEL_COL:  # 40 labels: keep the ones the condition touches
        tab = tab[(tab["holds"] > 0) | (tab["group"] == target)]
        tab = tab.sort_values("holds", ascending=False)
    n_hit = int(hit.sum())
    t = tab.set_index("group").loc[target]
    conf = t["holds"] / n_hit if n_hit else float("nan")
    return tab, n_hit, conf, t["rows"] / len(df)


def cond_expr(column, op, cut):
    # one-hot columns are written back as plain conditions, usable in --cond
    if "=" in column:
        name, val = column.split("=")
        return f"{name} == '{val}'" if op == ">" else f"{name} != '{val}'"
    return f"{column} {op} {cut:g}"


def rule_scan(df, X, col, target, min_support, top):
    """Single-feature conditions ranked by confidence, support above a floor.
    Cuts are values taken from the target rows, so support is never zero by
    construction."""
    is_t = (df[col] == target).to_numpy()
    normal = (df[CLASS_COL] == "Normal").to_numpy()
    n_t = is_t.sum()
    rows = {}
    for c in X.columns:
        v = X[c].to_numpy()
        cuts = np.unique(np.quantile(v[is_t], np.linspace(0, 1, 11),
                                     method="nearest"))
        for cut in cuts:
            for op, hit in ((">", v > cut), ("<=", v <= cut)):
                n_hit = int(hit.sum())
                if n_hit == 0:
                    continue
                h_t = int((hit & is_t).sum())
                if h_t / n_t < min_support:
                    continue
                rows[cond_expr(c, op, cut)] = (c, h_t / n_t, h_t / n_hit, n_hit,
                                               int((hit & normal).sum()))
    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=["feature", "support", "confidence",
                                          "hits", "normal_hits"])
    out = out.sort_values(["confidence", "support"], ascending=False)
    # one line per feature; other cuts of it are probed with --cond
    out = out.groupby("feature", sort=False).head(1)
    return out.head(top).reset_index(names="condition").drop(columns="feature")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["class", "label"], default="class",
                    help="group by family (class) or by fine attack label")
    ap.add_argument("--target", help="a family, or a label with --by label")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cond", help="pandas expression, e.g. \"serror_rate > 0.8\"")
    ap.add_argument("--rules", action="store_true",
                    help="scan single-feature conditions for --target")
    ap.add_argument("--min-support", type=float, default=0.05,
                    help="floor on the share of target rows a rule covers")
    args = ap.parse_args()

    df, spec = load_dataset("nslkdd", "train"), get_spec("nslkdd")
    col = CLASS_COL if args.by == "class" else LABEL_COL
    if args.target and args.target not in set(df[col]):
        raise SystemExit(f"{args.target!r} is not a {args.by}: "
                         f"{sorted(set(df[col]))}")
    print(f"KDDTrain+  n={len(df)}")
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 140

    if args.cond or args.rules:
        if not args.target:
            raise SystemExit("--cond and --rules need --target")
    if args.cond:
        tab, n_hit, conf, base = condition_stats(df, args.cond, col, args.target)
        print(f"condition: {args.cond}")
        print(tab.to_string(index=False))
        print(f"holds on {n_hit} rows; confidence for {args.target} "
              f"{conf:.3f} (base rate {base:.3f})")
        return

    X = feature_table(df, spec)
    if args.rules:
        print(f"rules for {args.target}, support >= {args.min_support}; "
              "confidence counts every class")
        print(rule_scan(df, X, col, args.target, args.min_support,
                        args.top).to_string(index=False))
        return

    print("auc > 0.5: higher values go with the target; < 0.5: lower values")
    targets = [args.target] if args.target else ATTACKS
    for target in targets:
        print(f"\n{target} vs Normal")
        print(separation(df, X, col, target, args.top).to_string(index=False))


if __name__ == "__main__":
    main()