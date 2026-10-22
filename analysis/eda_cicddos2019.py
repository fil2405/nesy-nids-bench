"""EDA of the cleaned CIC-DDoS2019 Parquet files (train and test as files, no model).

    python analysis/eda_cicddos2019.py [--top 15] | tee eda_cicddos2019.txt
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nesybench.data import get_spec
from nesybench.data.base import CLASS_COL, LABEL_COL, REPO_ROOT

ATTACK, NORMAL = "DDoS", "Normal"


def show(title, table):
    print(f"\n== {title}")
    print(table.to_string() if hasattr(table, "to_string") else table)


def class_counts(parts):
    show("rows per class", pd.DataFrame({k: d[CLASS_COL].value_counts() for k, d in parts.items()}))
    show("rows per fine label", pd.DataFrame({k: d[LABEL_COL].value_counts() for k, d in parts.items()}).fillna(0).astype(int))


def inbound(parts):
    for name, df in parts.items():
        if "Inbound" not in df:
            print(f"\n== Inbound: not in {name}")
            continue
        tab = pd.crosstab(df["Inbound"], df[CLASS_COL])
        tab["DDoS share"] = tab[ATTACK] / tab.sum(axis=1)
        show(f"Inbound x class ({name})", tab)


def auc_one(df, col):
    y = (df[CLASS_COL] == ATTACK).to_numpy()
    if y.all() or not y.any() or df[col].nunique() < 2:
        return np.nan
    a = roc_auc_score(y, df[col].to_numpy(dtype=float))
    return max(a, 1 - a)  # direction does not matter


def single_feature_auc(parts, feats, top):
    train, test = parts["train"], parts["test"]
    rows = pd.DataFrame({"train_auc": {c: auc_one(train, c) for c in feats}})
    best = rows.sort_values("train_auc", ascending=False).head(top)
    best["test_auc"] = [auc_one(test, c) for c in best.index]
    best["distinct_train"] = [train[c].nunique() for c in best.index]
    show(f"top {top} input columns alone, AUC of {ATTACK} vs {NORMAL} (>= 0.5, direction ignored)", best.round(4))


def degenerate(parts, feats):
    train = parts["train"]
    nun = train[feats].nunique()
    show("constant input columns (train)", list(nun[nun <= 1].index) or "none")
    few = [c for c in nun[(nun > 1) & (nun <= 3)].index]
    for col in few:
        tab = pd.crosstab(train[col], train[CLASS_COL])
        show(f"input column with <= 3 values: {col} (train)", tab)


def protocol(parts):
    for name, df in parts.items():
        show(f"Protocol x class ({name})", pd.crosstab(df["Protocol"], df[CLASS_COL]))


def meta(parts):
    train, test = parts["train"], parts["test"]
    rows = {}
    for name, df in parts.items():
        for cls, g in df.groupby(CLASS_COL):
            rows[(name, cls)] = {
                "rows": len(g),
                "src_ip": g["Source IP"].nunique(),
                "dst_ip": g["Destination IP"].nunique(),
                "dst_port": g["Destination Port"].nunique(),
                "top_dst_port": g["Destination Port"].value_counts(normalize=True).iloc[0].round(3),
                "ts_min": g["Timestamp"].min(),
                "ts_max": g["Timestamp"].max(),
            }
    show("metadata per split and class (not model inputs)", pd.DataFrame(rows).T)
    shared = len(set(train["Destination IP"]) & set(test["Destination IP"]))
    show("destination IPs in both train and test", shared)
    if "Unnamed: 0" in train:
        show("Unnamed: 0 as a score (AUC, train / test)",
             f"{auc_one(train, 'Unnamed: 0'):.4f} / {auc_one(test, 'Unnamed: 0'):.4f}")
    if "SimillarHTTP" in train:
        show("SimillarHTTP values per class (train, top 5)",
             pd.crosstab(train["SimillarHTTP"], train[CLASS_COL]).sort_values(ATTACK, ascending=False).head(5))


def duplicates(parts, feats):
    show("rows sharing a feature vector with another row (any label)",
         {k: int(d.duplicated(subset=feats, keep=False).sum()) for k, d in parts.items()})


def report(parts, feats, top):
    class_counts(parts)
    inbound(parts)
    single_feature_auc(parts, feats, top)
    degenerate(parts, feats)
    protocol(parts)
    meta(parts)
    duplicates(parts, feats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)
    interim = REPO_ROOT / "data" / "interim"
    parts = {s: pd.read_parquet(interim / f"cicddos2019_{s}.parquet") for s in ("train", "test")}
    spec = get_spec("cicddos2019")
    feats = [c for c in parts["train"].columns
             if c not in spec.non_features and c not in spec.categorical]
    report(parts, feats, args.top)


if __name__ == "__main__":
    main()