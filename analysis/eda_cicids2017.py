"""EDA of the cleaned CIC-IDS2017 Parquet (one frame, no model).

    python analysis/eda_cicids2017.py [--top 5] | tee eda_cicids2017.txt
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nesybench.data import get_spec, load_dataset
from nesybench.data.base import CLASS_COL, LABEL_COL

NORMAL = "Normal"
NORMAL_CAP = 200_000  # Normal rows kept per AUC, for speed


def show(title, table):
    print(f"\n== {title}")
    print(table.to_string() if hasattr(table, "to_string") else table)


def class_counts(df):
    show("rows per class", df[CLASS_COL].value_counts())
    show("rows per fine label", df.groupby([CLASS_COL, LABEL_COL]).size())
    show("class x day", pd.crosstab(df[CLASS_COL], df["day"]))
    if "Attempted Category" in df:
        show("Attempted Category x class (-1 = not attempted)",
             pd.crosstab(df["Attempted Category"], df[CLASS_COL]))


def auc_one(attack, normal, col):
    y = np.r_[np.ones(len(attack)), np.zeros(len(normal))]
    x = np.r_[attack[col].to_numpy(dtype=float), normal[col].to_numpy(dtype=float)]
    if np.ptp(x) == 0:
        return np.nan
    a = roc_auc_score(y, x)
    return max(a, 1 - a)  # direction does not matter


def single_feature_auc(df, feats, top):
    normal = df[df[CLASS_COL] == NORMAL]
    normal = normal.sample(min(NORMAL_CAP, len(normal)), random_state=0)
    for cls in [c for c in df[CLASS_COL].unique() if c != NORMAL]:
        attack = df[df[CLASS_COL] == cls]
        auc = pd.Series({c: auc_one(attack, normal, c) for c in feats}).sort_values(ascending=False)
        out = pd.DataFrame({"auc": auc.head(top).round(4),
                            "distinct": [df[c].nunique() for c in auc.head(top).index]})
        show(f"{cls} vs {NORMAL}: top {top} input columns alone ({len(attack)} rows)", out)


def degenerate(df, feats):
    nun = df[feats].nunique()
    show("constant input columns", list(nun[nun <= 1].index) or "none")
    show("input columns with 2-3 values", {c: int(n) for c, n in nun[(nun > 1) & (nun <= 3)].items()} or "none")


def protocol(df):
    show("Protocol x class", pd.crosstab(df["Protocol"], df[CLASS_COL]))


def meta(df):
    rows = {}
    for cls, g in df.groupby(CLASS_COL):
        rows[cls] = {
            "rows": len(g),
            "src_ip": g["Src IP"].nunique(),
            "dst_ip": g["Dst IP"].nunique(),
            "dst_port": g["Dst Port"].nunique(),
            "top_dst_port": g["Dst Port"].value_counts(normalize=True).iloc[0].round(3),
            "ts_min": g["Timestamp"].min(),
            "ts_max": g["Timestamp"].max(),
        }
    show("metadata per class (not model inputs)", pd.DataFrame(rows).T)
    show("timestamp range per day", df.groupby("day")["Timestamp"].agg(["min", "max"]))


def duplicates(df, feats):
    exact = int(df.duplicated(subset=feats + [LABEL_COL]).sum())
    shared = int(df.duplicated(subset=feats, keep=False).sum())
    conflict = int(df.groupby(feats)[CLASS_COL].transform("nunique").gt(1).sum())
    show("duplicates", {"exact (vector and label)": exact, "rows sharing a vector": shared,
                        "rows sharing a vector across classes": conflict})


def report(df, feats, numeric, top):
    class_counts(df)
    protocol(df)
    degenerate(df, numeric)
    meta(df)
    single_feature_auc(df, numeric, top)
    duplicates(df, feats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)
    df = load_dataset("cicids2017", "train")
    spec = get_spec("cicids2017")
    feats = [c for c in df.columns if c not in spec.non_features]  # Protocol included
    numeric = [c for c in feats if c not in spec.categorical]
    report(df, feats, numeric, args.top)


if __name__ == "__main__":
    main()