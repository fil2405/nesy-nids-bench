"""CIC-IDS2017, improved version (Liu et al., CNS 2022): five daily CSV files.

No official test file: the whole cleaned dataset is the "train" frame and the
split strategy cuts it (splits.stratified_random).
"""
import json

import numpy as np
import pandas as pd

from .base import CLASS_COL, DATA_ROOT, LABEL_COL, DatasetSpec

DATA_DIR = DATA_ROOT / "cic-ids2017-improved"
INTERIM_DIR = DATA_ROOT.parent / "interim"
CACHE = INTERIM_DIR / "cicids2017_improved.parquet"
REPORT = INTERIM_DIR / "cicids2017_improved_report.json"
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")

ATTEMPTED_SUFFIX = " - Attempted"
PROTOCOLS = {0: "hopopt", 1: "icmp", 6: "tcp", 17: "udp"}
META = ("id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp",
        "Attempted Category")

SPEC = DatasetSpec(
    name="cicids2017",
    classes=("Normal", "DoS", "DDoS", "Probe", "R2L", "U2R", "Botnet"),
    categorical=("Protocol",),
    non_features=(LABEL_COL, CLASS_COL, "day", *META),
    fixed_test=False,
)

CLASS_TO_LABELS = {
    "Normal": ["BENIGN"],
    "DoS": ["DoS Hulk", "DoS GoldenEye", "DoS Slowloris", "DoS Slowhttptest", "Heartbleed"],
    "DDoS": ["DDoS"],
    "Probe": ["Portscan", "Infiltration - Portscan"],
    "R2L": ["FTP-Patator", "SSH-Patator", "Web Attack - Brute Force",
            "Web Attack - XSS", "Web Attack - SQL Injection"],
    "U2R": ["Infiltration"],
    "Botnet": ["Botnet"],
}
LABEL_TO_CLASS = {
    label: cls for cls, labels in CLASS_TO_LABELS.items() for label in labels
}


def map_classes(df: pd.DataFrame) -> pd.Series:
    """Class of each row. Attempted flows are Normal, as the authors recommend."""
    attempted = df[LABEL_COL].str.endswith(ATTEMPTED_SUFFIX).to_numpy()
    if not np.array_equal(attempted, (df["Attempted Category"] != -1).to_numpy()):
        raise ValueError("Attempted label and Attempted Category disagree")
    cls = df[LABEL_COL].map(LABEL_TO_CLASS).mask(attempted, "Normal")
    unknown = sorted(df.loc[cls.isna(), LABEL_COL].unique())
    if unknown:
        raise ValueError(f"Labels missing from CLASS_TO_LABELS: {unknown}")
    return cls


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in SPEC.non_features]


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def build(frames: dict) -> tuple[pd.DataFrame, dict]:
    """Daily frames -> one clean frame and a report of what was removed."""
    missing = [d for d in DAYS if d not in frames]
    if missing:
        raise KeyError(f"missing days: {missing}")
    parts, report = [], {"normal_by_day_raw": {}, "rows_by_day_raw": {}}
    for day in DAYS:
        df = frames[day].rename(columns={"Label": LABEL_COL}).copy()
        df["day"] = day
        df[CLASS_COL] = map_classes(df)
        report["rows_by_day_raw"][day] = len(df)
        report["normal_by_day_raw"][day] = int((df[CLASS_COL] == "Normal").sum())
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Protocol"] = df["Protocol"].map(lambda p: PROTOCOLS.get(p, str(p)))

    feats = _feature_columns(df)
    numeric = [c for c in feats if c not in SPEC.categorical]
    finite = np.isfinite(df[numeric].to_numpy(dtype=float)).all(axis=1)
    report["non_finite_dropped"] = _counts(df.loc[~finite, CLASS_COL])
    df = df[finite]

    dup = df.duplicated(subset=feats + [LABEL_COL])
    report["duplicates_dropped"] = _counts(df.loc[dup, CLASS_COL])
    df = df[~dup]
    # same feature vector under two labels: kept, the model cannot tell them apart
    report["rows_with_conflicting_labels"] = int(df.duplicated(subset=feats, keep=False).sum())

    df = df.reset_index(drop=True)
    df[numeric] = df[numeric].astype("float32")
    report["class_counts"] = _counts(df[CLASS_COL])
    report["label_counts"] = _counts(df[LABEL_COL])
    report["class_by_day"] = {d: _counts(g[CLASS_COL]) for d, g in df.groupby("day")}
    return df, report


def load_cicids2017(split: str = "train") -> pd.DataFrame:
    """Whole cleaned dataset, cached in data/interim on first use."""
    if split != "train":
        raise ValueError(f"{SPEC.name} has no official test file, got split {split!r}")
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    frames = {d: pd.read_csv(DATA_DIR / f"{d}.csv", low_memory=False) for d in DAYS}
    df, report = build(frames)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    REPORT.write_text(json.dumps(report, indent=2))
    return df
