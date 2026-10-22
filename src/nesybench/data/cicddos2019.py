"""CIC-DDoS2019: two zips, one per day (training 01-12, test 03-11).

Read straight from the zips. Normal is BENIGN, DDoS is every attack. The days are
far too large to train on (70 M flows, 0.16% benign), so the sample is fixed by
label caps: all benign flows, at most MAX_PER_LABEL flows per attack label.
The class comes from the Label column, never from the file name.
"""
import json
import re
import zipfile

import numpy as np
import pandas as pd

from .base import CLASS_COL, DATA_ROOT, LABEL_COL, DatasetSpec
from .cicids2017 import PROTOCOLS

DATA_DIR = DATA_ROOT / "cic-ddos2019"
INTERIM_DIR = DATA_ROOT.parent / "interim"
REPORT = INTERIM_DIR / "cicddos2019_report.json"
ZIPS = {"train": "CSV-01-12.zip", "test": "CSV-03-11.zip"}
MAX_PER_LABEL = {"train": 90_000, "test": 30_000}
SAMPLE_SEED = 0
CHUNK = 500_000
ENCODING = "latin-1"

BENIGN = "BENIGN"
LABEL_FIX = {"UDP-lag": "UDPLag"}
ATTACKS = ("DrDoS_DNS", "DrDoS_LDAP", "DrDoS_MSSQL", "DrDoS_NTP", "DrDoS_NetBIOS",
           "DrDoS_SNMP", "DrDoS_SSDP", "DrDoS_UDP", "Syn", "TFTP", "UDPLag", "WebDDoS",
           "LDAP", "MSSQL", "NetBIOS", "Portmap", "UDP")
META = ("Unnamed: 0", "Flow ID", "Source IP", "Source Port", "Destination IP",
        "Destination Port", "Timestamp", "SimillarHTTP")
CAPTURE = ("Inbound",)  # capture direction, not traffic: kept in the data, never a model input

SPEC = DatasetSpec(
    name="cicddos2019",
    classes=("Normal", "DDoS"),
    categorical=("Protocol",),
    non_features=(LABEL_COL, CLASS_COL, "day", *META, *CAPTURE),
    fixed_test=True,
)


def cache_path(split: str):
    return INTERIM_DIR / f"cicddos2019_{split}.parquet"


def map_class(label: str) -> str:
    if label == BENIGN:
        return "Normal"
    if label in ATTACKS:
        return "DDoS"
    raise ValueError(f"Label missing from ATTACKS: {label!r}")


def _members(zf: zipfile.ZipFile) -> list[str]:
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    return sorted(n for n in names if not n.rsplit("/", 1)[-1].startswith(".~"))


def _label_column(zf, member) -> str:
    with zf.open(member) as f:
        columns = pd.read_csv(f, nrows=0, encoding=ENCODING).columns
    return next(c for c in columns if c.strip().lower() == "label")


def _clean_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace(LABEL_FIX)


def _label_codes(zf, members, vocab) -> list[np.ndarray]:
    """Label of every row of every file as integer codes (first pass: Label column only)."""
    codes = []
    for member in members:
        column, parts = _label_column(zf, member), []
        with zf.open(member) as f:
            for chunk in pd.read_csv(f, usecols=[column], chunksize=CHUNK, encoding=ENCODING):
                labels = _clean_labels(chunk[column])
                for label in labels.unique():
                    vocab.setdefault(label, len(vocab))
                parts.append(labels.map(vocab).to_numpy(np.int16))
        codes.append(np.concatenate(parts) if parts else np.zeros(0, np.int16))
    return codes


def _choose(codes: list[np.ndarray], vocab: dict, cap: int, seed: int) -> list[np.ndarray]:
    """Row masks per file: every benign row, `cap` random rows of each attack label."""
    rng = np.random.default_rng(seed)
    everything = np.concatenate(codes)
    keep = np.zeros(len(everything), dtype=bool)
    for label, code in vocab.items():
        rows = np.flatnonzero(everything == code)
        if label != BENIGN and len(rows) > cap:
            rows = rng.choice(rows, cap, replace=False)
        keep[rows] = True
    sizes = np.cumsum([len(c) for c in codes])[:-1]
    return np.split(keep, sizes)


def _drop_repeated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CICFlowMeter writes Fwd Header Length twice; pandas names the copy `.1`."""
    for column in [c for c in df.columns if re.search(r"\.\d+$", c)]:
        base = column.rsplit(".", 1)[0]
        if base not in df.columns or not df[column].equals(df[base]):
            raise ValueError(f"column {column!r} is not a copy of {base!r}")
        df = df.drop(columns=column)
    return df


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def read_day(zip_path, cap: int, seed: int = SAMPLE_SEED, day: str = "") -> tuple[pd.DataFrame, dict]:
    """One day as a clean sampled frame and a report of what was counted and removed."""
    with zipfile.ZipFile(zip_path) as zf:
        members = _members(zf)
        vocab: dict = {}
        codes = _label_codes(zf, members, vocab)
        masks = _choose(codes, vocab, cap, seed)
        parts = []
        for member, mask in zip(members, masks):
            offset = 0
            with zf.open(member) as f:
                for chunk in pd.read_csv(f, chunksize=CHUNK, encoding=ENCODING, low_memory=False):
                    chunk.columns = [c.strip() for c in chunk.columns]
                    chunk = chunk.rename(columns={"Label": LABEL_COL})
                    chunk[LABEL_COL] = _clean_labels(chunk[LABEL_COL])
                    parts.append(chunk[mask[offset:offset + len(chunk)]])
                    offset += len(chunk)
    inverse = {code: label for label, code in vocab.items()}
    totals = np.bincount(np.concatenate(codes), minlength=len(vocab))
    report = {
        "rows_raw": int(totals.sum()),
        "label_counts_raw": {inverse[c]: int(n) for c, n in enumerate(totals)},
        "cap_per_attack_label": cap,
        "sample_seed": seed,
    }
    df = _drop_repeated_columns(pd.concat(parts, ignore_index=True))
    df[CLASS_COL] = df[LABEL_COL].map(map_class)
    df["day"] = day
    report["rows_sampled"] = len(df)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Protocol"] = df["Protocol"].map(lambda p: PROTOCOLS.get(p, str(p)))
    feats = [c for c in df.columns if c not in SPEC.non_features or c in CAPTURE]
    numeric = [c for c in feats if c not in SPEC.categorical]
    text = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    if text:
        raise ValueError(f"non-numeric feature columns, add them to META: {text}")
    finite = np.isfinite(df[numeric].to_numpy(dtype=float)).all(axis=1)
    report["non_finite_dropped"] = _counts(df.loc[~finite, LABEL_COL])
    df = df[finite]
    dup = df.duplicated(subset=feats + [LABEL_COL])
    report["duplicates_dropped"] = _counts(df.loc[dup, LABEL_COL])
    df = df[~dup].reset_index(drop=True)
    report["rows_with_conflicting_labels"] = int(df.duplicated(subset=feats, keep=False).sum())
    df[numeric] = df[numeric].astype("float32")
    # chunks parsed as numbers or text give mixed columns, which Parquet refuses
    for column in [c for c in SPEC.non_features if c in df and df[c].dtype == object]:
        df[column] = df[column].astype(str)
    report["class_counts"] = _counts(df[CLASS_COL])
    report["label_counts"] = _counts(df[LABEL_COL])
    return df, report


def load_cicddos2019(split: str = "train") -> pd.DataFrame:
    """Sampled day (train or test), cached in data/interim on first use."""
    if split not in ZIPS:
        raise ValueError(f"{SPEC.name} has splits {sorted(ZIPS)}, got {split!r}")
    if cache_path(split).exists():
        return pd.read_parquet(cache_path(split))
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    reports, frames = {}, {}
    for name, archive in ZIPS.items():
        frames[name], reports[name] = read_day(DATA_DIR / archive, MAX_PER_LABEL[name], day=name)
    for name, df in frames.items():
        df.to_parquet(cache_path(name))
    REPORT.write_text(json.dumps(reports, indent=2))
    return frames[split]
