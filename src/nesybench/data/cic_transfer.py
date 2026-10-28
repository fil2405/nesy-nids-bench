"""E9 transfer: train and validation from CIC-IDS2017, test = the CIC-DDoS2019 test day.

Common schema: the flow columns of both after name normalisation (the two CICFlowMeter
versions spell them differently), named as in CIC-DDoS2019. Source rows: Normal and the
DoS and DDoS classes, both as Attack; the other attack classes are dropped.
"""
import re

import pandas as pd

from . import cicddos2019, cicids2017
from .base import CLASS_COL, LABEL_COL, DatasetSpec

SOURCE_CLASSES = {"Normal": "Normal", "DoS": "Attack", "DDoS": "Attack"}
TARGET_CLASSES = {"Normal": "Normal", "DDoS": "Attack"}
# source name -> target name where normalisation does not match them
RENAME = {
    "Total Bwd packets": "Total Backward Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count",  # misspelt in the CIC-DDoS2019 header
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

SPEC = DatasetSpec(
    name="cic_transfer",
    classes=("Normal", "Attack"),
    categorical=("Protocol",),
    non_features=(LABEL_COL, CLASS_COL),
    fixed_test=True,
)


def norm(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return key.replace("packets", "packet").replace("bytes", "byte")


def common_columns(source_cols, target_cols) -> dict:
    """source name -> target name for the columns both have; error if a key is ambiguous."""
    target = {}
    for col in target_cols:
        key = norm(col)
        if key in target:
            raise ValueError(f"target columns {target[key]!r} and {col!r} normalise alike")
        target[key] = col
    names = set(target.values())
    out = {}
    for col in source_cols:
        name = RENAME.get(col)
        if name not in names:
            name = target.get(norm(col))
        if name is not None:
            if name in out.values():
                raise ValueError(f"two source columns map to {name!r}")
            out[col] = name
    return out


def features(df, spec: DatasetSpec) -> list[str]:
    return [c for c in df.columns if c not in spec.non_features]


def mapping(source: pd.DataFrame, target: pd.DataFrame) -> dict:
    return common_columns(features(source, cicids2017.SPEC), features(target, cicddos2019.SPEC))


def build(source: pd.DataFrame, target: pd.DataFrame, split: str) -> pd.DataFrame:
    cols = mapping(source, target)
    if "Protocol" not in cols.values():
        raise ValueError("Protocol missing from the common schema")
    if split == "train":
        df = source[source[CLASS_COL].isin(SOURCE_CLASSES)].rename(columns=cols)
        classes = SOURCE_CLASSES
    elif split == "test":
        df = target
        classes = TARGET_CLASSES
    else:
        raise ValueError(f"{SPEC.name} has splits train and test, got {split!r}")
    out = df[[*cols.values(), LABEL_COL]].copy()
    out[CLASS_COL] = df[CLASS_COL].map(classes).to_numpy()
    return out.reset_index(drop=True)


def load_cic_transfer(split: str = "train") -> pd.DataFrame:
    return build(cicids2017.load_cicids2017("train"), cicddos2019.load_cicddos2019("test"), split)
