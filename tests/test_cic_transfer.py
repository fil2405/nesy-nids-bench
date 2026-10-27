import pandas as pd
import pytest

from nesybench.data import cic_transfer as ct
from nesybench.data import get_spec
from nesybench.data.base import CLASS_COL, LABEL_COL


def source():
    return pd.DataFrame({
        "id": [1, 2, 3, 4], "Src IP": ["a"] * 4, "Protocol": ["tcp", "tcp", "udp", "tcp"],
        "Total Fwd Packet": [1.0, 2.0, 3.0, 4.0], "Flow Bytes/s": [5.0, 6.0, 7.0, 8.0],
        "Only In Source": [0.0] * 4, "day": ["monday"] * 4, "Attempted Category": [-1] * 4,
        LABEL_COL: ["BENIGN", "DoS Hulk", "DDoS", "Portscan"],
        CLASS_COL: ["Normal", "DoS", "DDoS", "Probe"],
    })


def target():
    return pd.DataFrame({
        "Source IP": ["b"] * 2, "Inbound": [1, 0], "Protocol": ["udp", "tcp"],
        "Total Fwd Packets": [9.0, 1.0], "Flow Bytes/s": [3.0, 4.0], "Only In Target": [1.0, 1.0],
        "day": ["test"] * 2, LABEL_COL: ["Portmap", "BENIGN"], CLASS_COL: ["DDoS", "Normal"],
    })


def test_common_columns_match_names_across_versions():
    cols = ct.mapping(source(), target())
    assert cols == {"Protocol": "Protocol", "Total Fwd Packet": "Total Fwd Packets",
                    "Flow Bytes/s": "Flow Bytes/s"}


def test_train_keeps_normal_dos_ddos_as_two_classes():
    df = ct.build(source(), target(), "train")
    assert list(df.columns) == ["Protocol", "Total Fwd Packets", "Flow Bytes/s", LABEL_COL, CLASS_COL]
    assert df[CLASS_COL].tolist() == ["Normal", "Attack", "Attack"]
    assert df[LABEL_COL].tolist() == ["BENIGN", "DoS Hulk", "DDoS"]


def test_test_part_is_the_target_day_with_the_same_columns():
    df = ct.build(source(), target(), "test")
    assert list(df.columns) == ["Protocol", "Total Fwd Packets", "Flow Bytes/s", LABEL_COL, CLASS_COL]
    assert df[CLASS_COL].tolist() == ["Attack", "Normal"]


def test_capture_and_address_columns_never_enter():
    cols = ct.mapping(source(), target())
    assert not {"Inbound", "Source IP", "Src IP", "id", "day"} & (set(cols) | set(cols.values()))


def test_ambiguous_names_raise():
    with pytest.raises(ValueError):
        ct.common_columns(["A"], ["Total Fwd Packet", "Total Fwd Packets"])


def test_registered_with_a_fixed_test():
    spec = get_spec("cic_transfer")
    assert spec.classes == ("Normal", "Attack") and spec.fixed_test
