"""CIC-DDoS2019 loader on small synthetic zips."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from nesybench.data import cicddos2019 as ddos
from nesybench.data import get_spec

FILES = {  # file name -> labels inside (the file name is not the label)
    "01-12/DrDoS_DNS.csv": ["DrDoS_DNS"] * 60 + ["BENIGN"] * 5,
    "01-12/UDPLag.csv": ["UDP-lag"] * 30 + ["WebDDoS"] * 3 + ["BENIGN"] * 4,
    "01-12/Syn.csv": ["Syn"] * 40 + ["BENIGN"] * 6,
}


def _csv(labels, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)
    df = pd.DataFrame({
        "Unnamed: 0": np.arange(n),
        " Flow ID": "f",
        " Source IP": "1.1.1.1",
        " Source Port": rng.integers(1, 65000, n),
        " Destination IP": "2.2.2.2",
        " Destination Port": rng.integers(1, 65000, n),
        " Protocol": np.resize([6, 17], n),
        " Timestamp": pd.date_range("2018-12-01 10:00", periods=n, freq="s").astype(str),
        " Flow Duration": rng.random(n) + seed,
        " Flow Bytes/s": rng.random(n),
        " Fwd Header Length": rng.integers(0, 100, n),
        " Fwd Header Length.1": 0,
        "SimillarHTTP": 0,
        " Label": labels,
    })
    df[" Fwd Header Length.1"] = df[" Fwd Header Length"]
    return df


def _zip(path, files=FILES, extra=None):
    with zipfile.ZipFile(path, "w") as zf:
        for i, (name, labels) in enumerate(files.items()):
            df = _csv(labels, i)
            if extra is not None and i == 0:
                df = extra(df)
            zf.writestr(name, df.to_csv(index=False))
        zf.writestr("01-12/.~lock.UDPLag.csv#", "junk")
    return path


@pytest.fixture
def archive(tmp_path):
    return _zip(tmp_path / "day.zip")


def test_spec():
    spec = get_spec("cicddos2019")
    assert spec.classes == ("Normal", "DDoS") and spec.fixed_test
    for col in ("Source IP", "Destination IP", "Timestamp", "Flow ID", "day"):
        assert col in spec.non_features


def test_class_comes_from_the_label_and_the_names_are_unified(archive):
    df, _ = ddos.read_day(archive, cap=1000, day="train")
    assert set(df["label"]) == {"BENIGN", "DrDoS_DNS", "UDPLag", "WebDDoS", "Syn"}
    assert (df.loc[df["label"] == "BENIGN", "class"] == "Normal").all()
    assert (df.loc[df["label"] != "BENIGN", "class"] == "DDoS").all()
    assert not any(c.startswith(" ") or c.endswith(".1") for c in df.columns)
    assert df["Protocol"].isin(["tcp", "udp"]).all()


def test_cap_keeps_all_benign_and_caps_each_attack_label(archive):
    df, report = ddos.read_day(archive, cap=20, day="train")
    counts = df["label"].value_counts()
    assert counts["BENIGN"] == 15
    assert counts["DrDoS_DNS"] == 20 and counts["Syn"] == 20 and counts["UDPLag"] == 20
    assert counts["WebDDoS"] == 3
    assert report["rows_raw"] == 148 and report["cap_per_attack_label"] == 20
    assert report["label_counts_raw"]["BENIGN"] == 15


def test_sample_is_reproducible_and_depends_on_the_seed(archive):
    a, _ = ddos.read_day(archive, cap=20, seed=1)
    b, _ = ddos.read_day(archive, cap=20, seed=1)
    c, _ = ddos.read_day(archive, cap=20, seed=2)
    assert a["Flow Duration"].equals(b["Flow Duration"])
    assert not a["Flow Duration"].equals(c["Flow Duration"])


def test_non_finite_rows_and_duplicates_are_dropped_and_counted(tmp_path):
    def spoil(df):
        df.loc[0, " Flow Bytes/s"] = np.inf
        df.loc[2, df.columns != " Label"] = df.loc[1, df.columns != " Label"]
        return df
    df, report = ddos.read_day(_zip(tmp_path / "d.zip", extra=spoil), cap=1000)
    assert report["non_finite_dropped"] == {"DrDoS_DNS": 1}
    assert report["duplicates_dropped"] == {"DrDoS_DNS": 1}
    assert len(df) == report["rows_sampled"] - 2


def test_unknown_label_is_an_error(tmp_path):
    files = {"01-12/x.csv": ["BENIGN", "Mystery"]}
    with pytest.raises(ValueError, match="Mystery"):
        ddos.read_day(_zip(tmp_path / "d.zip", files), cap=10)


def test_a_text_feature_column_is_an_error(tmp_path):
    def spoil(df):
        df["Odd"] = "x"
        return df
    with pytest.raises(ValueError, match="non-numeric"):
        ddos.read_day(_zip(tmp_path / "d.zip", extra=spoil), cap=10)


def test_load_builds_both_days_once(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    raw = tmp_path / "raw"
    raw.mkdir()
    _zip(raw / "CSV-01-12.zip")
    _zip(raw / "CSV-03-11.zip")
    monkeypatch.setattr(ddos, "DATA_DIR", raw)
    monkeypatch.setattr(ddos, "INTERIM_DIR", tmp_path / "interim")
    monkeypatch.setattr(ddos, "REPORT", tmp_path / "interim" / "report.json")
    monkeypatch.setattr(ddos, "MAX_PER_LABEL", {"train": 20, "test": 10})
    train = ddos.load_cicddos2019("train")
    test = ddos.load_cicddos2019("test")
    assert (train["day"] == "train").all() and (test["day"] == "test").all()
    assert test["label"].value_counts()["Syn"] == 10
    assert (tmp_path / "interim" / "report.json").exists()
    with pytest.raises(ValueError, match="splits"):
        ddos.load_cicddos2019("val")
