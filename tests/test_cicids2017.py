"""CIC-IDS2017 loader and the random stratified split, on synthetic days."""
import numpy as np
import pandas as pd
import pytest

from nesybench.data import cicids2017 as cic
from nesybench.data import get_spec
from nesybench.data.base import class_index
from nesybench.data.preprocess import preprocess
from nesybench.data.splits import split_data

LABELS = ["BENIGN", "DoS Hulk", "DDoS", "Portscan", "FTP-Patator", "Infiltration",
          "Botnet", "Heartbleed", "Infiltration - Portscan", "DoS Hulk - Attempted"]


def _day(seed, n=60):
    rng = np.random.default_rng(seed)
    labels = np.resize(LABELS, n)
    return pd.DataFrame({
        "id": np.arange(n),
        "Flow ID": "f",
        "Src IP": "1.1.1.1",
        "Src Port": rng.integers(1, 65000, n),
        "Dst IP": "2.2.2.2",
        "Dst Port": rng.integers(1, 65000, n),
        "Protocol": np.resize([6, 17, 1], n),
        "Timestamp": pd.date_range("2017-07-03", periods=n, freq="s").astype(str),
        "Flow Duration": rng.random(n),
        "Flow Bytes/s": rng.random(n),
        "Label": labels,
        "Attempted Category": np.where(pd.Series(labels).str.endswith(" - Attempted"), 0, -1),
    })


@pytest.fixture
def frames():
    return {d: _day(i) for i, d in enumerate(cic.DAYS)}


def test_spec_keeps_normal_first_and_addresses_out_of_the_input():
    spec = get_spec("cicids2017")
    assert spec.classes[0] == "Normal"
    assert spec.categorical == ("Protocol",)
    assert not spec.fixed_test
    for col in ("Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp", "day"):
        assert col in spec.non_features


def test_labels_map_to_families():
    df = _day(0)
    df.loc[0, "Label"] = "Web Attack - XSS - Attempted"
    df.loc[0, "Attempted Category"] = 0
    df = df.rename(columns={"Label": "label"})
    cls = dict(zip(df["label"], cic.map_classes(df)))
    assert cls["BENIGN"] == "Normal"
    assert cls["DoS Hulk"] == "DoS"
    assert cls["Heartbleed"] == "DoS"
    assert cls["Infiltration"] == "U2R"
    assert cls["Infiltration - Portscan"] == "Probe"
    assert cls["Portscan"] == "Probe"
    assert cls["FTP-Patator"] == "R2L"
    assert cls["Web Attack - XSS - Attempted"] == "Normal"
    assert cls["DoS Hulk - Attempted"] == "Normal"


def test_unknown_label_is_an_error():
    df = _day(0).rename(columns={"Label": "label"})
    df.loc[0, "label"] = "New Attack"
    with pytest.raises(ValueError, match="New Attack"):
        cic.map_classes(df)


def test_attempted_label_and_category_must_agree():
    df = _day(0).rename(columns={"Label": "label"})
    df["Attempted Category"] = -1
    with pytest.raises(ValueError, match="disagree"):
        cic.map_classes(df)


def test_missing_day_is_an_error(frames):
    del frames["friday"]
    with pytest.raises(KeyError, match="friday"):
        cic.build(frames)


def test_build_adds_class_and_names_protocols(frames):
    df, _ = cic.build(frames)
    assert set(df["class"]) <= set(cic.SPEC.classes)
    assert set(df["Protocol"]) == {"tcp", "udp", "icmp"}
    assert df["Flow Duration"].dtype == np.float32
    assert df["Timestamp"].dtype.kind == "M"


def test_normal_per_day_counts_attempted_and_is_taken_before_cleaning(frames):
    frames["monday"] = pd.concat([frames["monday"], frames["monday"]], ignore_index=True)
    df, report = cic.build(frames)
    per_day = frames["tuesday"]["Label"].isin(["BENIGN", "DoS Hulk - Attempted"]).sum()
    assert report["normal_by_day_raw"]["tuesday"] == per_day
    assert report["normal_by_day_raw"]["monday"] == 2 * per_day
    assert report["rows_by_day_raw"]["monday"] == 120
    assert (df["day"] == "monday").sum() == 60


def test_non_finite_rows_are_dropped_and_reported(frames):
    frames["monday"].loc[3, "Flow Bytes/s"] = np.inf
    df, report = cic.build(frames)
    assert sum(report["non_finite_dropped"].values()) == 1
    assert len(df) == 5 * 60 - 1


def test_exact_duplicates_are_dropped_and_reported(frames):
    frames["tuesday"] = pd.concat([frames["tuesday"], frames["tuesday"].iloc[:5]],
                                  ignore_index=True)
    df, report = cic.build(frames)
    assert sum(report["duplicates_dropped"].values()) == 5
    assert len(df) == 5 * 60


def test_same_features_with_another_label_is_kept_and_counted(frames):
    row = frames["monday"].iloc[[1]].copy()
    row["Label"] = "DDoS"
    frames["monday"] = pd.concat([frames["monday"], row], ignore_index=True)
    df, report = cic.build(frames)
    assert report["rows_with_conflicting_labels"] == 2
    assert len(df) == 5 * 60 + 1


def test_report_counts_agree_with_the_frame(frames):
    df, report = cic.build(frames)
    assert sum(report["class_counts"].values()) == len(df)
    assert report["class_counts"]["U2R"] == int((df["class"] == "U2R").sum())


def _big(n_per_class=100):
    spec = get_spec("cicids2017")
    rng = np.random.default_rng(0)
    n = n_per_class * len(spec.classes)
    return spec, pd.DataFrame({"x": rng.random(n),
                               "class": np.repeat(spec.classes, n_per_class)})


def test_random_split_is_disjoint_and_covers_everything():
    spec, df = _big()
    train, val, test = split_data("stratified_random", df, None, spec, seed=0)
    assert len(train) + len(val) + len(test) == len(df)
    assert not (set(train.index) & set(val.index) | set(train.index) & set(test.index)
                | set(val.index) & set(test.index))
    assert len(test) == pytest.approx(0.2 * len(df), abs=len(spec.classes))
    assert len(val) == pytest.approx(0.2 * len(df), abs=len(spec.classes))


def test_random_split_is_stratified():
    spec, df = _big()
    for part in split_data("stratified_random", df, None, spec, seed=3):
        counts = part["class"].value_counts()
        assert set(counts.index) == set(spec.classes)
        assert counts.max() - counts.min() <= 1


def test_test_part_is_the_same_for_every_seed_but_validation_is_not():
    spec, df = _big()
    _, val0, test0 = split_data("stratified_random", df, None, spec, seed=0)
    _, val1, test1 = split_data("stratified_random", df, None, spec, seed=1)
    assert set(test0.index) == set(test1.index)
    assert set(val0.index) != set(val1.index)


def test_random_split_refuses_a_dataset_with_an_official_test():
    spec, df = _big()
    with pytest.raises(ValueError, match="official"):
        split_data("stratified_random", df, df, spec, seed=0)


def test_preprocess_one_hot_protocol_and_hides_addresses(frames):
    df, _ = cic.build(frames)
    spec = get_spec("cicids2017")
    train, val, test = split_data("stratified_random", df, None, spec, seed=0)
    s = preprocess(train, val, test, spec)
    assert {"Protocol_tcp", "Protocol_udp", "Protocol_icmp"} <= set(s.feature_names)
    assert not any("Port" in n or "IP" in n for n in s.feature_names)
    assert s.class_names == list(spec.classes)
    assert set(np.unique(s.y_train)) <= set(range(len(spec.classes)))
    assert class_index(train, spec).notna().all()
