import pytest

from nesybench.data.nslkdd import DATA_DIR, FILES, load_nslkdd


def _require(split):
    if not (DATA_DIR / FILES[split]).exists():
        pytest.skip("dataset not found: run pytest from the repo root")


def test_train_shape_and_classes():
    _require("train")
    df = load_nslkdd("train")
    assert df.shape == (125973, 44)
    assert df["class"].value_counts().to_dict() == {
        "Normal": 67343, "DoS": 45927, "Probe": 11656, "R2L": 995, "U2R": 52,
    }


def test_test_split_has_no_unknown_labels():
    _require("test")
    df = load_nslkdd("test")
    assert df.shape == (22544, 44)
    assert set(df["class"]) == {"Normal", "DoS", "Probe", "R2L", "U2R"}