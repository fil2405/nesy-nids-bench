import numpy as np
import pandas as pd
import pytest

from nesybench.data.base import REPO_ROOT
from nesybench.logic.kb import (Condition, antecedents, condition_degree, load_kb,
                                target_index)

KB_YAML = """
version: test_v1
axioms:
  - id: A1
    group: ddos
    origin: mechanism
    target: DoS
    train: {support: 0.5, confidence: 0.9}
    all:
      - {col: flag, op: eq, value: S0}
      - {col: count, op: gt, tau: 30, s: 5}
  - id: A2
    group: dataset
    origin: data
    target: Probe
    all:
      - {col: rate, op: le, tau: 0.5, s: 0.1}
"""


@pytest.fixture
def kb(tmp_path):
    path = tmp_path / "kb.yaml"
    path.write_text(KB_YAML)
    return load_kb(path)


@pytest.fixture
def df():
    return pd.DataFrame({
        "flag": ["S0", "S0", "SF", "S0"],
        "count": [100.0, 30.0, 100.0, 1.0],
        "rate": [0.0, 0.5, 1.0, 0.9],
    })


def write(tmp_path, text):
    path = tmp_path / "kb.yaml"
    path.write_text(text)
    return path


def test_load(kb):
    assert kb.version == "test_v1"
    assert kb.ids == ["A1", "A2"]
    assert kb.axioms[0].train["confidence"] == 0.9
    assert kb.axioms[1].conditions[0] == Condition("rate", "le", tau=0.5, s=0.1)


def test_crisp_edges(df):
    gt = Condition("count", "gt", tau=30.0)
    le = Condition("rate", "le", tau=0.5)
    assert condition_degree(df, gt, crisp=True).tolist() == [1, 0, 1, 0]  # 30 is not > 30
    assert condition_degree(df, le, crisp=True).tolist() == [1, 1, 0, 0]  # 0.5 is <= 0.5


def test_graded_half_at_tau(df):
    d = condition_degree(df, Condition("count", "gt", tau=30.0, s=5.0))
    assert d[1] == pytest.approx(0.5)
    assert d[0] > 0.99 and d[3] < 0.01


def test_s_scale_widens_the_sigmoid_and_keeps_tau(df):
    cond = Condition("count", "gt", tau=30.0, s=5.0)
    base, wide = condition_degree(df, cond), condition_degree(df, cond, s_scale=4.0)
    assert wide[1] == pytest.approx(0.5)      # still half at tau
    assert wide[0] < base[0] and wide[3] > base[3]
    assert np.allclose(condition_degree(df, cond, s_scale=1.0), base)


def test_s_scale_is_ignored_when_crisp(df):
    cond = Condition("count", "gt", tau=30.0)
    assert (condition_degree(df, cond, crisp=True, s_scale=7.0)
            == condition_degree(df, cond, crisp=True)).all()


def test_s_scale_must_be_positive(kb, df):
    with pytest.raises(ValueError, match="s_scale"):
        antecedents(kb, df, s_scale=0.0)


def test_graded_tends_to_crisp(df):
    gt = Condition("count", "gt", tau=30.0, s=1e-3)
    assert np.allclose(condition_degree(df, gt)[[0, 2, 3]], [1, 1, 0])


def test_graded_needs_s(df):
    with pytest.raises(ValueError, match="s not set"):
        condition_degree(df, Condition("count", "gt", tau=30.0))


def test_extreme_values_stay_finite():
    d = pd.DataFrame({"count": [1e12, -1e12]})
    out = condition_degree(d, Condition("count", "gt", tau=0.0, s=1e-3))
    assert np.isfinite(out).all() and out.tolist() == [1.0, 0.0]


def test_antecedents_product(kb, df):
    a = antecedents(kb, df)
    assert a.shape == (4, 2)
    assert a[2, 0] == 0.0                     # flag is not S0
    assert a[1, 0] == pytest.approx(0.5)      # S0 and count at tau
    assert a[0, 1] > 0.99 and a[3, 1] < 0.02
    assert antecedents(kb, df, crisp=True)[:, 0].tolist() == [1, 0, 0, 0]


def test_missing_column(kb, df):
    with pytest.raises(KeyError, match="rate"):
        antecedents(kb, df.drop(columns="rate"))


def test_target_index(kb):
    assert target_index(kb, ["Normal", "DoS", "Probe"]).tolist() == [1, 2]
    with pytest.raises(ValueError, match="DoS"):
        target_index(kb, ["Normal", "Probe"])


@pytest.mark.parametrize("bad, match", [
    ("op: ge", "op"),
    ("op: gt", "tau"),
    ("op: eq", "value"),
    ("op: gt, tau: 1, s: 0", "s must be"),
])
def test_bad_condition(tmp_path, bad, match):
    text = ("version: v\naxioms:\n  - {id: X, group: ddos, origin: data, target: DoS,\n"
            f"     all: [{{col: c, {bad}}}]}}\n")
    with pytest.raises(ValueError, match=match):
        load_kb(write(tmp_path, text))


def test_bad_group_and_duplicates(tmp_path):
    one = "{id: X, group: %s, origin: data, target: DoS, all: [{col: c, op: eq, value: a}]}"
    with pytest.raises(ValueError, match="group"):
        load_kb(write(tmp_path, f"version: v\naxioms: [{one % 'nope'}]\n"))
    with pytest.raises(ValueError, match="duplicate"):
        load_kb(write(tmp_path, f"version: v\naxioms: [{one % 'ddos'}, {one % 'ddos'}]\n"))


@pytest.mark.skipif(not (REPO_ROOT / "kb" / "nslkdd_v1.yaml").exists(),
                    reason="kb/nslkdd_v1.yaml not in the repo yet")
def test_shipped_kb_parses():
    kb = load_kb(REPO_ROOT / "kb" / "nslkdd_v1.yaml")
    assert target_index(kb, ["Normal", "DoS", "Probe", "R2L", "U2R"]).min() >= 0
