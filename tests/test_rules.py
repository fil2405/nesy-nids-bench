import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nesybench import models
from nesybench.models.rules import RuleOnly

NAMES = ["Normal", "DoS", "Probe"]

KB_YAML = """
version: test_v1
axioms:
  - {id: A1, group: ddos, origin: mechanism, target: DoS,
     all: [{col: flag, op: eq, value: S0}]}
  - {id: A2, group: dataset, origin: data, target: Probe,
     all: [{col: rate, op: gt, tau: 0.5}]}
"""


@pytest.fixture
def cfg():
    path = Path(tempfile.mkdtemp()) / "kb.yaml"
    path.write_text(KB_YAML)
    return {"kb": str(path)}


def _splits(df):
    return SimpleNamespace(class_names=NAMES, raw=lambda part: df)


def test_registered():
    assert models.get_model("rule_only") is RuleOnly


def test_fired_rule_predicts_its_target_else_normal(cfg):
    df = pd.DataFrame({"flag": ["S0", "SF", "SF"], "rate": [0.0, 0.9, 0.1]})
    m = RuleOnly()
    m.fit(_splits(df), cfg)
    assert m.predict_proba(_splits(df), "val").argmax(axis=1).tolist() == [1, 2, 0]


def test_output_is_one_hot(cfg):
    df = pd.DataFrame({"flag": ["S0", "SF"], "rate": [0.0, 0.0]})
    m = RuleOnly()
    m.fit(_splits(df), cfg)
    p = m.predict_proba(_splits(df), "val")
    assert p.shape == (2, 3)
    assert (p.sum(axis=1) == 1).all()


def test_conflict_goes_to_the_first_axiom_and_is_counted(cfg):
    df = pd.DataFrame({"flag": ["S0", "S0"], "rate": [0.9, 0.0]})
    m = RuleOnly()
    info = m.fit(_splits(df), cfg)
    assert m.predict_proba(_splits(df), "val").argmax(axis=1).tolist() == [1, 1]
    assert info["train_conflicts"] == 1
    assert info["train_covered"] == 2


def test_needs_a_kb():
    with pytest.raises(ValueError, match="kb"):
        RuleOnly().fit(_splits(pd.DataFrame()), {})


def test_kb_edit_removes_an_axiom_and_is_recorded(cfg):
    df = pd.DataFrame({"flag": ["S0", "SF", "SF"], "rate": [0.0, 0.9, 0.1]})
    m = RuleOnly()
    info = m.fit(_splits(df), {**cfg, "kb_edit": {"exclude": ["A2"]}})
    assert info["kb_ids"] == ["A1"]
    assert info["kb_edit"] == {"exclude": ["A2"]}
    assert m.predict_proba(_splits(df), "val").argmax(axis=1).tolist() == [1, 0, 0]


def test_kb_edit_saves_the_edited_kb(cfg, tmp_path):
    df = pd.DataFrame({"flag": ["S0", "SF"], "rate": [0.0, 0.9]})
    m = RuleOnly()
    m.fit(_splits(df), {**cfg, "kb_edit": {"exclude": ["A1"]}})
    m.save(tmp_path)
    assert "A1" not in (tmp_path / "kb.yaml").read_text()
