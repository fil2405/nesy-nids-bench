"""The shipped tuning configs give every arm the same budget and a valid search."""
from pathlib import Path

import pytest

import nesybench
from nesybench import models, tune
from nesybench.data import get_spec
from nesybench.data.base import REPO_ROOT
from nesybench.run import load_config

CONFIGS = sorted((REPO_ROOT / "configs" / "tuning").glob("*.yaml"))
NEURAL = ("mlp_ce", "ltn_sup", "ltn_full")
COMMON = ("network.hidden_units", "network.dropout", "training.batch_size",
          "training.max_epochs")
PACKAGE = Path(nesybench.__file__).parent


def _load(path):
    tuning = tune.load_tuning(path)
    base_path = Path(tuning["base"])
    base = load_config(base_path if base_path.is_absolute() else REPO_ROOT / base_path)
    return tuning, base


def test_there_are_tuning_configs():
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_is_valid(path):
    tuning, base = _load(path)
    assert tuning["tuning_id"] == f"tune_{path.stem}"
    assert not set(tuning["seeds"]) & set(base["seeds"])
    for i, params in enumerate(tune.sample_trials(
            tuning["space"], tuning["trials"], tuning["search_seed"])):
        config = tune.trial_config(base, tuning, i, params)
        assert config["evaluate_test"] is False
        training = config["training"]
        if "min_per_class" in training:
            n_classes = len(get_spec(config["dataset"]).classes)
            assert training["batch_size"] >= n_classes * training["min_per_class"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_searched_key_is_read_by_the_model(path):
    tuning, base = _load(path)
    module = models._MODELS[base["model"]].partition(":")[0]
    source = (PACKAGE.parent / (module.replace(".", "/") + ".py")).read_text()
    keys = [*tuning["space"], *tuning.get("set", {})]
    unread = [k for k in keys if f'"{k.rpartition(".")[2]}"' not in source]
    assert not unread, f"{base['model']} never reads {unread}"


def test_same_budget_in_every_arm_of_a_dataset():
    budgets = {}
    for path in CONFIGS:
        tuning, base = _load(path)
        budgets.setdefault(base["dataset"], {})[path.stem] = (
            tuning["trials"], tuning["search_seed"], tuple(tuning["seeds"]))
    for dataset, arms in budgets.items():
        assert len(set(arms.values())) == 1, f"{dataset}: {arms}"


def test_neural_arms_share_a_fixed_network():
    fixed, lr = {}, {}
    for path in CONFIGS:
        tuning, base = _load(path)
        if base["model"] not in NEURAL:
            continue
        assert not set(COMMON) & set(tuning["space"]), path.stem
        fixed[path.stem] = {k: tuning.get("set", {}).get(k) for k in COMMON}
        lr[path.stem] = str(tuning["space"].get("training.learning_rate"))
    assert len(fixed) >= 2
    assert all(None not in s.values() for s in fixed.values())
    assert all(s == next(iter(fixed.values())) for s in fixed.values())
    assert "None" not in lr.values() and len(set(lr.values())) == 1
