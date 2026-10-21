"""Tuning harness on a toy dataset with a stub model; no TensorFlow."""
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from nesybench import data, models, run, tune
from nesybench.data.base import DatasetSpec

TOY = DatasetSpec(
    name="toy",
    classes=("A", "B", "C"),
    categorical=("proto",),
    non_features=("label", "class"),
    fixed_test=True,
)
N_PER_CLASS = 40


def _toy_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_PER_CLASS * len(TOY.classes)
    labels = np.repeat(TOY.classes, N_PER_CLASS)
    return pd.DataFrame({
        "x1": rng.normal(size=n) + np.repeat([0.0, 3.0, 6.0], N_PER_CLASS),
        "x2": rng.normal(size=n),
        "proto": np.resize(["tcp", "udp"], n),
        "label": labels,
        "class": labels,
    })


def _load_toy(split="train"):
    return _toy_frame(0 if split == "train" else 1)


SPACE = {
    "params.quality": {"choice": [0.2, 0.5, 0.9]},
    "training.rate": {"loguniform": [1e-3, 1e-1]},
}


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls = {"fits": 0, "parts": []}

    class Stub:
        """Right on a share of the examples that grows with params.quality."""

        def fit(self, splits, config):
            calls["fits"] += 1
            if config["params"].get("boom"):
                raise FloatingPointError("non-finite loss at epoch 3")
            self.quality = config["params"]["quality"]
            return {"n_parameters": 0}

        def predict_proba(self, splits, part):
            calls["parts"].append(part)
            _, y = splits.part(part)
            guess = np.where(np.arange(len(y)) % 10 < self.quality * 10, y, (y + 1) % 3)
            return np.eye(3)[guess]

    monkeypatch.setitem(data._DATASETS, "toy", (TOY, _load_toy))
    monkeypatch.setitem(models._MODELS, "stub", Stub)
    monkeypatch.setattr(run, "set_global_seed", lambda seed: None)

    base = {
        "run_id": "toy_stub",
        "dataset": "toy",
        "split": "official_test_with_stratified_val",
        "split_params": {"val_fraction": 0.25},
        "model": "stub",
        "kb": None,
        "c5_kb": "kb/never_read.yaml",
        "evaluate_test": False,
        "seeds": list(range(10)),
        "params": {"quality": 0.5},
        "training": {"rate": 0.01},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    tuning = {
        "tuning_id": "tune_toy",
        "base": str(base_path),
        "trials": 6,
        "search_seed": 0,
        "seeds": [100, 101],
        "space": SPACE,
    }
    return SimpleNamespace(calls=calls, base=base, base_path=base_path, tuning=tuning,
                           results=tmp_path / "results")


def test_trials_are_reproducible_and_inside_the_space():
    a = tune.sample_trials(SPACE, 12, search_seed=3)
    assert a == tune.sample_trials(SPACE, 12, search_seed=3)
    assert a != tune.sample_trials(SPACE, 12, search_seed=4)
    assert len(a) == 12
    for params in a:
        assert params["params.quality"] in (0.2, 0.5, 0.9)
        assert 1e-3 <= params["training.rate"] <= 1e-1


def test_latin_hypercube_covers_every_choice_evenly():
    space = {"x": {"choice": ["a", "b", "c", "d", "e"]}, "y": {"uniform": [0.0, 1.0]}}
    trials = tune.sample_trials(space, 10, search_seed=0)
    assert Counter(t["x"] for t in trials) == {k: 2 for k in "abcde"}
    strata = sorted(int(t["y"] * 10) for t in trials)
    assert strata == list(range(10))


def test_int_and_choice_values_keep_their_type():
    space = {"d": {"int": [3, 5]}, "h": {"choice": [[64, 32], [128, 64]]}, "n": {"choice": [None, 10]}}
    for params in tune.sample_trials(space, 9, search_seed=1):
        assert isinstance(params["d"], int) and 3 <= params["d"] <= 5
        assert params["h"] in ([64, 32], [128, 64])
        assert params["n"] in (None, 10)
    assert {t["d"] for t in tune.sample_trials(space, 30, search_seed=1)} == {3, 4, 5}


@pytest.mark.parametrize("spec", [
    {"choice": []},
    {"uniform": [1.0, 1.0]},
    {"loguniform": [0.0, 1.0]},
    {"int": [5]},
    {"gauss": [0, 1]},
    {"uniform": [0, 1], "int": [0, 1]},
    [0, 1],
])
def test_bad_specs_are_refused(spec, tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({"tuning_id": "t", "base": "b", "trials": 2,
                                    "search_seed": 0, "seeds": [0], "space": {"k": spec}}))
    with pytest.raises(ValueError, match="k:"):
        tune.load_tuning(path)


def test_missing_keys_are_refused(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({"tuning_id": "t"}))
    with pytest.raises(ValueError, match="missing keys"):
        tune.load_tuning(path)


def test_set_dotted_creates_nested_keys():
    config = {"a": {"b": 1}}
    tune.set_dotted(config, "a.c", 2)
    tune.set_dotted(config, "x.y.z", 3)
    assert config == {"a": {"b": 1, "c": 2}, "x": {"y": {"z": 3}}}


def test_trial_config_never_scores_the_test_part(env):
    config = tune.trial_config(env.base, {**env.tuning, "set": {"training.mode": "x"}},
                               3, {"params.quality": 0.9})
    assert config["evaluate_test"] is False
    assert config["run_id"] == "tune_toy_t03" and config["seeds"] == [100, 101]
    assert config["params"]["quality"] == 0.9 and config["training"]["mode"] == "x"
    assert env.base["params"]["quality"] == 0.5 and "mode" not in env.base["training"]


def test_base_with_evaluate_test_is_refused(env):
    with pytest.raises(ValueError, match="test part"):
        tune.trial_config({**env.base, "evaluate_test": True}, env.tuning, 0, {})


def test_search_runs_every_trial_on_every_seed_and_picks_the_best(env):
    summary = tune.tune(env.tuning, env.results, log=lambda *_: None)
    assert env.calls["fits"] == 6 * 2
    assert len(list(env.results.glob("tune_toy_t*_seed*.json"))) == 12
    assert "test" not in env.calls["parts"]
    means = {t["trial"]: t["mean"] for t in summary["trials"]}
    best = summary["best"]
    assert means[best["trial"]] == max(means.values())
    assert best["params"]["params.quality"] == max(
        t["params"]["params.quality"] for t in summary["trials"])
    on_disk = json.loads((env.results / "tune_toy.json").read_text())
    assert on_disk["done"] == 6 and on_disk["budget"] == 6
    assert on_disk["criterion"].startswith("mean validation macro-F1")
    for trial in on_disk["trials"]:
        assert len(trial["seed_scores"]) == 2 and trial["std"] >= 0


def test_scores_are_the_validation_macro_f1_of_each_seed(env):
    summary = tune.tune(env.tuning, env.results, log=lambda *_: None)
    trial = summary["trials"][0]
    for seed, score in zip(env.tuning["seeds"], trial["seed_scores"]):
        record = json.loads((env.results / f"tune_toy_t00_seed{seed}.json").read_text())
        assert record["metrics"]["val"]["macro_f1"] == score
        assert set(record["metrics"]) == {"val"}


def test_best_config_is_ready_for_the_final_runs(env):
    summary = tune.tune(env.tuning, env.results, log=lambda *_: None)
    best = yaml.safe_load((env.results / "tune_toy_best.yaml").read_text())
    assert best["run_id"] == "toy_stub_tuned"
    assert best["seeds"] == list(range(10))
    assert best["evaluate_test"] is False
    assert best["c5_kb"] == "kb/never_read.yaml"
    assert best["params"]["quality"] == summary["best"]["params"]["params.quality"]
    assert best["training"]["rate"] == summary["best"]["params"]["training.rate"]


def test_final_run_id_can_be_named_in_the_tuning(env):
    env.tuning["final_run_id"] = "toy_final"
    tune.tune(env.tuning, env.results, log=lambda *_: None)
    best = yaml.safe_load((env.results / "tune_toy_best.yaml").read_text())
    assert best["run_id"] == "toy_final"


def test_a_second_call_reuses_finished_runs(env):
    first = tune.tune(env.tuning, env.results, log=lambda *_: None)
    fits = env.calls["fits"]
    second = tune.tune(env.tuning, env.results, log=lambda *_: None)
    assert env.calls["fits"] == fits
    assert second["best"] == first["best"]


def test_resume_after_a_partial_search(env):
    tune.tune(env.tuning, env.results, log=lambda *_: None)
    for path in env.results.glob("tune_toy_t0[3-5]_seed*.json"):
        path.unlink()  # as if the search had stopped after three trials
    fits = env.calls["fits"]
    tune.tune(env.tuning, env.results, log=lambda *_: None)
    assert env.calls["fits"] == fits + 3 * 2


def test_a_different_budget_is_a_different_search(env):
    tune.tune({**env.tuning, "trials": 3}, env.results, log=lambda *_: None)
    with pytest.raises(ValueError, match="another config"):
        tune.tune(env.tuning, env.results, log=lambda *_: None)


def test_results_of_another_config_are_not_reused_silently(env):
    tune.tune(env.tuning, env.results, log=lambda *_: None)
    changed = {**env.tuning, "set": {"params.extra": 1}}
    with pytest.raises(ValueError, match="another config"):
        tune.tune(changed, env.results, log=lambda *_: None)


def test_overwrite_reruns(env):
    tune.tune(env.tuning, env.results, log=lambda *_: None)
    fits = env.calls["fits"]
    tune.tune(env.tuning, env.results, overwrite=True, log=lambda *_: None)
    assert env.calls["fits"] == 2 * fits


def test_a_blown_up_loss_fails_the_trial_not_the_search(env):
    env.tuning["space"] = {**SPACE, "params.boom": {"choice": [False, True]}}
    summary = tune.tune(env.tuning, env.results, log=lambda *_: None)
    failed = [t for t in summary["trials"] if t["status"] == "failed"]
    assert failed and len(failed) < 6
    assert "non-finite" in failed[0]["error"]
    assert summary["best"]["trial"] not in {t["trial"] for t in failed}


def test_other_errors_stop_the_search(env, monkeypatch):
    class Broken:
        def fit(self, splits, config):
            raise KeyError("missing key")

    monkeypatch.setitem(models._MODELS, "stub", Broken)
    with pytest.raises(KeyError):
        tune.tune(env.tuning, env.results, log=lambda *_: None)
    assert len(list(env.results.glob("*.json"))) == 0


def test_every_trial_failing_is_an_error(env):
    env.tuning["space"] = {"params.boom": {"choice": [True]}, **SPACE}
    with pytest.raises(RuntimeError, match="every trial failed"):
        tune.tune(env.tuning, env.results, log=lambda *_: None)


def test_dry_run_prints_and_runs_nothing(env):
    lines = []
    assert tune.tune(env.tuning, env.results, dry_run=True, log=lines.append) is None
    assert len(lines) == 6 and env.calls["fits"] == 0
    assert not env.results.exists()


def test_cli(env, tmp_path):
    path = tmp_path / "tuning.yaml"
    path.write_text(yaml.safe_dump(env.tuning))
    tune.main(["--tuning", str(path), "--results-dir", str(env.results)])
    assert (env.results / "tune_toy_best.yaml").exists()


def _run_workers(env, n):
    for w in range(n):
        assert tune.tune(env.tuning, env.results, log=lambda *_: None, worker=w, workers=n) is None


def test_workers_split_the_trials_and_finalize_matches_the_sequential_search(env, tmp_path):
    _run_workers(env, 3)
    assert env.calls["fits"] == 6 * 2
    assert not (env.results / "tune_toy.json").exists()
    assert not (env.results / "tune_toy_best.yaml").exists()
    parallel = tune.finalize(env.tuning, env.results, log=lambda *_: None)
    assert (env.results / "tune_toy_best.yaml").exists()

    other = tmp_path / "sequential"
    sequential = tune.tune(env.tuning, other, log=lambda *_: None)
    assert parallel["best"] == sequential["best"]
    assert [t["mean"] for t in parallel["trials"]] == [t["mean"] for t in sequential["trials"]]


def test_each_worker_runs_only_its_trials(env):
    tune.tune(env.tuning, env.results, log=lambda *_: None, worker=1, workers=3)
    done = sorted(int(p.name.split("_t")[-1][:2]) for p in env.results.glob("tune_toy_t*_seed100.json"))
    assert done == [1, 4]


def test_finalize_names_the_unfinished_trials(env):
    tune.tune(env.tuning, env.results, log=lambda *_: None, worker=0, workers=2)
    with pytest.raises(RuntimeError, match=r"not finished: \[1, 3, 5\]"):
        tune.finalize(env.tuning, env.results, log=lambda *_: None)


def test_finalize_refuses_trials_of_another_search(env):
    _run_workers(env, 2)
    changed = {**env.tuning, "space": {**SPACE, "params.quality": {"choice": [0.1, 0.4, 0.7]}}}
    with pytest.raises(ValueError, match="another search"):
        tune.finalize(changed, env.results, log=lambda *_: None)


def test_a_failed_trial_of_a_worker_reaches_the_summary(env):
    env.tuning["space"] = {**SPACE, "params.boom": {"choice": [False, True]}}
    _run_workers(env, 2)
    summary = tune.finalize(env.tuning, env.results, log=lambda *_: None)
    assert any(t["status"] == "failed" for t in summary["trials"])
    assert summary["best"]["trial"] not in {t["trial"] for t in summary["trials"] if t["status"] == "failed"}


def test_worker_outside_the_range_is_refused(env):
    with pytest.raises(ValueError, match="outside"):
        tune.tune(env.tuning, env.results, worker=3, workers=3)


def test_dry_run_with_a_worker_lists_only_its_trials(env):
    lines = []
    tune.tune(env.tuning, env.results, dry_run=True, log=lines.append, worker=0, workers=3)
    assert len(lines) == 2


def test_cli_workers_then_finalize(env, tmp_path):
    path = tmp_path / "tuning.yaml"
    path.write_text(yaml.safe_dump(env.tuning))
    for w in range(2):
        tune.main(["--tuning", str(path), "--results-dir", str(env.results),
                   "--worker", str(w), "--workers", "2"])
    assert not (env.results / "tune_toy_best.yaml").exists()
    tune.main(["--tuning", str(path), "--results-dir", str(env.results), "--finalize"])
    assert (env.results / "tune_toy_best.yaml").exists()


def test_cli_worker_without_workers_is_refused(env, tmp_path):
    path = tmp_path / "tuning.yaml"
    path.write_text(yaml.safe_dump(env.tuning))
    with pytest.raises(SystemExit):
        tune.main(["--tuning", str(path), "--worker", "0"])
