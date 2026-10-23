"""Runner on a toy dataset with a stub model; no TensorFlow."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from nesybench import data, models, run
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
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n) + np.repeat([0.0, 3.0, 6.0], N_PER_CLASS),
            "x2": rng.normal(size=n),
            "proto": np.resize(["tcp", "udp"], n),
            "label": labels,
            "class": labels,
        }
    )


def _load_toy(split="train"):
    return _toy_frame(0 if split == "train" else 1)


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls = {"fits": 0, "parts": [], "seeds": [], "config": None}

    class Majority:


        def fit(self, splits, config):
            calls["fits"] += 1
            calls["config"] = config
            self.majority = np.bincount(splits.y_train).argmax()
            return {"n_parameters": 0, "epochs_run": 1}

        def predict_proba(self, splits, part):
            calls["parts"].append(part)
            X, _ = splits.part(part)
            proba = np.zeros((len(X), len(splits.class_names)))
            proba[:, self.majority] = 1.0
            return proba

    monkeypatch.setitem(data._DATASETS, "toy", (TOY, _load_toy))
    monkeypatch.setitem(models._MODELS, "majority", Majority)
    monkeypatch.setattr(run, "set_global_seed", lambda seed: calls["seeds"].append(seed))

    config = {
        "run_id": "toy_run",
        "dataset": "toy",
        "split": "official_test_with_stratified_val",
        "split_params": {"val_fraction": 0.25},
        "model": "majority",
        "seeds": [0, 1],
        "training": {},
    }
    return SimpleNamespace(
        calls=calls,
        config=config,
        config_path=tmp_path / "config.yaml",
        results=tmp_path / "results",
    )


def _main(env, *extra):
    env.config_path.write_text(yaml.safe_dump(env.config))
    run.main(["--config", str(env.config_path), "--results-dir", str(env.results), *extra])


def _record(env, seed):
    return json.loads((env.results / f"toy_run_seed{seed}.json").read_text())


def test_one_result_file_per_seed(env):
    _main(env)
    files = sorted(p.name for p in env.results.glob("*.json"))
    assert files == ["toy_run_seed0.json", "toy_run_seed1.json"]
    assert env.calls["seeds"] == [0, 1]


def test_seeds_option_overrides_the_configuration(env):
    _main(env, "--seeds", "1")
    assert [p.name for p in env.results.glob("*.json")] == ["toy_run_seed1.json"]


def test_record_has_metrics_timings_and_training_info(env):
    _main(env, "--seeds", "0")
    rec = _record(env, 0)
    assert rec["seed"] == 0
    assert rec["metrics"]["val"]["class_names"] == ["A", "B", "C"]
    for key in ("load_s", "split_s", "preprocess_s", "fit_s", "predict_val_s", "total_s"):
        assert rec["timings"][key] >= 0
    assert rec["training"] == {"n_parameters": 0, "epochs_run": 1}


def test_split_params_reach_the_split_strategy(env):
    _main(env, "--seeds", "0")
    n_val = sum(_record(env, 0)["metrics"]["val"]["support"])
    assert n_val == 0.25 * N_PER_CLASS * len(TOY.classes)


def test_model_gets_the_seed_but_the_saved_config_is_as_written(env):
    _main(env, "--seeds", "1")
    assert env.calls["config"]["seed"] == 1
    assert "seed" not in _record(env, 1)["config"]


def test_test_part_is_untouched_by_default(env):
    _main(env, "--seeds", "0")
    assert "test" not in env.calls["parts"]
    assert set(_record(env, 0)["metrics"]) == {"val"}


def test_test_part_is_evaluated_only_when_asked(env):
    env.config["evaluate_test"] = True
    _main(env, "--seeds", "0")
    rec = _record(env, 0)
    assert set(rec["metrics"]) == {"val", "test"}
    assert "predict_test_s" in rec["timings"]


def test_existing_result_stops_everything_before_training(env):
    _main(env, "--seeds", "0")
    fits = env.calls["fits"]
    with pytest.raises(FileExistsError, match="seed0"):
        _main(env, "--seeds", "0", "1")  # seed 1 is new but must not run either
    assert env.calls["fits"] == fits
    _main(env, "--seeds", "0", "--overwrite")
    assert env.calls["fits"] == fits + 1


def test_unknown_model_fails_before_loading_data(env, monkeypatch):
    def fail(split="train"):
        pytest.fail("data was loaded")

    monkeypatch.setitem(data._DATASETS, "toy", (TOY, fail))
    env.config["model"] = "nope"
    with pytest.raises(ValueError, match="nope"):
        _main(env)


def test_probabilities_of_the_wrong_shape_are_an_error(env, monkeypatch):
    class Broken:
        def fit(self, splits, config):
            return {}

        def predict_proba(self, splits, part):
            return np.zeros((3, 3))

    monkeypatch.setitem(models._MODELS, "majority", Broken)
    with pytest.raises(ValueError, match="shape"):
        _main(env, "--seeds", "0")


def test_missing_config_key_is_an_error(tmp_path):
    path = Path(tmp_path) / "c.yaml"
    path.write_text("run_id: x\n")
    with pytest.raises(ValueError, match="dataset"):
        run.load_config(path)


def test_record_has_ranking_and_operator_metrics(env):
    _main(env, "--seeds", "0")
    m = _record(env, 0)["metrics"]["val"]
    assert len(m["pr_auc"]) == 3
    assert set(m["fpr_at_tpr"]) == {"0.9", "0.95", "0.99"}
    assert set(m["tpr_at_fpr"]) == {"0.001", "0.01"}


TOY_KB = """
version: toy_v1
axioms:
  - id: T1
    group: dataset
    origin: data
    target: C
    all:
      - {col: x1, op: gt, tau: 4.5, s: 0.5}
"""


def test_c5_report_is_written_when_a_kb_is_given(env, tmp_path):
    kb_path = tmp_path / "kb.yaml"
    kb_path.write_text(TOY_KB)
    env.config["c5_kb"] = str(kb_path)
    _main(env, "--seeds", "0")
    c5 = _record(env, 0)["metrics"]["val"]["c5"]
    t1 = c5["axioms"]["T1"]
    assert c5["kb_version"] == "toy_v1"
    assert t1["n_covered"] > 0
    assert t1["violation_rate"] == 1.0  # the stub always answers one class, not C
    assert t1["rule_precision"] > 0.9  # x1 > 4.5 is almost only class C


def test_no_c5_report_without_a_kb(env):
    _main(env, "--seeds", "0")
    assert "c5" not in _record(env, 0)["metrics"]["val"]


def test_c5_falls_back_to_the_training_kb_key(env, tmp_path):
    kb_path = tmp_path / "kb.yaml"
    kb_path.write_text(TOY_KB)
    env.config["kb"] = str(kb_path)
    _main(env, "--seeds", "0")
    assert "c5" in _record(env, 0)["metrics"]["val"]


def test_evaluate_test_flag_scores_the_test_part(env):
    _main(env, "--seeds", "0", "--evaluate-test")
    rec = _record(env, 0)
    assert set(rec["metrics"]) == {"val", "test"}
    assert rec["config"]["evaluate_test"] is True


def test_resume_skips_the_seeds_already_done(env):
    _main(env, "--seeds", "0")
    fits = env.calls["fits"]
    _main(env, "--seeds", "0", "1", "--resume")
    assert env.calls["fits"] == fits + 1
    assert sorted(p.name for p in env.results.glob("*.json")) == [
        "toy_run_seed0.json", "toy_run_seed1.json"]


def test_resume_with_everything_done_runs_nothing(env):
    _main(env)
    fits = env.calls["fits"]
    _main(env, "--resume")
    assert env.calls["fits"] == fits


def test_resume_refuses_results_of_another_config(env):
    _main(env, "--seeds", "0")
    with pytest.raises(ValueError, match="another config"):
        _main(env, "--seeds", "0", "--resume", "--evaluate-test")


def test_resume_and_overwrite_exclude_each_other(env):
    with pytest.raises(SystemExit):
        _main(env, "--resume", "--overwrite")


def _saving_model(monkeypatch):
    base = models._MODELS["majority"]

    class Saving(base):
        def save(self, directory):
            (Path(directory) / "model.txt").write_text("x")

    monkeypatch.setitem(models._MODELS, "majority", Saving)


def test_model_and_preprocessing_are_saved_when_asked(env, tmp_path, monkeypatch):
    _saving_model(monkeypatch)
    root = tmp_path / "models"
    _main(env, "--seeds", "0", "--save-model-dir", str(root))
    folder = root / "toy_run_seed0"
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["class_names"] == ["A", "B", "C"]
    assert (folder / "preprocessing.joblib").is_file()
    artifacts = _record(env, 0)["artifacts"]
    assert artifacts["directory"] == "toy_run_seed0"
    assert set(artifacts["files"]) == {"model.txt", "preprocessing.joblib", "meta.json"}
    assert artifacts["files"]["model.txt"] == 1
    assert _record(env, 0)["timings"]["save_s"] >= 0


def test_nothing_is_saved_by_default(env):
    _main(env, "--seeds", "0")
    assert _record(env, 0)["artifacts"] is None


def test_a_model_without_save_is_an_error(env, tmp_path):
    with pytest.raises(ValueError, match="save"):
        _main(env, "--seeds", "0", "--save-model-dir", str(tmp_path / "models"))
    assert not list(env.results.glob("*.json"))


def test_resume_wants_the_saved_model_when_saving(env, tmp_path):
    _main(env, "--seeds", "0")
    with pytest.raises(ValueError, match="no saved model"):
        _main(env, "--seeds", "0", "--resume", "--save-model-dir", str(tmp_path / "m"))
