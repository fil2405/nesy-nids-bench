"""Run a config, one result file per seed.

    python -m nesybench.run --config configs/example.yaml [--seeds 0 1]

The test part is scored only with `evaluate_test: true` (final runs).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from nesybench.data import get_spec, load_dataset
from nesybench.data.base import REPO_ROOT
from nesybench.data.preprocess import preprocess
from nesybench.export import artifact_dir, file_sizes, save_preprocessing
from nesybench.data.splits import split_data
from nesybench.logic.c5 import axiom_report
from nesybench.logic.kb import load_kb
from nesybench.data.base import LABEL_COL
from nesybench.metrics import compute_metrics, operator_metrics, pr_auc_per_class, recall_by_label
from nesybench.models import get_model
from nesybench.timing import Stopwatch
from nesybench.utils.results import run_path, save_run
from nesybench.utils.seed import set_global_seed

REQUIRED_KEYS = ("run_id", "dataset", "split", "model", "seeds", "training")


def load_config(path) -> dict:
    config = yaml.safe_load(Path(path).read_text()) or {}
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    return config


def _load_c5_kb(config: dict):
    """KB for the C5 report: `c5_kb`, else `kb`. Not the same as using it to train."""
    path = config.get("c5_kb") or config.get("kb")
    if not path:
        return None
    path = Path(path)
    return load_kb(path if path.is_absolute() else REPO_ROOT / path)


def _evaluate(model, splits, part: str, timings: dict, kb=None):
    _, y = splits.part(part)
    with Stopwatch(timings, f"predict_{part}_s"):
        proba = np.asarray(model.predict_proba(splits, part))
    expected = (len(y), len(splits.class_names))
    if proba.shape != expected:
        raise ValueError(
            f"predict_proba({part!r}) returned shape {proba.shape}, expected {expected}"
        )
    
    metrics = compute_metrics(y, proba.argmax(axis=1), splits.class_names)
    metrics["pr_auc"] = pr_auc_per_class(y, proba, splits.class_names)
    metrics.update(operator_metrics(y, proba))
    raw = getattr(splits, f"raw_{part}", None)
    if raw is not None and LABEL_COL in raw.columns:
        metrics["by_label"] = recall_by_label(raw[LABEL_COL], y, proba.argmax(axis=1))
    if kb is not None:  # crisp: a rule fires when every condition holds at its tau
        metrics["c5"] = axiom_report(kb, splits.raw(part), proba, splits.class_names,
                                     crisp=True, y_true=y)
    return metrics


def _save_artifacts(model, splits, root, run_id: str, seed: int) -> dict:
    if not hasattr(model, "save"):
        raise ValueError(f"model {type(model).__name__} has no save()")
    directory = artifact_dir(root, run_id, seed)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory)
    save_preprocessing(splits, directory)
    return {"directory": directory.name, "files": file_sizes(directory)}


def run_one(config: dict, seed: int, results_dir, overwrite: bool = False, model_dir=None):
    model_class = get_model(config["model"])  # fail fast, before loading data
    kb = _load_c5_kb(config)
    timings: dict = {}
    with Stopwatch(timings, "total_s"):
        set_global_seed(seed)
        dataset = config["dataset"]
        spec = get_spec(dataset)
        with Stopwatch(timings, "load_s"):
            train_df = load_dataset(dataset, "train")
            test_df = load_dataset(dataset, "test") if spec.fixed_test else None
        with Stopwatch(timings, "split_s"):
            train_df, val_df, test_df = split_data(
                config["split"], train_df, test_df, spec, seed,
                **config.get("split_params", {}),
            )
        with Stopwatch(timings, "preprocess_s"):
            splits = preprocess(train_df, val_df, test_df, spec)

        model = model_class()
        with Stopwatch(timings, "fit_s"):
            training = model.fit(splits, {**config, "seed": seed})

        metrics = {"val": _evaluate(model, splits, "val", timings, kb)}
        if config.get("evaluate_test", False):
            metrics["test"] = _evaluate(model, splits, "test", timings, kb)

    artifacts = None
    if model_dir is not None:  # the result file is written last: it marks a finished run
        with Stopwatch(timings, "save_s"):
            artifacts = _save_artifacts(model, splits, model_dir, config["run_id"], seed)

    path = save_run(
        str(results_dir), config["run_id"], config, seed, metrics, timings,
        overwrite=overwrite, training=training, artifacts=artifacts,
    )
    return path, metrics, timings


def _finished(config: dict, results_dir, seed: int, model_dir=None) -> bool:
    """True if this seed has a result file made with this config; an error if it differs."""
    path = run_path(results_dir, config["run_id"], seed)
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record["config"] != json.loads(json.dumps(config)):
        raise ValueError(f"{path} was made with another config; pass --overwrite")
    if config.get("evaluate_test") and "test" not in record["metrics"]:
        raise ValueError(f"{path} has no test part; pass --overwrite")
    if model_dir is not None and not artifact_dir(model_dir, config["run_id"], seed).is_dir():
        raise ValueError(f"{path} has no saved model; pass --overwrite")
    return True


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0]) #type: ignore
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="seeds to run; default: `seeds` of the configuration")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--overwrite", action="store_true",
                        help="replace result files that already exist")
    parser.add_argument("--resume", action="store_true",
                        help="skip seeds that already have a result file")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="score the test part (final runs)")
    parser.add_argument("--save-model-dir",
                        help="write model and preprocessing of each run under this folder")
    args = parser.parse_args(argv)
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite exclude each other")

    config = load_config(args.config)
    if args.evaluate_test:
        config["evaluate_test"] = True
    seeds = args.seeds if args.seeds is not None else config["seeds"]

    if args.resume:
        seeds = [s for s in seeds
                 if not _finished(config, args.results_dir, s, args.save_model_dir)]
    elif not args.overwrite:
        existing = [
            str(p) for p in (run_path(args.results_dir, config["run_id"], s) for s in seeds)
            if p.exists()
        ]
        if existing:
            raise FileExistsError(f"already exist, pass --overwrite to replace: {existing}")

    for seed in seeds:
        path, metrics, timings = run_one(
            config, seed, args.results_dir, args.overwrite, args.save_model_dir)
        print(f"{path.name}: val macro-F1 {metrics['val']['macro_f1']:.4f}, "
              f"fit {timings['fit_s']:.1f} s, total {timings['total_s']:.1f} s")


if __name__ == "__main__":
    main()
