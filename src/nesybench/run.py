"""Run a config, one result file per seed.

    python -m nesybench.run --config configs/example.yaml [--seeds 0 1]

The test part is scored only with `evaluate_test: true` (final runs).
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from nesybench.data import get_spec, load_dataset
from nesybench.data.base import REPO_ROOT
from nesybench.data.preprocess import preprocess
from nesybench.data.splits import split_data
from nesybench.logic.c5 import axiom_report
from nesybench.logic.kb import load_kb
from nesybench.metrics import compute_metrics, operator_metrics, pr_auc_per_class
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
    if kb is not None:  # crisp: a rule fires when every condition holds at its tau
        metrics["c5"] = axiom_report(kb, splits.raw(part), proba, splits.class_names,
                                     crisp=True, y_true=y)
    return metrics


def run_one(config: dict, seed: int, results_dir, overwrite: bool = False):
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

    path = save_run(
        str(results_dir), config["run_id"], config, seed, metrics, timings,
        overwrite=overwrite, training=training,
    )
    return path, metrics, timings


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0]) #type: ignore
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="seeds to run; default: `seeds` of the configuration")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--overwrite", action="store_true",
                        help="replace result files that already exist")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else config["seeds"]

    if not args.overwrite:
        existing = [
            str(p) for p in (run_path(args.results_dir, config["run_id"], s) for s in seeds)
            if p.exists()
        ]
        if existing:
            raise FileExistsError(f"already exist, pass --overwrite to replace: {existing}")

    for seed in seeds:
        path, metrics, timings = run_one(config, seed, args.results_dir, args.overwrite)
        print(f"{path.name}: val macro-F1 {metrics['val']['macro_f1']:.4f}, "
              f"fit {timings['fit_s']:.1f} s, total {timings['total_s']:.1f} s")


if __name__ == "__main__":
    main()
