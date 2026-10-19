"""Hyperparameter search with the same budget for every arm.

    python -m nesybench.tune --tuning configs/tuning/nslkdd_xgb.yaml [--dry-run]

Each trial is scored on the validation macro-F1, averaged over the tuning
seeds. The test part is never scored. Finished runs are reused, so an
interrupted search resumes where it stopped.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

from nesybench.data.base import REPO_ROOT
from nesybench.run import load_config, run_one
from nesybench.utils.results import run_path

REQUIRED_KEYS = ("tuning_id", "base", "trials", "search_seed", "seeds", "space")
KINDS = ("choice", "uniform", "loguniform", "int")


def load_tuning(path) -> dict:
    tuning = yaml.safe_load(Path(path).read_text()) or {}
    missing = [key for key in REQUIRED_KEYS if key not in tuning]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if tuning["trials"] < 1 or not tuning["seeds"] or not tuning["space"]:
        raise ValueError(f"{path}: needs trials >= 1, seeds and a space")
    for key, spec in tuning["space"].items():
        _check_spec(key, spec)
    return tuning


def _check_spec(key: str, spec) -> None:
    kinds = [k for k in KINDS if isinstance(spec, dict) and k in spec]
    if len(kinds) != 1 or len(spec) != 1:
        raise ValueError(f"{key}: one of {list(KINDS)} expected, got {spec}")
    kind, arg = kinds[0], spec[kinds[0]]
    if kind == "choice":
        if not isinstance(arg, list) or not arg:
            raise ValueError(f"{key}: choice needs a non-empty list")
        return
    if not isinstance(arg, list) or len(arg) != 2 or not arg[0] < arg[1]:
        raise ValueError(f"{key}: {kind} needs [low, high] with low < high")
    if kind == "loguniform" and arg[0] <= 0:
        raise ValueError(f"{key}: loguniform needs low > 0")


def _draw(spec: dict, u: float):
    kind = next(k for k in KINDS if k in spec)
    arg = spec[kind]
    if kind == "choice":
        return copy.deepcopy(arg[min(int(u * len(arg)), len(arg) - 1)])
    low, high = arg
    if kind == "int":
        return low + min(int(u * (high - low + 1)), high - low)
    if kind == "loguniform":
        value = float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
    else:
        value = low + u * (high - low)
    return float(f"{value:.4g}")


def sample_trials(space: dict, n: int, search_seed: int) -> list[dict]:
    """n parameter sets from a Latin hypercube: every dimension is covered evenly."""
    unit = qmc.LatinHypercube(d=len(space), seed=search_seed).random(n)
    return [{key: _draw(spec, unit[i, j]) for j, (key, spec) in enumerate(space.items())}
            for i in range(n)]


def set_dotted(config: dict, key: str, value) -> None:
    *path, last = key.split(".")
    node = config
    for part in path:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[last] = value


def _apply(base: dict, tuning: dict, params: dict) -> dict:
    config = copy.deepcopy(base)
    for key, value in {**tuning.get("set", {}), **params}.items():
        set_dotted(config, key, copy.deepcopy(value))
    return config


def trial_config(base: dict, tuning: dict, index: int, params: dict) -> dict:
    if base.get("evaluate_test"):
        raise ValueError("base config scores the test part, tuning never does")
    config = _apply(base, tuning, params)
    config["evaluate_test"] = False
    config["c5_kb"] = None  # not needed to choose hyperparameters
    config["run_id"] = f"{tuning['tuning_id']}_t{index:02d}"
    config["seeds"] = list(tuning["seeds"])
    return config


def tuned_config(base: dict, tuning: dict, params: dict) -> dict:
    """Base config with the chosen parameters, ready for the final runs."""
    config = _apply(base, tuning, params)
    config["run_id"] = tuning.get("final_run_id") or f"{base['run_id']}_tuned"
    return config


def _normal(config: dict) -> dict:
    return json.loads(json.dumps(config))


def _run_seed(config: dict, seed: int, results_dir: Path, overwrite: bool) -> dict:
    path = run_path(str(results_dir), config["run_id"], seed)
    if path.exists() and not overwrite:
        record = json.loads(path.read_text())
        if record["config"] != _normal(config):
            raise ValueError(f"{path} was made with another config; pass --overwrite")
        return record
    run_one(config, seed, results_dir, overwrite=overwrite)
    return json.loads(path.read_text())


def run_trial(config: dict, params: dict, index: int, results_dir: Path,
              overwrite: bool = False) -> dict:
    result = {"trial": index, "params": params}
    records = []
    try:
        for seed in config["seeds"]:
            records.append(_run_seed(config, seed, results_dir, overwrite))
    except FloatingPointError as error:  # a loss that blows up is a result, not a bug
        return {**result, "status": "failed", "error": str(error)}
    scores = [r["metrics"]["val"]["macro_f1"] for r in records]
    return {
        **result,
        "status": "ok",
        "seed_scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "fit_s_mean": float(np.mean([r["timings"]["fit_s"] for r in records])),
        "git_commits": sorted({r["git_commit"] for r in records}),
        "git_dirty": any(r["git_dirty"] for r in records),
    }


def pick_best(trials: list[dict]) -> dict | None:
    ok = [t for t in trials if t["status"] == "ok"]
    return max(ok, key=lambda t: (t["mean"], -t["trial"])) if ok else None


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _summary(tuning: dict, base_path, trials: list[dict]) -> dict:
    best = pick_best(trials)
    return {
        "tuning_id": tuning["tuning_id"],
        "base": str(base_path),
        "criterion": "mean validation macro-F1 over the tuning seeds",
        "budget": tuning["trials"],
        "search_seed": tuning["search_seed"],
        "seeds": tuning["seeds"],
        "set": tuning.get("set", {}),
        "space": tuning["space"],
        "done": len(trials),
        "trials": trials,
        "best": None if best is None else {k: best[k] for k in ("trial", "params", "mean", "std")},
    }


def tune(tuning: dict, results_dir, overwrite: bool = False, dry_run: bool = False,
         log=print) -> dict | None:
    base_path = Path(tuning["base"])
    base_path = base_path if base_path.is_absolute() else REPO_ROOT / base_path
    base = load_config(base_path)
    params_list = sample_trials(tuning["space"], tuning["trials"], tuning["search_seed"])
    configs = [trial_config(base, tuning, i, p) for i, p in enumerate(params_list)]

    if dry_run:
        for i, params in enumerate(params_list):
            log(f"trial {i:02d} {params}")
        return None

    results_dir = Path(results_dir)
    summary_path = results_dir / f"{tuning['tuning_id']}.json"
    trials = []
    for i, (config, params) in enumerate(zip(configs, params_list)):
        trial = run_trial(config, params, i, results_dir, overwrite)
        trials.append(trial)
        _write_atomic(summary_path, json.dumps(_summary(tuning, base_path, trials), indent=2))
        if trial["status"] == "ok":
            log(f"trial {i + 1}/{len(configs)}: val macro-F1 {trial['mean']:.4f} "
                f"+- {trial['std']:.4f}, fit {trial['fit_s_mean']:.1f} s")
        else:
            log(f"trial {i + 1}/{len(configs)}: failed ({trial['error']})")

    best = pick_best(trials)
    if best is None:
        raise RuntimeError(f"{tuning['tuning_id']}: every trial failed")
    best_config = tuned_config(base, tuning, best["params"])
    _write_atomic(results_dir / f"{tuning['tuning_id']}_best.yaml",
                  yaml.safe_dump(best_config, sort_keys=False))
    log(f"best: trial {best['trial']}, val macro-F1 {best['mean']:.4f}, {best['params']}")
    return _summary(tuning, base_path, trials)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # type: ignore
    parser.add_argument("--tuning", required=True, help="tuning YAML")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "tuning"))
    parser.add_argument("--overwrite", action="store_true",
                        help="rerun trials whose result files already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the trials and stop")
    args = parser.parse_args(argv)
    tune(load_tuning(args.tuning), args.results_dir, args.overwrite, args.dry_run)


if __name__ == "__main__":
    main()
