"""Writing one JSON result file per run."""
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _git_commit() -> str:
    return _git("rev-parse", "--short", "HEAD") or "unknown"


def _git_dirty() -> bool | None:
    """True if the code had uncommitted changes: the commit alone does not
    describe the code that produced the numbers."""
    status = _git("status", "--porcelain")
    return None if status is None else status != ""


def _versions() -> dict:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("tensorflow", "ltn", "numpy", "sklearn", "xgboost"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


def _jsonable(obj):
    """Let json.dumps write NumPy arrays and scalars (metrics.py returns them)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def run_path(results_dir: str, run_id: str, seed: int) -> Path:
    """Where the result of one run is written."""
    return Path(results_dir) / f"{run_id}_seed{seed}.json"


def save_run(
    results_dir: str,
    run_id: str,
    config: dict,
    seed: int,
    metrics: dict,
    timings: dict,
    kb_version: str | None = None,
    overwrite: bool = False,
    training: dict | None = None,
) -> Path:
    """Save everything needed to trace a number in the thesis back to its run.

    `metrics` maps a part ("val", "test") to its metrics, `timings` holds the
    wall-clock seconds measured by the runner, `training` is what the model
    reports about its own training (epochs, chosen epoch, parameters, ...).
    An existing file is not replaced unless overwrite=True, so that a result
    is never lost silently.
    """
    path = run_path(results_dir, run_id, seed)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

    record = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "versions": _versions(),
        "machine": platform.platform(),
        "seed": seed,
        "kb_version": kb_version,
        "config": config,
        "metrics": metrics,
        "timings": timings,
        "training": training,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=_jsonable))
    return path