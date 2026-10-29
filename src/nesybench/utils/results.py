"""One JSON result file per run."""
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
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
    status = _git("status", "--porcelain")
    return None if status is None else status != ""


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    try:  # some packages (ltn) have no __version__; the install metadata does
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _versions() -> dict:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("tensorflow", "ltn", "numpy", "sklearn", "xgboost"):
        out[name] = _package_version(name)
    return out


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def run_path(results_dir: str, run_id: str, seed: int) -> Path:
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
    artifacts: dict | None = None,
    split_sizes: dict | None = None,
) -> Path:
    """`metrics` is keyed by part ("val", "test"). Refuses to overwrite by default."""
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
        "artifacts": artifacts,
        "split_sizes": split_sizes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=_jsonable))
    return path