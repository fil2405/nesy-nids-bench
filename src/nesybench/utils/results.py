"""Writing one JSON result file per run."""
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("tensorflow", "ltn", "numpy", "sklearn", "xgboost"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


def save_run(
    results_dir: str,
    run_id: str,
    config: dict,
    seed: int,
    metrics: dict,
    timings: dict,
    kb_version: str | None = None,
) -> Path:
    """Save everything needed to trace a number in the thesis back to its run."""
    record = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "machine": platform.platform(),
        "seed": seed,
        "kb_version": kb_version,
        "config": config,
        "metrics": metrics,
        "timings": timings,
    }
    path = Path(results_dir) / f"{run_id}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path
