"""Files of a fitted model and its preprocessing, written by the final runs."""
import json
from pathlib import Path

import joblib


def artifact_dir(root, run_id: str, seed: int) -> Path:
    return Path(root) / f"{run_id}_seed{seed}"


def save_preprocessing(splits, directory) -> None:
    directory = Path(directory)
    joblib.dump(splits.transformer, directory / "preprocessing.joblib")
    meta = {
        "feature_names": list(splits.feature_names),
        "class_names": list(splits.class_names),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))


def file_sizes(directory) -> dict:
    """Bytes per file, path relative to `directory`."""
    directory = Path(directory)
    return {
        str(p.relative_to(directory)): p.stat().st_size
        for p in sorted(directory.rglob("*")) if p.is_file()
    }
