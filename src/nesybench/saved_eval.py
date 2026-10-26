"""Score saved models without training again: macro-F1 before and after the symbolic post-check (E12, E17).

    python -m nesybench.saved_eval --config CONFIG --models results/models \
        [--kb FILE] [--modes annotate enforce] [--part test] [--seeds 0 1 ...] [--out results/saved_eval]

CONFIG is the config of the final runs (dataset, split, run_id, seeds). Per seed the part is rebuilt
with the same split, transformed with the saved preprocessing and fed to the saved model.
The macro-F1 stored by the run is printed next to the recomputed one.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from nesybench.export import artifact_dir
from nesybench.latency import load_predictor
from nesybench.logic.kb import load_kb
from nesybench.logic.postcheck import MODES, run_postcheck
from nesybench.metrics import compute_metrics

BATCH = 8192


def predict_all(predict, x, size=BATCH) -> np.ndarray:
    return np.vstack([np.asarray(predict(x[i:i + size])) for i in range(0, len(x), size)])


def score(proba, y, raw, kb, class_names, modes) -> dict:
    """Metrics of the model alone and after each post-check mode."""
    base = compute_metrics(y, proba.argmax(axis=1), class_names)
    out = {"macro_f1": base["macro_f1"], "accuracy": base["accuracy"], "postcheck": {}}
    for mode in modes:
        after, report = run_postcheck(kb, raw, proba, class_names, mode)
        scores = compute_metrics(y, after, class_names)
        out["postcheck"][mode] = {**report, "macro_f1": scores["macro_f1"], "accuracy": scores["accuracy"]}
    return out


def summarize(per_seed: dict, modes) -> dict:
    def stat(values):
        values = np.asarray(values, dtype=float)
        return {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0}

    out = {"seeds": len(per_seed), "macro_f1": stat([r["macro_f1"] for r in per_seed.values()])}
    for mode in modes:
        rows = [r["postcheck"][mode] for r in per_seed.values()]
        out[mode] = {
            "macro_f1": stat([r["macro_f1"] for r in rows]),
            "n_changed": stat([r["n_changed"] for r in rows]),
            **{k: stat([r["explanations"][k] for r in rows]) for k in ("coverage", "fidelity", "conciseness")},
        }
    return out


def _stored(results_dir, run_id, seed, part):
    path = Path(results_dir) / f"{run_id}_seed{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("metrics", {}).get(part, {}).get("macro_f1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default="results/models")
    parser.add_argument("--kb", help="default: c5_kb or kb of the config")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--part", default="test", choices=["val", "test"])
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/saved_eval")
    args = parser.parse_args(argv)

    from nesybench.data import get_spec, load_dataset
    from nesybench.data.preprocess import _split_xy
    from nesybench.data.splits import split_data
    from nesybench.run import load_config

    config = load_config(args.config)
    spec = get_spec(config["dataset"])
    kb_path = args.kb or config.get("c5_kb") or config.get("kb")
    if not kb_path:
        raise SystemExit("no KB: pass --kb or set c5_kb in the config")
    kb = load_kb(kb_path)
    seeds = args.seeds if args.seeds else config["seeds"]
    run_id = config["run_id"]

    train = load_dataset(config["dataset"], "train")
    test = load_dataset(config["dataset"], "test") if spec.fixed_test else None
    per_seed, missing = {}, []
    for seed in seeds:
        directory = artifact_dir(args.models, run_id, seed)
        if not (directory / "meta.json").exists():
            missing.append(seed)
            continue
        meta = json.loads((directory / "meta.json").read_text())
        if meta["class_names"] != list(spec.classes):
            raise SystemExit(f"{directory.name}: classes {meta['class_names']} differ from the dataset spec")
        frames = split_data(config["split"], train, test, spec, seed, **config.get("split_params", {}))
        raw = frames[2 if args.part == "test" else 1].reset_index(drop=True)
        features, y = _split_xy(raw, spec)
        x = np.asarray(joblib.load(directory / "preprocessing.joblib").transform(features), dtype=np.float32)
        predict, _ = load_predictor(directory, n_classes=len(spec.classes))
        proba = predict_all(predict, x)
        result = score(proba, y, raw, kb, list(spec.classes), args.modes)
        stored = _stored(args.results_dir, run_id, seed, args.part)
        result["stored_macro_f1"] = stored
        per_seed[seed] = result
        diff = "" if stored is None else f", salvato {stored:.4f}"
        print(f"seed {seed}: {args.part} macro-F1 {result['macro_f1']:.4f}{diff}; "
              + ", ".join(f"{m} {result['postcheck'][m]['macro_f1']:.4f}" for m in args.modes))

    if missing:
        print(f"modelli mancanti per i seed {missing}")
    if not per_seed:
        raise SystemExit("nessun modello trovato")
    report = {"run_id": run_id, "part": args.part, "kb": kb.version, "modes": args.modes,
              "per_seed": {str(k): v for k, v in per_seed.items()}, "summary": summarize(per_seed, args.modes)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}_{args.part}_{kb.version}.json"
    path.write_text(json.dumps(report, indent=2))
    print(path)


if __name__ == "__main__":
    main()
