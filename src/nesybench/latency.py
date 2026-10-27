"""Inference latency of a saved model (E12, first part).

    python -m nesybench.latency --model-dir results/models/<run_id>_seed0 \
        [--batch-sizes 1 64 1024] [--repeats 200] [--warmup 20] [--device cpu|default] \
        [--config CONFIG --seed 0 --part test [--postcheck [--kb FILE]]] [--out FILE.json]

Times the model call on already preprocessed rows, one call per batch. Without
--config the rows are standard normal noise of the right width (the cost of a
network does not depend on the values, the one of trees does a little).
--postcheck also times model + rules (annotate) and model + rules + override
(enforce) on the same batches; the rules read the raw columns of the same rows.
"""
import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

PERCENTILES = (50, 95, 99)


def measure(predict, x, batch_sizes=(1, 64, 1024), repeats=200, warmup=20) -> dict:
    """Per batch size: latency percentiles (ms) of predict(batch) and rows per second."""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup >= 0")
    out = {}
    for size in batch_sizes:
        if size < 1:
            raise ValueError(f"batch size must be >= 1, got {size}")
        batch = x[np.arange(size) % len(x)]
        for _ in range(warmup):
            predict(batch)
        seconds = np.empty(repeats)
        for i in range(repeats):
            start = perf_counter()
            predict(batch)
            seconds[i] = perf_counter() - start
        ms = seconds * 1000
        out[str(size)] = {
            **{f"p{q}_ms": float(np.percentile(ms, q)) for q in PERCENTILES},
            "mean_ms": float(ms.mean()),
            "rows_per_second": float(size / seconds.mean()),
        }
    return out


def measure_pipeline(predict, x, raw, kb, class_names, batch_sizes=(1, 64, 1024),
                     repeats=200, warmup=20) -> dict:
    """measure() of model, model + annotate and model + enforce on the same row batches."""
    from nesybench.logic.kb import target_index
    from nesybench.logic.postcheck import enforce, fired_rules

    if len(raw) != len(x):
        raise ValueError(f"raw has {len(raw)} rows, x has {len(x)}")
    cols = sorted({c.col for a in kb.axioms for c in a.conditions})
    raw = raw[cols].reset_index(drop=True)
    targets = target_index(kb, class_names)
    rows = np.arange(len(x))

    def model(idx):
        return predict(x[idx])

    def annotate(idx):
        return predict(x[idx]), fired_rules(kb, raw.iloc[idx])

    def enforced(idx):
        proba, fired = annotate(idx)
        return enforce(np.asarray(proba).argmax(axis=1), fired, targets)

    return {name: measure(step, rows, batch_sizes, repeats, warmup)
            for name, step in (("model", model), ("annotate", annotate), ("enforce", enforced))}


def load_predictor(directory, n_classes=None):
    """(predict function on a float array, kind) for the model saved in `directory`.

    n_classes: width of the output; a forest fitted without some class gets zero columns for it.
    """
    directory = Path(directory)
    if (directory / "model.keras").exists():
        import tensorflow as tf
        import tf_keras
        model = tf_keras.models.load_model(str(directory / "model.keras"), compile=False)

        def predict(batch):
            return model(tf.constant(batch, dtype=tf.float32), training=False).numpy()
        return predict, "keras"
    if (directory / "model.joblib").exists():
        import joblib
        model = joblib.load(directory / "model.joblib")

        def predict(batch):
            proba = model.predict_proba(batch)
            if n_classes is None or proba.shape[1] == n_classes:
                return proba
            out = np.zeros((len(batch), n_classes))
            out[:, model.classes_] = proba
            return out
        return predict, "random_forest"
    if (directory / "model.json").exists():
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(directory / "model.json"))
        best = int(booster.attr("best_round") or 0)   # rounds kept after early stopping
        rounds = (0, best) if best else (0, 0)
        return (lambda batch: booster.predict(xgb.DMatrix(batch), iteration_range=rounds)), "xgboost"
    raise FileNotFoundError(f"no model.keras, model.joblib or model.json in {directory}")


def _rows(config_path, seed, part):
    from nesybench.data import get_spec, load_dataset
    from nesybench.data.preprocess import preprocess
    from nesybench.data.splits import split_data
    from nesybench.run import load_config

    config = load_config(config_path)
    spec = get_spec(config["dataset"])
    train = load_dataset(config["dataset"], "train")
    test = load_dataset(config["dataset"], "test") if spec.fixed_test else None
    splits = preprocess(*split_data(config["split"], train, test, spec, seed,
                                    **config.get("split_params", {})), spec)
    x = np.asarray(splits.part(part)[0], dtype=np.float32)
    return x, splits.raw(part), config, list(splits.class_names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 1024])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "default"], default="default")
    parser.add_argument("--config", help="real rows: split and preprocessing of this config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--part", default="test", choices=["val", "test"])
    parser.add_argument("--postcheck", action="store_true", help="also model + rules (needs --config)")
    parser.add_argument("--kb", help="with --postcheck; default: c5_kb or kb of the config")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.postcheck and not args.config:
        parser.error("--postcheck needs --config (the rules read the raw rows)")

    directory = Path(args.model_dir)
    meta = json.loads((directory / "meta.json").read_text())
    if args.device == "cpu":
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
        except ImportError:
            pass
    predict, kind = load_predictor(directory, n_classes=len(meta["class_names"]))
    raw = config = None
    if args.config:
        x, raw, config, class_names = _rows(args.config, args.seed, args.part)
        source = f"{args.part} of {args.config}"
    else:
        rng = np.random.default_rng(0)
        x, source = rng.standard_normal((4096, len(meta["feature_names"]))).astype(np.float32), "synthetic"
    result = {
        "model_dir": directory.name,
        "kind": kind,
        "device": args.device,
        "input": source,
        "n_features": int(x.shape[1]),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "batches": measure(predict, x, args.batch_sizes, args.repeats, args.warmup),
        "pipeline": None,
        "size_bytes": {p.name: p.stat().st_size for p in sorted(directory.iterdir())
                       if p.is_file()},
    }
    if args.postcheck:
        from nesybench.logic.kb import load_kb
        kb_path = args.kb or config.get("c5_kb") or config.get("kb")
        if not kb_path:
            raise SystemExit("--postcheck: no KB, pass --kb or set c5_kb in the config")
        kb = load_kb(kb_path)
        result["kb"] = kb.version
        result["pipeline"] = measure_pipeline(predict, x, raw, kb, class_names, args.batch_sizes,
                                              args.repeats, args.warmup)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
