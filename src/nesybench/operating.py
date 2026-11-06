"""Operating point fixed on validation, and cost curves, on the saved final models (no training).

    python -m nesybench.operating --config CONFIG --models results/models [--seeds 0 1 ...] [--out FILE]

Attack vs Normal, scored by -P(Normal). For each budget (FPR) the threshold is set on the
validation part and applied unchanged to the test part. Cost curve: lower envelope of the test
ROC, normalised expected cost against the probability cost PC(+) (Drummond and Holte 2006).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from nesybench.data.base import LABEL_COL
from nesybench.export import artifact_dir
from nesybench.saved_eval import model_proba_fn

BUDGETS = (1e-3, 1e-2)
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
BENIGN = 0


def threshold_at(score, attack, fpr):
    """Threshold t such that at most floor(fpr * n_benign) benign rows have score > t."""
    benign = np.sort(np.asarray(score)[~np.asarray(attack)])[::-1]
    k = int(np.floor(fpr * len(benign)))
    return float(benign[k]) if k < len(benign) else -np.inf


def rates(score, attack, t):
    alert = np.asarray(score) > t
    attack = np.asarray(attack)
    return float(alert[attack].mean()), float(alert[~attack].mean())


def cost_envelope(score, attack, grid=GRID):
    """Lower envelope of the cost lines of the ROC points, on `grid` of PC(+)."""
    fpr, tpr, _ = roc_curve(attack, score)
    pc = np.asarray(grid)[:, None]
    return ((1 - tpr)[None, :] * pc + fpr[None, :] * (1 - pc)).min(axis=1)


def seed_report(val_score, val_attack, test_score, test_attack, test_labels=None, unseen=(),
                pred_attack=None, budgets=BUDGETS):
    out = {"val_n_benign": int((~np.asarray(val_attack)).sum()),
           "test_n_benign": int((~np.asarray(test_attack)).sum()), "threshold": {}}
    for b in budgets:
        t = threshold_at(val_score, val_attack, b)
        val_tpr, val_fpr = rates(val_score, val_attack, t)
        tpr, fpr = rates(test_score, test_attack, t)
        row = {"t": t, "val_tpr": val_tpr, "val_fpr": val_fpr, "test_tpr": tpr, "test_fpr": fpr,
               "gap": fpr - b}
        if test_labels is not None:
            alert = np.asarray(test_score) > t
            labels = np.asarray(test_labels)
            row["by_label"] = {str(n): float(alert[labels == n].mean()) for n in sorted(set(labels))}
            rows = np.isin(labels, list(unseen))
            row["unseen_detected"] = float(alert[rows].mean()) if rows.any() else float("nan")
        out["threshold"][f"{b:g}"] = row
    if pred_attack is not None:
        pred_attack = np.asarray(pred_attack)
        out["argmax"] = {"test_tpr": float(pred_attack[test_attack].mean()),
                         "test_fpr": float(pred_attack[~test_attack].mean())}
    env = cost_envelope(test_score, test_attack)
    out["cost"] = {"envelope": [float(v) for v in env], "area": float(np.trapezoid(env, GRID))}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default="results/models")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--out", default="results/operating", help="folder, or a .json file")
    args = parser.parse_args(argv)

    from nesybench.data import get_spec, load_dataset
    from nesybench.data.preprocess import _split_xy
    from nesybench.data.splits import split_data
    from nesybench.run import load_config

    config = load_config(args.config)
    spec = get_spec(config["dataset"])
    run_id = config["run_id"]
    seeds = args.seeds if args.seeds else config["seeds"]
    train = load_dataset(config["dataset"], "train")
    test = load_dataset(config["dataset"], "test") if spec.fixed_test else None

    per_seed, missing = {}, []
    for seed in seeds:
        directory = artifact_dir(args.models, run_id, seed)
        if not (directory / "meta.json").exists():
            missing.append(seed)
            continue
        frames = split_data(config["split"], train, test, spec, seed, **config.get("split_params", {}))
        proba_of = model_proba_fn(directory, spec.classes)
        parts = {}
        for name, i in (("val", 1), ("test", 2)):
            raw = frames[i].reset_index(drop=True)
            feats, y = _split_xy(raw, spec)
            proba = proba_of(raw, feats)
            parts[name] = (raw, -proba[:, BENIGN].astype(np.float64), np.asarray(y) != BENIGN,
                           proba.argmax(axis=1) != BENIGN)
        raw_t, s_t, a_t, p_t = parts["test"]
        labels = raw_t[LABEL_COL] if LABEL_COL in raw_t.columns else None
        unseen = set(labels) - set(frames[0][LABEL_COL]) if labels is not None else set()
        per_seed[str(seed)] = rep = seed_report(parts["val"][1], parts["val"][2], s_t, a_t,
                                                labels, unseen, p_t)
        if unseen:
            rep["unseen_labels"] = sorted(map(str, unseen))
        print(f"seed {seed}: " + "; ".join(
            f"FPR {b} -> test TPR {r['test_tpr']:.4f} FPR {r['test_fpr']:.5f}"
            for b, r in rep["threshold"].items()) + f"; area costo {rep['cost']['area']:.4f}")

    if missing:
        print(f"modelli mancanti per i seed {missing}")
    if not per_seed:
        raise SystemExit("nessun modello trovato")
    out = Path(args.out)
    path = out if out.suffix == ".json" else out / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run_id, "budgets": list(BUDGETS), "grid": GRID.tolist(),
                                "per_seed": per_seed}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
