"""Where the NSL-KDD test score is lost: attack names that the train file never shows.

    python analysis/test_gap_nslkdd.py                              # data only
    python analysis/test_gap_nslkdd.py --models-dir <dir>           # plus recall of the saved models

--models-dir holds the extracted <run_id>_seed<k> folders of the final runs. Scores here are a
diagnosis of the finished runs, nothing is tuned or selected on the test file.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nesybench.data import get_spec, load_dataset
from nesybench.data.base import CLASS_COL, LABEL_COL

ARMS = ("mlp_noweights", "mlp_weights", "ltn_sup", "ltn_full", "rf", "xgb")


def seen_flags(train, test):
    return test[LABEL_COL].isin(set(train[LABEL_COL])).to_numpy()


def data_tables(train, test, spec):
    seen = seen_flags(train, test)
    rows = []
    for cls in spec.classes:
        m = (test[CLASS_COL] == cls).to_numpy()
        rows.append((cls, int((train[CLASS_COL] == cls).sum()), int(m.sum()),
                     int((m & ~seen).sum()), (m & ~seen).sum() / max(m.sum(), 1)))
    print(pd.DataFrame(rows, columns=["class", "train_rows", "test_rows", "test_unseen_label",
                                      "unseen_share"]).to_string(index=False, float_format="{:.3f}".format))
    print(f"\ntest rows with a label absent from train: {int((~seen).sum())} of {len(test)}")
    tab = (test.assign(seen=seen).groupby([CLASS_COL, LABEL_COL, "seen"]).size()
           .rename("rows").reset_index().sort_values([CLASS_COL, "rows"], ascending=[True, False]))
    print("\nper attack name (test):")
    print(tab.to_string(index=False))


def predict(run_dir, arm, X):
    run_dir = Path(run_dir)
    X = np.asarray(joblib.load(run_dir / "preprocessing.joblib").transform(X))
    if arm == "rf":
        return joblib.load(run_dir / "model.joblib").predict(X)
    if arm == "xgb":
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(run_dir / "model.json"))
        best = int(booster.attr("best_round"))
        return booster.predict(xgb.DMatrix(X), iteration_range=(0, best)).argmax(axis=1)
    import nesybench  # legacy keras switch, before tf_keras
    import tf_keras
    model = tf_keras.models.load_model(run_dir / "model.keras", compile=False)
    return np.asarray(model.predict(X, batch_size=4096, verbose=0)).argmax(axis=1)


def recall_tables(test, spec, seen, models_dir, seeds):
    X = test.drop(columns=[c for c in spec.non_features if c in test.columns])
    y = test[CLASS_COL].map({c: i for i, c in enumerate(spec.classes)}).to_numpy()
    out = []
    for arm in ARMS:
        runs = sorted(Path(models_dir).glob(f"nslkdd_{arm}_tuned_seed*"))
        runs = [r for r in runs if int(r.name.rsplit("seed", 1)[1]) in seeds]
        if not runs:
            print(f"{arm}: no saved model found")
            continue
        acc = {}
        for r in runs:
            try:
                pred = predict(r, arm, X)
            except Exception as exc:
                print(f"{arm} {r.name}: {type(exc).__name__}: {exc}")
                break
            for i, cls in enumerate(spec.classes):
                for name, mask in (("seen", seen), ("unseen", ~seen)):
                    m = (y == i) & mask
                    if m.any():
                        acc.setdefault((cls, name), []).append((pred[m] == i).mean())
        for (cls, name), v in acc.items():
            out.append((arm, cls, name, len(v), float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0))
    if out:
        tab = pd.DataFrame(out, columns=["arm", "class", "label_in_train", "seeds", "recall", "sd"])
        print("\nrecall of the saved models, split by whether the attack name appears in train:")
        print(tab.pivot(index=["arm", "class"], columns="label_in_train", values="recall")
              .to_string(float_format="{:.3f}".format))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()
    spec = get_spec("nslkdd")
    train, test = load_dataset("nslkdd", "train"), load_dataset("nslkdd", "test")
    data_tables(train, test, spec)
    if args.models_dir:
        recall_tables(test, spec, seen_flags(train, test), args.models_dir, set(args.seeds))


if __name__ == "__main__":
    main()
