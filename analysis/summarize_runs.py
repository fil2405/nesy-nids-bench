import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FOCUS = ("R2L", "U2R")


def load_runs(results_dir):
    runs = defaultdict(dict)
    for path in sorted(Path(results_dir).glob("*.json")):
        rec = json.loads(path.read_text())
        runs[rec["config"]["run_id"]][rec["seed"]] = rec
    return runs


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def class_pr(rec, name):
    m = rec["metrics"]["val"]
    i = m["class_names"].index(name)
    return m["precision"][i], m["recall"][i]


def macro_f1(rec):
    return rec["metrics"]["val"]["macro_f1"]


def summarize(runs, baseline):
    if baseline not in runs:
        raise SystemExit(f"baseline run_id not found: {baseline}")
    for run_id, by_seed in runs.items():
        recs = [by_seed[s] for s in sorted(by_seed)]
        commits = sorted({r["git_commit"] for r in recs})
        dirty = sorted({bool(r["git_dirty"]) for r in recs})
        print(f"{run_id}  n={len(recs)}  commit={','.join(commits)}  git_dirty={dirty}")
        m, s = mean_std([macro_f1(r) for r in recs])
        print(f"  macro-F1  {m:.4f} +- {s:.4f}")
        for name in FOCUS:
            p, r_ = zip(*(class_pr(r, name) for r in recs))
            (pm, ps), (rm, rs) = mean_std(p), mean_std(r_)
            print(f"  {name}  P {pm:.3f} +- {ps:.3f}  R {rm:.3f} +- {rs:.3f}")
        for key, label in (("best_epoch", "best epoch"), ("best_round", "best round")):
            if key in recs[0]["training"]:  # rf has neither
                ep = [r["training"][key] for r in recs]
                em, es = mean_std(ep)
                print(f"  {label}  {em:.1f} +- {es:.1f}  [{min(ep)}-{max(ep)}]")
        if run_id != baseline:
            common = sorted(set(by_seed) & set(runs[baseline]))
            diff = [macro_f1(runs[baseline][k]) - macro_f1(by_seed[k]) for k in common]
            wins = sum(d > 0 for d in diff)
            print(f"  {baseline} wins in {wins}/{len(common)} seeds, "
                  f"mean diff {np.mean(diff):+.4f}")
            if len(common) != len(by_seed) or len(common) != len(runs[baseline]):
                print("  warning: seed sets differ")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--baseline", default="nslkdd_mlp_noweights")
    args = ap.parse_args()
    summarize(load_runs(args.results_dir), args.baseline)


if __name__ == "__main__":
    main()
