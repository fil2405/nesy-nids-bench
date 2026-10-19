import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FOCUS = ("R2L", "U2R")
PER_100K = 100_000


def load_runs(results_dir):
    runs = defaultdict(dict)
    for path in sorted(Path(results_dir).glob("*.json")):
        rec = json.loads(path.read_text())
        runs[rec["config"]["run_id"]][rec["seed"]] = rec
    return runs


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def mean_std_finite(values):
    # nan = class or side missing in that seed; skipped, count reported
    v = np.asarray(values, dtype=float)
    ok = v[np.isfinite(v)]
    if len(ok) == 0:
        return float("nan"), float("nan"), 0
    return ok.mean(), (ok.std(ddof=1) if len(ok) > 1 else 0.0), len(ok)


def class_pr(rec, name, part):
    m = rec["metrics"][part]
    i = m["class_names"].index(name)
    return m["precision"][i], m["recall"][i]


def macro_f1(rec, part):
    return rec["metrics"][part]["macro_f1"]


def seeds_note(k, n):
    return "" if k == n else f"  [{k}/{n} seeds]"


def print_pr_auc(recs, part):
    if any("pr_auc" not in r["metrics"][part] for r in recs):
        print("  PR-AUC  n/a")
        return
    names = recs[0]["metrics"][part]["class_names"]
    cells = []
    for i, name in enumerate(names):
        m, s, k = mean_std_finite([r["metrics"][part]["pr_auc"][i] for r in recs])
        cells.append(f"{name} {m:.3f} +- {s:.3f}{seeds_note(k, len(recs))}")
    print("  PR-AUC  " + "  ".join(cells))


def print_operator(recs, part):
    # levels are string keys in the json
    for key, label, fixed in (("fpr_at_tpr", "FPR @ TPR", "tpr"),
                              ("tpr_at_fpr", "TPR @ FPR", "fpr")):
        if any(key not in r["metrics"][part] for r in recs):
            print(f"  {label}  n/a")
            continue
        levels = sorted(recs[0]["metrics"][part][key], key=float)
        for lvl in levels:
            m, s, k = mean_std_finite([r["metrics"][part][key][lvl] for r in recs])
            # false alarms per 100k benign = FPR * 100k
            fpr = m if fixed == "tpr" else float(lvl)
            print(f"  {label} {float(lvl):g}  {m:.4f} +- {s:.4f}  "
                  f"({fpr * PER_100K:.0f} false alarms/100k benign)"
                  f"{seeds_note(k, len(recs))}")


def print_c5(recs, part):
    if any("c5" not in r["metrics"][part] for r in recs):
        return
    for ident in recs[0]["metrics"][part]["c5"]["axioms"]:
        rows = [r["metrics"][part]["c5"]["axioms"][ident] for r in recs]
        n = np.mean([x["n_covered"] for x in rows])
        v, vs, vk = mean_std_finite([x["violation_rate"] for x in rows])
        sat, ss, sk = mean_std_finite([x["satisfaction"] for x in rows])
        prec = mean_std_finite([x.get("rule_precision", float("nan")) for x in rows])[0]
        print(f"  C5 {ident}  covered {n:.0f}  violation {v:.3f} +- {vs:.3f}"
              f"{seeds_note(vk, len(recs))}  satisfaction {sat:.3f} +- {ss:.3f}"
              f"{seeds_note(sk, len(recs))}  rule precision {prec:.3f}")


def summarize(runs, baseline, part):
    if baseline not in runs:
        raise SystemExit(f"baseline run_id not found: {baseline}")
    for run_id, by_seed in runs.items():
        recs = [by_seed[s] for s in sorted(by_seed)]
        commits = sorted({r["git_commit"] for r in recs})
        dirty = sorted({bool(r["git_dirty"]) for r in recs})
        print(f"{run_id}  n={len(recs)}  commit={','.join(commits)}  git_dirty={dirty}")
        if any(part not in r["metrics"] for r in recs):
            print(f"  no {part} metrics in some seeds\n")
            continue
        m, s = mean_std([macro_f1(r, part) for r in recs])
        print(f"  macro-F1  {m:.4f} +- {s:.4f}")
        for name in FOCUS:
            p, r_ = zip(*(class_pr(r, name, part) for r in recs))
            (pm, ps), (rm, rs) = mean_std(p), mean_std(r_)
            print(f"  {name}  P {pm:.3f} +- {ps:.3f}  R {rm:.3f} +- {rs:.3f}")
        print_pr_auc(recs, part)
        print_operator(recs, part)
        print_c5(recs, part)
        for key, label in (("best_epoch", "best epoch"), ("best_round", "best round")):
            if key in recs[0]["training"]:  # rf has neither
                ep = [r["training"][key] for r in recs]
                em, es = mean_std(ep)
                print(f"  {label}  {em:.1f} +- {es:.1f}  [{min(ep)}-{max(ep)}]")
        if run_id != baseline:
            common = sorted(set(by_seed) & set(runs[baseline]))
            base = runs[baseline]
            if any(part not in base[k]["metrics"] for k in common):
                print(f"  baseline has no {part} metrics")
            else:
                diff = [macro_f1(base[k], part) - macro_f1(by_seed[k], part)
                        for k in common]
                wins = sum(d > 0 for d in diff)
                print(f"  {baseline} wins in {wins}/{len(common)} seeds, "
                      f"mean diff {np.mean(diff):+.4f}")
            if len(common) != len(by_seed) or len(common) != len(base):
                print("  warning: seed sets differ")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--baseline", default="nslkdd_mlp_noweights")
    ap.add_argument("--part", default="val", choices=["val", "test"])
    args = ap.parse_args()
    summarize(load_runs(args.results_dir), args.baseline, args.part)


if __name__ == "__main__":
    main()
