"""Riepilogo dei tuning: prova migliore, bordi dello spazio, tempi, tetto di epoche.

    python analysis/tuning_report.py [results/tuning_cic]

Legge i tune_*.json di ogni braccio e le run tune_*_tNN_seedNN.json se ci sono.
Solo validation, solo lettura.
"""
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

EDGE = 0.05  # quota del range che conta come bordo per i parametri continui


def is_summary(path):
    return not re.search(r"_trial\d+$|_seed\d+$", Path(path).stem)


def edge_of(value, spec):
    """'min', 'max' o None per un valore rispetto alla sua voce dello spazio."""
    kind, arg = next(iter(spec.items()))
    if kind == "choice":
        nums = [a for a in arg if isinstance(a, (int, float)) and not isinstance(a, bool)]
        if len(nums) != len(arg) or len(arg) < 3 or not isinstance(value, (int, float)):
            return None
        return "min" if value == min(arg) else "max" if value == max(arg) else None
    low, high = arg
    if kind == "int":
        return "min" if value == low else "max" if value == high else None
    if kind == "loguniform":
        pos = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    else:
        pos = (value - low) / (high - low)
    return "min" if pos <= EDGE else "max" if pos >= 1 - EDGE else None


def arm_report(summary):
    tid = summary["tuning_id"]
    trials = summary["trials"]
    ok = sorted((t for t in trials if t["status"] == "ok"), key=lambda t: -t["mean"])
    print(f"{tid}  prove {len(ok)}/{summary['budget']} ok, {len(trials) - len(ok)} fallite")
    if summary.get("trial_set"):
        print(f"  solo nelle prove: {summary['trial_set']}")
    if not ok:
        print()
        return
    best = ok[0]
    print(f"  migliore: prova {best['trial']}  {best['mean']:.4f} +- {best['std']:.4f}  "
          f"seed {[round(s, 4) for s in best['seed_scores']]}")
    gaps = "  ".join(f"{t['mean']:.4f} (prova {t['trial']})" for t in ok[1:3])
    print(f"  seguenti: {gaps}   peggiore {ok[-1]['mean']:.4f}")
    edges = []
    for key, value in best["params"].items():
        side = edge_of(value, summary["space"][key])
        edges.append(f"{key}={value}" + (f" [{side}]" if side else ""))
    print("  parametri: " + ", ".join(edges))
    fit = sum(t["fit_s_mean"] * len(summary["seeds"]) for t in ok)
    print(f"  fit: migliore {best['fit_s_mean']:.0f} s per seed, "
          f"somma di tutte le prove {fit / 3600:.2f} h")
    commits = sorted({c for t in ok for c in t["git_commits"]})
    dirty = any(t["git_dirty"] for t in ok)
    print(f"  commit {','.join(commits)}  git_dirty={dirty}\n")


def seed_files(root):
    runs = defaultdict(list)
    for f in sorted(glob.glob(str(root / "tune_*_t??_seed*.json"))):
        r = json.loads(Path(f).read_text())
        runs[re.match(r"(.+)_t(\d+)$", r["run_id"]).group(1)].append(r)
    return runs


def epoch_report(runs, best_trial):
    rows = []
    for tid, recs in sorted(runs.items()):
        stats = []
        for r in recs:
            tr = r.get("training") or {}
            cap = (r["config"].get("training") or {}).get("max_epochs")
            if "epochs_run" not in tr or not cap:
                continue
            trial = int(re.search(r"_t(\d+)$", r["run_id"]).group(1))
            stats.append((trial, tr["epochs_run"] >= cap, tr["best_epoch"] >= cap - 1))
        if stats:
            n = len(stats)
            chosen = [s for s in stats if s[0] == best_trial.get(tid)]
            rows.append((tid, n, sum(s[1] for s in stats), sum(s[2] for s in stats),
                         len(chosen), sum(s[1] for s in chosen), sum(s[2] for s in chosen)))
    if not rows:
        return
    print("Tetto di epoche (run al tetto, migliore epoca nelle ultime due; poi la prova scelta)")
    for tid, n, hit, late, nc, hc, lc in rows:
        print(f"  {tid:34} {hit:>3}/{n:<3} {late:>3}/{n:<3} | scelta: {hc}/{nc} {lc}/{nc}")
    print()


def machines(runs):
    seen = {(r["machine"], r["versions"].get("tensorflow"), r["versions"].get("xgboost"),
             r["versions"].get("sklearn")) for recs in runs.values() for r in recs}
    for m in sorted(seen, key=str):
        print(f"macchina: {m[0]}  tensorflow {m[1]}  xgboost {m[2]}  sklearn {m[3]}")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/tuning_cic")
    summaries = [json.loads(Path(f).read_text())
                 for f in sorted(glob.glob(str(root / "tune_*.json"))) if is_summary(f)]
    if not summaries:
        raise SystemExit(f"nessun riepilogo tune_*.json in {root}")
    for s in summaries:
        arm_report(s)
    runs = seed_files(root)
    best_trial = {s["tuning_id"]: s["best"]["trial"] for s in summaries if s.get("best")}
    epoch_report(runs, best_trial)
    machines(runs)


if __name__ == "__main__":
    main()
