"""Selected values of the runs, read from the local result files.

    python analysis/selected_values.py [--results-dir results] [--logs logs]

Read only. Sections: epochs of the finals, tuning choices, inputs of CIC-DDoS2019, split sizes,
commit and versions of each series, oneDNN in the logs.
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from summarize_runs import load_runs  # noqa: E402

DATASETS = ("nslkdd", "cicids2017", "cicddos2019")
NEURAL = ("mlp_noweights", "mlp_weights", "cnn", "ltn_sup", "ltn_full")
ARMS = (*NEURAL, "rf", "xgb")


def section(title):
    print(f"\n== {title}")


def finals_epochs(runs):
    section("finals: best epoch min-max, runs at the cap (App. C tab:app-epochs, Ch. 9 epoch cap)")
    for ds in DATASETS:
        for arm in NEURAL:
            by_seed = runs.get(f"{ds}_{arm}_tuned")
            if not by_seed:
                continue
            best = [r["training"]["best_epoch"] for r in by_seed.values()]
            run = [r["training"]["epochs_run"] for r in by_seed.values()]
            cap = {r["config"]["training"].get("max_epochs") for r in by_seed.values()}
            at_cap = sum(e == max(cap) for e in run) if cap and None not in cap else "?"
            print(f"  {ds:12} {arm:14} n={len(best):2d}  best {min(best)}-{max(best)}  "
                  f"run {min(run)}-{max(run)}  cap {sorted(cap)}  at cap {at_cap}")


def tuning_choices(tuning_dirs):
    section("tuning: selected values (App. B), file and date of the search")
    for d in tuning_dirs:
        for path in sorted(glob.glob(f"{d}/tune_*_best.yaml")):
            cfg = yaml.safe_load(Path(path).read_text())
            keep = {"params": cfg.get("params"), "training": cfg.get("training"), "network": cfg.get("network")}
            keep = {k: v for k, v in keep.items() if v}
            print(f"  {path}  {json.dumps(keep, sort_keys=True)}")
        for path in sorted(glob.glob(f"{d}/tune_*.json")):
            if re.search(r"_t\d+_seed\d+\.json$|_trial\d+\.json$", path):
                continue
            s = json.loads(Path(path).read_text())
            best = s.get("best") or {}
            print(f"  {Path(path).name}: trials {len(s.get('trials', []))}, best trial {best.get('trial')}, "
                  f"written {s.get('timestamp_utc', '?')}")


def final_params(runs):
    section("finals: params of RF and XGBoost on NSL-KDD (Ch. 9 tuning_round1)")
    for arm in ("rf", "xgb"):
        by_seed = runs.get(f"nslkdd_{arm}_tuned")
        if by_seed:
            r = by_seed[min(by_seed)]
            print(f"  nslkdd_{arm}_tuned: {json.dumps(r['config'].get('params'), sort_keys=True)}  "
                  f"tuning {r['config'].get('tuning_id', r['config'].get('source', '?'))}")


def ddos_inputs(latency_dir):
    section("network inputs (Ch. 9 n_features)")
    for path in sorted(glob.glob(f"{latency_dir}/*_tuned.json")):
        rep = json.loads(Path(path).read_text())
        if "n_features" in rep:
            print(f"  {Path(path).stem}: {rep['n_features']}")


def split_sizes(runs):
    section("split_sizes of seed 0 (Ch. 9 tab:impl-splits)")
    for ds in DATASETS:
        ids = sorted(i for i in runs if i.startswith(f"{ds}_") and 0 in runs[i]
                     and runs[i][0].get("split_sizes") and re.search(r"_e(4|11|3|13)_", i))
        if ids:
            print(f"  {ds} ({ids[0]}): {json.dumps(runs[ids[0]][0]['split_sizes'])}")


def series(runs):
    section("series: commit, dirty flag, versions, dates")
    groups = defaultdict(list)
    for run_id, by_seed in runs.items():
        ds = next((d for d in (*DATASETS, "cic_transfer") if run_id.startswith(d + "_")), run_id)
        m = re.search(r"_(e\d+t?)_", run_id + "_")
        groups[(ds, m[1] if m else "final")] += list(by_seed.values())
    for (ds, kind), recs in sorted(groups.items()):
        commits = sorted({r.get("git_commit") for r in recs})
        vers = sorted({json.dumps(r.get("versions"), sort_keys=True) for r in recs})
        dates = sorted(r.get("timestamp_utc", "")[:10] for r in recs)
        print(f"  {ds:12} {kind:6} runs {len(recs):4d}  {dates[0]} to {dates[-1]}  commit {commits}  "
              f"dirty {sorted({str(r.get('git_dirty')) for r in recs})}")
        for v in vers:
            print(f"      {v}")


def onednn(logs):
    section("oneDNN in the logs (Ch. 8 TF_ENABLE_ONEDNN_OPTS)")
    on = [p for p in glob.glob(f"{logs}/**/*", recursive=True)
          if Path(p).is_file() and "oneDNN custom operations are on" in Path(p).read_text(errors="ignore")]
    files = [p for p in glob.glob(f"{logs}/**/*", recursive=True) if Path(p).is_file()]
    print(f"  {len(files)} log files, {len(on)} with oneDNN on")
    for p in on[:20]:
        print(f"    {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args()
    runs = load_runs(args.results_dir)
    print(f"{len(runs)} run ids in {args.results_dir}")
    finals_epochs(runs)
    tuning_choices([f"{args.results_dir}/tuning", f"{args.results_dir}/tuning_cic"])
    final_params(runs)
    ddos_inputs(f"{args.results_dir}/latency")
    split_sizes(runs)
    series(runs)
    onednn(args.logs)


if __name__ == "__main__":
    main()
