"""Quante run di tuning toccano il tetto di epoche (decisioni_protocollo.md §36, punto 21).

    python analysis/epoch_cap_report.py [results/tuning]

Per ogni braccio neurale: quota di run con epochs_run == max_epochs, quota con la migliore
epoca nelle ultime due, e gli stessi numeri per la prova scelta. Solo lettura dei JSON.
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/tuning")
best_trial = {}
for f in glob.glob(str(root / "tune_*.json")):
    name = Path(f).stem
    if re.search(r"_trial\d+$|_seed\d+$|_t\d+_seed\d+$", name):
        continue
    summary = json.loads(Path(f).read_text())
    if summary.get("best"):
        best_trial[summary["tuning_id"]] = summary["best"]["trial"]

arms = defaultdict(list)
for f in sorted(glob.glob(str(root / "tune_*_t??_seed*.json"))):
    r = json.loads(Path(f).read_text())
    tr = r.get("training") or {}
    cap = (r["config"].get("training") or {}).get("max_epochs")
    if "epochs_run" not in tr or not cap:
        continue
    m = re.match(r"(.+)_t(\d+)$", r["run_id"])
    arms[m.group(1)].append({
        "trial": int(m.group(2)), "cap": cap,
        "hit": tr["epochs_run"] >= cap, "late": tr["best_epoch"] >= cap - 1,
    })

print(f"{'braccio':34} {'run':>5} {'al tetto':>9} {'migliore finale':>16} | {'prova scelta: run':>18} {'al tetto':>9} {'finale':>7}")
for tid, runs in sorted(arms.items()):
    n = len(runs)
    hit, late = sum(x["hit"] for x in runs), sum(x["late"] for x in runs)
    chosen = [x for x in runs if x["trial"] == best_trial.get(tid)]
    print(f"{tid:34} {n:>5} {hit:>4}/{n:<4} {late:>9}/{n:<5} | {len(chosen):>18} "
          f"{sum(x['hit'] for x in chosen):>9} {sum(x['late'] for x in chosen):>7}")
if not arms:
    print("nessuna run con epochs_run: niente da leggere (i bracci classici non hanno epoche)")
