#!/usr/bin/env bash
# E13 timing: ltn_full epoch time against the number of axioms, one process, every K on this machine.
#   bash e13_time.sh <dataset> [epochs] [seeds...]
# dataset: nslkdd | cicids2017 | cicddos2019; epochs default 5 (no early stop); seeds default 0 1 2.
# Order: seed outer, K inner, so a slow spell of the machine hits every K. Finished runs are skipped.
# Results in results/e13_time/, summary printed at the end (median epoch time, first epoch left out).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
[ -d configs ] || cd .. || exit 1
ROOT="$PWD"

DS="${1:-}"
case " nslkdd cicids2017 cicddos2019 " in *" $DS "*) ;; *) echo "Uso: bash e13_time.sh <nslkdd|cicids2017|cicddos2019> [epoche] [seed...]"; exit 2 ;; esac
EPOCHS="${2:-5}"
shift $(( $# >= 2 ? 2 : 1 ))
SEEDS="${*:-0 1 2}"
PY="$(command -v python || command -v python3)" || { echo "python non trovato"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
OUT="results/e13_time"
mkdir -p "$OUT/configs" logs

# configs: the base (no copies) and every E13 level, epochs fixed, patience off
"$PY" - "$DS" "$EPOCHS" "$OUT/configs" <<'PY' || exit 1
import glob, re, sys, yaml
from nesybench.logic.edit import kb_file
from nesybench.logic.kb import load_kb
ds, epochs, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
files = sorted(glob.glob(f"configs/stress/{ds}_ltn_full_tuned_e13_k*.yaml"))
if not files:
    sys.exit(f"nessuna config E13 per {ds}")
base = yaml.safe_load(open(files[0]))
base.pop("kb_edit")
base["run_id"] = re.sub(r"_e13_k\d+$", "", base["run_id"])
k0 = len(load_kb(kb_file(base)).axioms)
levels = [(k0, base)] + [(int(re.search(r"_k(\d+)\.yaml$", f)[1]), yaml.safe_load(open(f))) for f in files]
for k, cfg in levels:
    cfg["run_id"] = f"{ds}_ltn_full_tuned_e13t_k{k}"
    cfg["training"]["max_epochs"] = epochs
    cfg["training"]["early_stopping_patience"] = epochs + 1
    cfg["evaluate_test"] = False
    yaml.safe_dump(cfg, open(f"{out}/{cfg['run_id']}.yaml", "w"), sort_keys=False)
print("K:", " ".join(str(k) for k, _ in sorted(levels, key=lambda t: t[0])))
PY

CFGS=$(ls "$OUT"/configs/"${DS}"_ltn_full_tuned_e13t_k*.yaml | sort -V)
for seed in $SEEDS; do
    for CFG in $CFGS; do
        RUN_ID="$(basename "$CFG" .yaml)"
        echo "$(date +%T) $RUN_ID seed $seed"
        "$PY" -m nesybench.run --config "$CFG" --seeds "$seed" --resume --results-dir "$OUT" \
            > "logs/${RUN_ID}_s${seed}.log" 2>&1 || { echo "ERRORE: logs/${RUN_ID}_s${seed}.log"; exit 1; }
    done
done

"$PY" - "$OUT" "$DS" <<'PY'
import glob, json, re, sys
import numpy as np
out, ds = sys.argv[1], sys.argv[2]
by_k = {}
for f in glob.glob(f"{out}/{ds}_ltn_full_tuned_e13t_k*_seed*.json"):
    r = json.load(open(f))
    k = int(re.search(r"_k(\d+)_seed", f)[1])
    by_k.setdefault(k, []).extend(r["training"]["epoch_seconds"][1:])
base = np.median(by_k[min(by_k)])
print(f"{ds} E13: secondi per epoca (mediana, prima epoca esclusa), un processo, {r['machine']}")
for k in sorted(by_k):
    m = np.median(by_k[k])
    print(f"  K={k:2d}  epoche {len(by_k[k]):3d}  mediana {m:7.3f} s  rapporto sulla base {m / base:5.3f}")
PY
