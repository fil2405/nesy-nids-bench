#!/usr/bin/env bash
# Final runs of one arm on a CIC dataset: test scored, model and preprocessing saved.
#   bash final_cic.sh <dataset> <arm> [workers]
# dataset: cicids2017 | cicddos2019
# arm: mlp_noweights | mlp_weights | ltn_sup | ltn_full | rf | xgb | rule_only
# workers: seeds running at the same time (default 1). Finished seeds are skipped on restart.
# BEST_DIR: where the tune_<dataset>_<arm>_best.yaml files are (default results/tuning, then results/tuning_cic).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
[ -d configs ] || cd .. || exit 1
ROOT="$PWD"

DATASETS="cicids2017 cicddos2019"
ARMS="mlp_noweights mlp_weights ltn_sup ltn_full rf xgb rule_only"
DS="${1:-}"
ARM="${2:-}"
WORKERS="${3:-1}"
die() { echo "ERRORE: $*" >&2; exit 1; }

usage() {
    echo "Uso: bash final_cic.sh <dataset> <braccio> [processi]"
    echo "Dataset: $DATASETS"; echo "Bracci: $ARMS"; exit 2
}
case " $DATASETS " in *" $DS "*) ;; *) usage ;; esac
case " $ARMS " in *" $ARM "*) ;; *) usage ;; esac
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || die "processi: serve un intero >= 1 (ricevuto: $WORKERS)"
PY="$(command -v python || command -v python3)" || die "python non trovato (attivare il venv)"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true   # seeds of the same arm share the GPU

RES="results"
MODELS="$RES/models"
LOGS="logs"
TAG="${DS}_${ARM}"
mkdir -p "$RES" "$MODELS" "$LOGS"

# rows of the parts the runs read (train is the whole file on cicids2017: the split is made in the run)
case "$DS" in
    cicids2017) PARTS="train:1715341" ;;
    cicddos2019) PARTS="train:340560 test:79083" ;;
esac

# config: the best one of the tuning; rule_only has its own, no tuning
case "$ARM" in
    rule_only) CFG="configs/${DS}_rules.yaml"; MODEL="rule_only" ;;
    mlp_noweights|mlp_weights) MODEL="mlp_ce" ;;
    *) MODEL="$ARM" ;;
esac
if [ "$ARM" != rule_only ]; then
    CFG=""
    for d in "${BEST_DIR:-}" results/tuning results/tuning_cic; do
        [ -n "$d" ] || continue
        if [ -f "$d/tune_${TAG}_best.yaml" ]; then CFG="$d/tune_${TAG}_best.yaml"; break; fi
    done
    [ -n "$CFG" ] || die "manca tune_${TAG}_best.yaml (BEST_DIR=${BEST_DIR:-}, results/tuning, results/tuning_cic)"
fi
[ -f "$CFG" ] || die "manca $CFG"

# one launch per arm at a time
if command -v flock >/dev/null; then
    exec 9> "$RES/.final_${TAG}.lock"
    flock -n 9 || die "il braccio $TAG e' gia' in esecuzione"
fi

echo "== controllo rapido ($TAG, config $CFG)"
if ! "$PY" - "$CFG" "$ARM" "$MODEL" "$DS" "$PARTS" <<'PY' 2>&1 | tee "$LOGS/final_${TAG}_check.log"
import os
import sys

import yaml

import nesybench  # sets TF_USE_LEGACY_KERAS before tensorflow is imported

cfg_path, arm, model, dataset, parts = sys.argv[1:6]
cfg = yaml.safe_load(open(cfg_path))
errors = []


def fail(msg):
    errors.append(msg)
    print("KO  " + msg)


for key, want in (("dataset", dataset), ("model", model)):
    if cfg.get(key) != want:
        fail(f"{key}: {cfg.get(key)!r}, atteso {want!r}")
print(f"ok  config      run_id {cfg.get('run_id')}, seed {cfg.get('seeds')}")
if cfg.get("evaluate_test"):
    print("!!  evaluate_test e' gia' true nel config")
if cfg.get("kb") and not os.path.exists(cfg["kb"]):
    fail(f"manca {cfg['kb']}")
if cfg.get("c5_kb") and not os.path.exists(cfg["c5_kb"]):
    fail(f"manca {cfg['c5_kb']}")
if "network" in cfg:
    print(f"ok  rete        {cfg['network']['hidden_units']} dropout {cfg['network']['dropout']}, "
          f"batch {cfg['training']['batch_size']}, max_epochs {cfg['training']['max_epochs']}")

if model in ("mlp_ce", "ltn_sup", "ltn_full"):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("ok  GPU         " + ", ".join(g.name for g in gpus))
    elif os.environ.get("NESY_ALLOW_CPU") == "1":
        print("!!  nessuna GPU, NESY_ALLOW_CPU=1: si prosegue su CPU")
    else:
        fail("nessuna GPU NVIDIA visibile a TensorFlow")

from nesybench.data import load_dataset

for item in parts.split():
    part, rows = item.split(":")
    n = len(load_dataset(dataset, part))
    print(f"{'ok ' if n == int(rows) else 'KO '} dati {dataset} {part}: {n} righe")
    if n != int(rows):
        errors.append(f"{dataset} {part}: {n} righe, attese {rows}")

print("controllo KO" if errors else "controllo OK")
sys.exit(1 if errors else 0)
PY
then
    die "controllo fallito, le run non partono (log: $LOGS/final_${TAG}_check.log)"
fi

RUN_ID="$("$PY" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CFG")"
SEEDS="$("$PY" -c "import sys,yaml;print(*yaml.safe_load(open(sys.argv[1]))['seeds'])" "$CFG")"
N_SEEDS="$(echo $SEEDS | wc -w)"
count() { find "$RES" -maxdepth 1 -name "${RUN_ID}_seed*.json" | wc -l; }
echo "== finali $TAG: $WORKERS processi, seed gia' fatti $(count)/$N_SEEDS"

PIDS=()
PSEED=()
trap 'kill "${PIDS[@]}" 2>/dev/null; echo; echo "Interrotto: rilanciare lo stesso comando per riprendere."; exit 130' INT TERM
last=-1
progress() {
    local n; n="$(count)"
    if [ "$n" != "$last" ]; then echo "$(date +%T) seed finiti $n/$N_SEEDS"; last="$n"; fi
}
for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 10; progress; done
    "$PY" -m nesybench.run --config "$CFG" --evaluate-test --resume \
        --save-model-dir "$MODELS" --seeds "$seed" \
        > "$LOGS/final_${TAG}_s${seed}.log" 2> "$LOGS/final_${TAG}_s${seed}.err" &
    PIDS+=($!)
    PSEED+=("$seed")
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do sleep 10; progress; done
progress

FAILED=0
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then
        echo "ERRORE: seed ${PSEED[$k]} fallito, ultime righe:" >&2
        tail -n 15 "$LOGS/final_${TAG}_s${PSEED[$k]}.err" >&2
        FAILED=1
    fi
done

echo "== verifica dei risultati"
if ! "$PY" - "$RES" "$RUN_ID" "$MODELS" $SEEDS <<'PY'
import json
import sys
from pathlib import Path

res, run_id, models, *seeds = sys.argv[1:]
bad = []
total = 0
for seed in seeds:
    path = Path(res) / f"{run_id}_seed{seed}.json"
    if not path.exists():
        bad.append(f"seed {seed}: manca {path.name}")
        continue
    rec = json.loads(path.read_text())
    if "test" not in rec["metrics"]:
        bad.append(f"seed {seed}: parte test non valutata")
    if not rec.get("artifacts"):
        bad.append(f"seed {seed}: modello non salvato")
    else:
        total += sum(rec["artifacts"]["files"].values())
    print(f"seed {seed}: test macro-F1 {rec['metrics'].get('test', {}).get('macro_f1', float('nan')):.4f}, "
          f"commit {rec['git_commit']}, git_dirty {rec['git_dirty']}")
print(f"modelli salvati: {total / 1e6:.1f} MB in {len(seeds)} seed (somma dei file)")
print("\n".join("KO  " + b for b in bad) if bad else "verifica OK")
sys.exit(1 if bad else 0)
PY
then
    FAILED=1
fi
du -sh "$MODELS/${RUN_ID}"_seed* 2>/dev/null

# archives of this arm only
SUFFIX=""
[ "$FAILED" -eq 0 ] || SUFFIX="_incompleto"
OUT="risultati_final_${TAG}${SUFFIX}.tar.gz"
OUTM="modelli_${TAG}${SUFFIX}.tar"
LIST="$(mktemp)"
find "$RES" -maxdepth 1 -type f -name "${RUN_ID}_seed*.json" > "$LIST"
find "$LOGS" -maxdepth 1 -type f -name "final_${TAG}_*" >> "$LIST"
tar -czf "$OUT" -T "$LIST" && rm -f "$LIST"
tar -cf "$OUTM" -C "$MODELS" $(cd "$MODELS" && ls -d "${RUN_ID}"_seed* 2>/dev/null)
sha256sum "$OUT" "$OUTM"

if [ "$FAILED" -ne 0 ]; then
    echo "Finali NON complete. Rilanciare lo stesso comando per riprendere; in ogni caso rimandare $OUT."
    exit 1
fi
rm -f "risultati_final_${TAG}_incompleto.tar.gz" "modelli_${TAG}_incompleto.tar"
echo "Finali $TAG completate. Da rimandare: $OUT e $OUTM"
