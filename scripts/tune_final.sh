#!/usr/bin/env bash
# Tuning then final runs of one arm on one dataset, in one go.
#   bash tune_final.sh <dataset> <arm> [workers]
# The trials are split over the workers, then --finalize writes results/tuning/tune_<dataset>_<arm>_best.yaml
# and final_cic.sh runs the 10 final seeds with it. Finished trials and seeds are skipped on restart.
# Archives: risultati_tuning_<dataset>_<arm>.tar.gz, then those of final_cic.sh.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
[ -d configs ] || cd .. || exit 1
ROOT="$PWD"

DS="${1:-}"
ARM="${2:-}"
WORKERS="${3:-1}"
TUNING="configs/tuning/${DS}_${ARM}.yaml"
[ -f "$TUNING" ] || { echo "Uso: bash tune_final.sh <dataset> <braccio> [processi] (manca $TUNING)"; exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "processi: serve un intero >= 1"; exit 1; }
PY="$(command -v python || command -v python3)" || { echo "python non trovato"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p results/tuning logs
TAG="${DS}_${ARM}"

echo "== tuning $TAG ($WORKERS processi, $(date +%T))"
PIDS=()
if [ "$WORKERS" -eq 1 ]; then
    "$PY" -m nesybench.tune --tuning "$TUNING" > "logs/tune_${TAG}_w0.log" 2>&1 &
    PIDS+=($!)
else
    for i in $(seq 0 $((WORKERS - 1))); do
        "$PY" -m nesybench.tune --tuning "$TUNING" --worker "$i" --workers "$WORKERS" \
            > "logs/tune_${TAG}_w${i}.log" 2>&1 &
        PIDS+=($!)
    done
fi
trap 'kill "${PIDS[@]}" 2>/dev/null; echo "Interrotto: rilanciare lo stesso comando per riprendere."; exit 130' INT TERM
FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || FAILED=1; done
trap - INT TERM
if [ "$FAILED" -ne 0 ]; then
    echo "ERRORE nel tuning di $TAG, log in logs/tune_${TAG}_w*.log"; tail -n 15 logs/tune_"${TAG}"_w*.log; exit 1
fi
if [ "$WORKERS" -gt 1 ]; then
    "$PY" -m nesybench.tune --tuning "$TUNING" --finalize > "logs/tune_${TAG}_finalize.log" 2>&1 \
        || { echo "ERRORE in --finalize"; tail -n 15 "logs/tune_${TAG}_finalize.log"; exit 1; }
fi
[ -f "results/tuning/tune_${TAG}_best.yaml" ] || { echo "manca results/tuning/tune_${TAG}_best.yaml"; exit 1; }
tar -czf "risultati_tuning_${TAG}.tar.gz" results/tuning/tune_"${TAG}"* logs/tune_"${TAG}"_*
echo "== tuning $TAG finito ($(date +%T)), archivio risultati_tuning_${TAG}.tar.gz"

bash scripts/final_cic.sh "$DS" "$ARM" "$WORKERS"
