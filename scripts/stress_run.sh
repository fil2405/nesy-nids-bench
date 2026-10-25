#!/usr/bin/env bash
# Runs stress configs with the test scored; finished seeds are skipped on restart.
#   bash stress_run.sh <pattern> [workers]
# pattern: glob on configs/stress, e.g. 'cicids2017_ltn_full_tuned_e9_*' (quote it)
set -uo pipefail
cd "$(dirname "$0")" || exit 1
[ -d configs ] || cd .. || exit 1
ROOT="$PWD"
PAT="${1:-}"
WORKERS="${2:-1}"
[ -n "$PAT" ] || { echo "Uso: bash stress_run.sh '<pattern>' [processi]"; exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "processi: serve un intero >= 1"; exit 1; }
PY="$(command -v python || command -v python3)" || { echo "python non trovato"; exit 1; }

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p results logs

shopt -s nullglob
FILES=(configs/stress/${PAT}.yaml)
[ "${#FILES[@]}" -gt 0 ] || { echo "nessuna config per '$PAT'"; exit 1; }
echo "== ${#FILES[@]} config, $WORKERS processi"

FAILED=0
for CFG in "${FILES[@]}"; do
    RUN_ID="$("$PY" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CFG")"
    SEEDS="$("$PY" -c "import sys,yaml;print(*yaml.safe_load(open(sys.argv[1]))['seeds'])" "$CFG")"
    echo "== $RUN_ID"
    PIDS=(); PSEED=()
    for seed in $SEEDS; do
        while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
        "$PY" -m nesybench.run --config "$CFG" --evaluate-test --resume --seeds "$seed" \
            > "logs/stress_${RUN_ID}_s${seed}.log" 2> "logs/stress_${RUN_ID}_s${seed}.err" &
        PIDS+=($!); PSEED+=("$seed")
    done
    for k in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$k]}"; then
            echo "ERRORE: $RUN_ID seed ${PSEED[$k]} fallito" >&2
            tail -n 10 "logs/stress_${RUN_ID}_s${PSEED[$k]}.err" >&2
            FAILED=1
        fi
    done
done

OUT="risultati_stress_$(date +%Y%m%d_%H%M).tar.gz"
LIST="$(mktemp)"
for CFG in "${FILES[@]}"; do
    RUN_ID="$("$PY" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CFG")"
    find results -maxdepth 1 -name "${RUN_ID}_seed*.json" >> "$LIST"
    find logs -maxdepth 1 -name "stress_${RUN_ID}_*" >> "$LIST"
done
tar -czf "$OUT" -T "$LIST" && rm -f "$LIST"
sha256sum "$OUT"
[ "$FAILED" -eq 0 ] && echo "Completato. Da rimandare: $OUT" || { echo "Con errori: rilanciare per riprendere."; exit 1; }
