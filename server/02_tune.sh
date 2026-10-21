#!/usr/bin/env bash
# Tuning di un dataset: 20 prove x 3 seed per braccio. Riprende da dove si e' fermato.
# Uso: bash server/02_tune.sh DATASET [braccio...]
#      NESY_JOBS_NEURAL=4 NESY_JOBS_CLASSIC=2 bash server/02_tune.sh cicids2017
# Con N > 1 le prove si dividono tra N worker (prova i -> worker i % N) e un --finalize
# scrive riepilogo e _best.yaml. Il test non viene mai valutato.
source "$(dirname "$0")/common.sh"
DS="${1:?dataset: nslkdd | cicids2017 | cicddos2019}"; shift
mapfile -t ARMS < <(arms_or_default "$DS" tune "$@")
[ ${#ARMS[@]} -gt 0 ] || die "nessun braccio pronto per $DS (vedi: python server/plan.py table)"

for ARM in "${ARMS[@]}"; do
    CFG="configs/tuning/${DS}_${ARM}.yaml"
    [ -f "$CFG" ] || { fail "tune $DS/$ARM: manca $CFG"; continue; }
    N="$(jobs_for "$ARM")"
    echo "$(date +%T) tune $DS/$ARM con $N worker"
    if [ "$N" -le 1 ]; then
        python -m nesybench.tune --tuning "$CFG" > "$LOGS/tune_${DS}_${ARM}.log" 2>&1 \
            || fail "tune $DS/$ARM (vedi $LOGS/tune_${DS}_${ARM}.log)"
    else
        seq 0 $((N - 1)) | xargs -P "$N" -I{} bash -c \
            'python -m nesybench.tune --tuning "$0" --worker {} --workers "$1" > "$2" 2>&1' \
            "$CFG" "$N" "$LOGS/tune_${DS}_${ARM}_w{}.log" \
            || { fail "tune $DS/$ARM: un worker e' fallito (log in $LOGS/tune_${DS}_${ARM}_w*.log)"; continue; }
        python -m nesybench.tune --tuning "$CFG" --finalize > "$LOGS/tune_${DS}_${ARM}_finalize.log" 2>&1 \
            || fail "finalize $DS/$ARM (vedi $LOGS/tune_${DS}_${ARM}_finalize.log)"
    fi
    progress
done
[ -s "$FAILED" ] && { echo "Qualcosa e' fallito, vedi $FAILED"; exit 1; }
echo "tuning $DS finito. Raccogli con: bash server/05_collect.sh tune_$DS"
