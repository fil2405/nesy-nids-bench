#!/usr/bin/env bash
# Run finali: 10 seed per braccio con la config scelta dal tuning, test valutato una volta.
# Uso: bash server/03_final.sh DATASET [braccio...]
# Rifiuta di partire con il checkout sporco (git_dirty falso o la run non e' citabile).
# Le run gia' fatte si saltano; non ne sovrascrive nessuna.
source "$(dirname "$0")/common.sh"
DS="${1:?dataset}"; shift
require_clean
mapfile -t ARMS < <(arms_or_default "$DS" final "$@")
[ ${#ARMS[@]} -gt 0 ] || die "nessun braccio pronto per $DS"
read -ra SEEDS <<< "$(python server/plan.py seeds)"

for ARM in "${ARMS[@]}"; do
    IFS=$'\t' read -r CFG RUN_ID < <(python server/make_final_config.py "$DS" "$ARM") \
        || { fail "final $DS/$ARM: config non pronto"; continue; }
    N="$(jobs_for "$ARM")"
    todo=()
    for s in "${SEEDS[@]}"; do [ -f "results/${RUN_ID}_seed${s}.json" ] || todo+=("$s"); done
    echo "$(date +%T) final $DS/$ARM: ${#todo[@]} seed da fare, $N in parallelo"
    [ ${#todo[@]} -eq 0 ] && continue
    printf '%s\n' "${todo[@]}" | xargs -P "$N" -I{} bash -c \
        'python -m nesybench.run --config "$0" --seeds {} > "$1" 2>&1' \
        "$CFG" "$LOGS/final_${DS}_${ARM}_seed{}.log" \
        || fail "final $DS/$ARM: almeno un seed e' fallito (log in $LOGS/final_${DS}_${ARM}_seed*.log)"
    progress
done
python server/plan.py audit || fail "ci sono run non citabili (vedi sopra)"
[ -s "$FAILED" ] && { echo "Qualcosa e' fallito, vedi $FAILED"; exit 1; }
echo "finali $DS finite. Raccogli con: bash server/05_collect.sh final_$DS"
