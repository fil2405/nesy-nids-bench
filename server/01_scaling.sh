#!/usr/bin/env bash
# Test di scaling: P processi uguali insieme (P = 1 2 4 8), con seed fuori dai finali.
# Uso: bash server/01_scaling.sh [config...]     LEVELS="1 2 4" per cambiare i livelli
# Default: i quattro config di misura su CIC-IDS2017 (rete, LTN-sup, RF, XGBoost).
# Un livello che va in OOM e' un risultato: resta nel file di watch con exit_code != 0.
source "$(dirname "$0")/common.sh"
LEVELS="${LEVELS:-1 2 4 8}"
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(configs/measure_cicids2017_mlp.yaml configs/measure_cicids2017_ltn_sup.yaml \
                                     configs/measure_cicids2017_rf.yaml configs/measure_cicids2017_xgb.yaml)

for cfg in "${CONFIGS[@]}"; do
    name="$(basename "$cfg" .yaml)"; out="$SRV/scaling/$name"
    for p in $LEVELS; do
        dir="$out/P$p"; rm -rf "$dir"; mkdir -p "$dir/results"
        echo "$(date +%T) $name P=$p"
        for i in $(seq 0 $((p - 1))); do
            python server/runwatch.py --out "$dir/watch_$i.json" -- \
                python -m nesybench.run --config "$cfg" --seeds $((200 + i)) --results-dir "$dir/results" \
                > "$dir/p$i.log" 2>&1 &
        done
        wait
    done
    python server/scaling_summary.py "$out" | tee "$out/summary.txt"
done
echo "tabelle in $SRV/scaling/*/summary.tsv"
