#!/usr/bin/env bash
# Cost measurements for the compute plan (not results). Run: nohup bash scripts/measure_block0.sh &
# Logs, time -v and GPU samples go to $OUT; the repo is not touched while it runs.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-$HOME/tmp_results/measure}"
mkdir -p "$OUT/results"

measure() {  # name config [extra args for nesybench.run]
    local name=$1 config=$2; shift 2
    echo "$(date +%T) start $name"
    nvidia-smi dmon -s um -d 5 > "$OUT/$name.dmon" 2>/dev/null &
    local dmon=$!
    /usr/bin/time -v -o "$OUT/$name.time" \
        python -m nesybench.run --config "$config" --results-dir "$OUT/results" "$@" \
        > "$OUT/$name.log" 2>&1 || echo "$name failed"
    kill "$dmon" 2>/dev/null; wait "$dmon" 2>/dev/null
    echo "$(date +%T) end $name"
}

measure M2_ltn_sup        configs/measure_cicids2017_ltn_sup.yaml
measure M1_xgb            configs/measure_cicids2017_xgb.yaml
measure M3_mlp_f025       configs/measure_cicids2017_mlp_f025.yaml
measure M3_mlp_f050       configs/measure_cicids2017_mlp_f050.yaml
measure M4_mlp_seed1      configs/measure_cicids2017_mlp.yaml --seeds 1
measure M4_mlp_seed2      configs/measure_cicids2017_mlp.yaml --seeds 2
measure M1_rf             configs/measure_cicids2017_rf.yaml   # last: highest OOM risk

python analysis/measure_table.py "$OUT" | tee "$OUT/table.txt"
