#!/usr/bin/env bash
# Every table of Ch. 10 and Appendix C from the result files, plus the values the open FIXMEs ask for.
#   bash make_tables.sh [results_dir]
# Output in <results_dir>/tables: one .txt per command, the .tex tables and the figures,
# then tables.tar.gz. A command whose files are missing is reported and skipped.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
[ -d configs ] || cd .. || exit 1
ROOT="$PWD"
RES="${1:-results}"
OUT="$RES/tables"
PY="$(command -v python || command -v python3)" || { echo "python non trovato"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
FAILED=()

run() {  # run <name> <command...>
    local name="$1"; shift
    if "$@" > "$OUT/$name.txt" 2> "$OUT/$name.err"; then
        rm -f "$OUT/$name.err"; echo "ok      $name"
    else
        echo "MANCA   $name ($(tail -n 1 "$OUT/$name.err"))"; FAILED+=("$name")
    fi
}

A=analysis
for DS in nslkdd cicids2017 cicddos2019; do
    T() { echo "${DS}_$1_tuned"; }
    run "${DS}_summary" "$PY" $A/summarize_runs.py --results-dir "$RES" --part test --prefix "$DS" \
        --baseline "$(T mlp_noweights)"
    run "${DS}_pairs" "$PY" $A/compare_runs.py --results-dir "$RES" --part test \
        --pair "$(T mlp_noweights)" "$(T mlp_weights)" --pair "$(T mlp_weights)" "$(T ltn_sup)" \
        --pair "$(T ltn_sup)" "$(T ltn_full)" --pair "$(T ltn_full)" "$(T rf)" --pair "$(T ltn_full)" "$(T xgb)" \
        --rank "$(T mlp_noweights)" "$(T mlp_weights)" "$(T ltn_sup)" "$(T ltn_full)" "$(T rf)" "$(T xgb)" \
        --tex "$OUT/${DS}_pairs.tex" --cd-plot "$OUT/${DS}_cd.pdf"
    run "${DS}_cnn" "$PY" $A/compare_runs.py --results-dir "$RES" --part test \
        --pair "$(T mlp_weights)" "$(T cnn)"
    for E in e7 e8; do
        run "${DS}_$E" "$PY" $A/stress_curves.py --results-dir "$RES" --exp $E --dataset "$DS" \
            --tex "$OUT/${DS}_$E" --plot "$OUT/${DS}_$E.pdf"
    done
    for E in e3 e4 e11 e13; do
        [ "$DS$E" = cicddos2019e4 ] && continue  # no E4 on CIC-DDoS2019
        run "${DS}_$E" "$PY" $A/stress_tables.py $E --results-dir "$RES" --dataset "$DS" --tex "$OUT/${DS}_$E.tex"
    done
    for E in postcheck e10 e17 threshold; do
        run "${DS}_$E" "$PY" $A/saved_tables.py $E --dataset "$DS" --tex "$OUT/${DS}_$E.tex" \
            $( [ $E = threshold ] && echo --dir "$RES/operating" || echo --dir "$RES/saved_eval" )
    done
    run "${DS}_e12" "$PY" $A/saved_tables.py e12 --dataset "$DS" --dir "$RES/latency" --tex "$OUT/${DS}_e12.tex"
    [ "$DS" = nslkdd ] && run latency "$PY" $A/saved_tables.py latency --dir "$RES/latency" --tex "$OUT/latency.tex"
    run "${DS}_cost" "$PY" $A/saved_tables.py cost --dataset "$DS" --dir "$RES/operating" \
        --tex "$OUT/${DS}_cost.tex" --fig "$OUT/${DS}_cost.pdf"
done
run nslkdd_e3_weak "$PY" $A/stress_tables.py e3 --results-dir "$RES" --dataset nslkdd --tag _weak
run nslkdd_e7_no_A3 "$PY" $A/stress_curves.py --results-dir "$RES" --exp e7 --dataset nslkdd --ltn-full no_A3
run cicddos2019_e7_no_A2 "$PY" $A/stress_curves.py --results-dir "$RES" --exp e7 --dataset cicddos2019 --ltn-full no_A2
run cicids2017_e9 "$PY" $A/stress_tables.py e9 --results-dir "$RES" --dataset cicids2017 --tex "$OUT/cicids2017_e9.tex"
run transfer "$PY" $A/stress_tables.py transfer --results-dir "$RES" --tex "$OUT/transfer.tex"
run transfer_postcheck "$PY" $A/saved_tables.py postcheck --dataset cic_transfer --dir "$RES/saved_eval"
run tuning_epoch_cap "$PY" $A/epoch_cap_report.py "$RES/tuning"
run selected_values "$PY" $A/selected_values.py --results-dir "$RES"

tar -czf tables.tar.gz -C "$RES" tables
echo "Archivio tables.tar.gz, comandi saltati: ${#FAILED[@]} ${FAILED[*]:-}"
