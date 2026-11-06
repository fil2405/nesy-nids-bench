#!/usr/bin/env bash
# E10, E12, E17 on the saved final models: post-check, perturbations, latency, operating point and cost curves.
# Inference only.
#   bash saved_runs.sh <dataset> [models_dir]
# dataset: nslkdd | cicids2017 | cicddos2019; models_dir: folders <run_id>_seed<k> (default results/models)
# The configs are taken from the final run files results/<run_id>_seed0.json.
# Reports: results/saved_eval/<run_id>_test_e10.json, results/operating/<run_id>.json,
# results/latency/<run_id>.json (seed 0, CPU).
# Finished reports are skipped on restart.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
[ -d configs ] || cd .. || exit 1
ROOT="$PWD"

DS="${1:-}"
MODELS="${2:-results/models}"
case " nslkdd cicids2017 cicddos2019 " in *" $DS "*) ;; *) echo "Uso: bash saved_runs.sh <nslkdd|cicids2017|cicddos2019> [cartella modelli]"; exit 2 ;; esac
[ -d "$MODELS" ] || { echo "manca la cartella dei modelli $MODELS"; exit 1; }
PY="$(command -v python || command -v python3)" || { echo "python non trovato"; exit 1; }
PERTURB="${PERTURB:-noise:0.1 noise:0.25 noise:0.5 evade:0.1}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p results/saved_eval results/operating results/latency results/configs_finals logs

FAILED=0
for ARM in mlp_noweights mlp_weights cnn ltn_sup ltn_full rf xgb rules; do
    RUN_ID="${DS}_${ARM}_tuned"; [ "$ARM" = rules ] && RUN_ID="${DS}_rules"
    SRC="results/${RUN_ID}_seed0.json"
    CFG="results/configs_finals/${RUN_ID}.yaml"
    if [ ! -f "$SRC" ]; then echo "SALTATO $RUN_ID: manca $SRC"; FAILED=1; continue; fi
    if [ ! -f "$MODELS/${RUN_ID}_seed0/meta.json" ]; then echo "SALTATO $RUN_ID: mancano i modelli in $MODELS"; FAILED=1; continue; fi
    "$PY" -c "import json,sys,yaml; yaml.safe_dump(json.load(open(sys.argv[1]))['config'], open(sys.argv[2], 'w'), sort_keys=False)" "$SRC" "$CFG"

    OUT="results/saved_eval/${RUN_ID}_test_e10.json"
    if [ -f "$OUT" ]; then
        echo "== $RUN_ID: post-check e perturbazioni gia' fatti"
    else
        echo "== $RUN_ID: post-check e perturbazioni ($(date +%T))"
        # shellcheck disable=SC2086
        if ! "$PY" -m nesybench.saved_eval --config "$CFG" --models "$MODELS" --perturb $PERTURB \
                --out "$OUT" > "logs/saved_${RUN_ID}.log" 2> "logs/saved_${RUN_ID}.err"; then
            echo "ERRORE $RUN_ID"; tail -n 5 "logs/saved_${RUN_ID}.err"; FAILED=1; rm -f "$OUT"
        else
            grep -E "^seed|mancanti" "logs/saved_${RUN_ID}.log" | head -n 3
        fi
    fi

    OP="results/operating/${RUN_ID}.json"
    if [ ! -f "$OP" ]; then
        echo "== $RUN_ID: soglia sul validation e curve di costo ($(date +%T))"
        if ! "$PY" -m nesybench.operating --config "$CFG" --models "$MODELS" --out "$OP" \
                > "logs/operating_${RUN_ID}.log" 2> "logs/operating_${RUN_ID}.err"; then
            echo "ERRORE soglia $RUN_ID"; tail -n 5 "logs/operating_${RUN_ID}.err"; FAILED=1; rm -f "$OP"
        fi
    fi

    [ "$ARM" = rules ] && continue  # latency reads no rule_only model
    LAT="results/latency/${RUN_ID}.json"
    if [ ! -f "$LAT" ]; then
        echo "== $RUN_ID: latenza seed 0 ($(date +%T))"
        if ! "$PY" -m nesybench.latency --model-dir "$MODELS/${RUN_ID}_seed0" --config "$CFG" --seed 0 \
                --postcheck --device cpu --out "$LAT" > "logs/latency_${RUN_ID}.log" 2> "logs/latency_${RUN_ID}.err"; then
            echo "ERRORE latenza $RUN_ID"; tail -n 5 "logs/latency_${RUN_ID}.err"; FAILED=1; rm -f "$LAT"
        fi
    fi
done

[ "$FAILED" -eq 0 ] && echo "Completato $DS." || { echo "Con errori o salti: rilanciare dopo aver sistemato."; exit 1; }
