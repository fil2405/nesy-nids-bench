#!/usr/bin/env bash
# Esperimenti di stress (E3, E7, E9, E11, E13): configs/stress/<E>_*.yaml, seed 0-9, test valutato.
# Uso: bash server/04_stress.sh E7 [config...]
# Per ora non ci sono config: lo script lo dice invece di far finta. I requisiti mancanti
# sono in server/plan.yaml (campo needs) e in server/WORKFLOW.md.
source "$(dirname "$0")/common.sh"
EXP="${1:?esperimento: E3 | E7 | E9 | E11 | E13}"; shift
require_clean
if [ $# -gt 0 ]; then CFGS=("$@"); else mapfile -t CFGS < <(ls configs/stress/"${EXP}"_*.yaml 2>/dev/null); fi
[ ${#CFGS[@]} -gt 0 ] || die "nessun config in configs/stress/${EXP}_*.yaml (TODO: vedi server/WORKFLOW.md, fase 3)"
read -ra SEEDS <<< "$(python server/plan.py seeds)"

for CFG in "${CFGS[@]}"; do
    RUN_ID="$(python - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
assert c.get("evaluate_test") is True, "evaluate_test deve essere true nelle run di stress"
print(c["run_id"])
PY
    )" || { fail "stress $CFG: config non valido"; continue; }
    kind="$(python - "$CFG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["model"])
PY
    )"
    case "$kind" in rf|xgb|rule_only) N="$JOBS_CLASSIC" ;; *) N="$JOBS_NEURAL" ;; esac
    todo=()
    for s in "${SEEDS[@]}"; do [ -f "results/${RUN_ID}_seed${s}.json" ] || todo+=("$s"); done
    echo "$(date +%T) stress $RUN_ID: ${#todo[@]} seed da fare"
    [ ${#todo[@]} -eq 0 ] && continue
    printf '%s\n' "${todo[@]}" | xargs -P "$N" -I{} bash -c \
        'python -m nesybench.run --config "$0" --seeds {} > "$1" 2>&1' \
        "$CFG" "$LOGS/stress_${RUN_ID}_seed{}.log" || fail "stress $RUN_ID"
done
python server/plan.py audit || fail "ci sono run non citabili"
[ -s "$FAILED" ] && { echo "Qualcosa e' fallito, vedi $FAILED"; exit 1; }
echo "stress $EXP finito. Raccogli con: bash server/05_collect.sh stress_$EXP"
