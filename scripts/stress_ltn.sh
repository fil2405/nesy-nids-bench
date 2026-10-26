#!/usr/bin/env bash
# LTN-full stress runs (E3, E11, E13 and the weak KB), test part scored.
#   bash stress_ltn.sh <dataset> <group> [workers]
# dataset: nslkdd | cicids2017 | cicddos2019
# group: e3 | e11 | e13 | weak (nslkdd only) | all
# workers: seeds running at the same time (default 1). Finished seeds are skipped on restart.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
[ -d configs ] || cd .. || exit 1
ROOT="$PWD"

DATASETS="nslkdd cicids2017 cicddos2019"
GROUPS_OK="e3 e11 e13 weak all"
DS="${1:-}"
GROUP="${2:-}"
WORKERS="${3:-1}"
die() { echo "ERRORE: $*" >&2; exit 1; }

usage() {
    echo "Uso: bash stress_ltn.sh <dataset> <gruppo> [processi]"
    echo "Dataset: $DATASETS"; echo "Gruppi: $GROUPS_OK"; exit 2
}
case " $DATASETS " in *" $DS "*) ;; *) usage ;; esac
case " $GROUPS_OK " in *" $GROUP "*) ;; *) usage ;; esac
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || die "processi: serve un intero >= 1 (ricevuto: $WORKERS)"
PY="$(command -v python || command -v python3)" || die "python non trovato (attivare il venv)"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

RES="results"
LOGS="logs"
mkdir -p "$RES" "$LOGS"

# config selection by name: weak ones carry _weak in the run_id
FILES=()
for f in configs/stress/${DS}_*.yaml; do
    [ -f "$f" ] || continue
    name="$(basename "$f" .yaml)"
    case "$name" in
        *_weak*) kind=weak ;;
        *_e3_*) kind=e3 ;;
        *_e11_*) kind=e11 ;;
        *_e13_*) kind=e13 ;;
        *) continue ;;
    esac
    if [ "$GROUP" = all ] || [ "$GROUP" = "$kind" ]; then FILES+=("$f"); fi
done
[ "${#FILES[@]}" -gt 0 ] || die "nessuna config $DS/$GROUP in configs/stress"

# one launch at a time
if command -v flock >/dev/null; then
    exec 9> "$RES/.stress_${DS}.lock"
    flock -n 9 || die "un lancio di stress_ltn.sh su $DS e' gia' in esecuzione"
fi

echo "== controllo rapido ($DS $GROUP, ${#FILES[@]} config)"
if ! "$PY" - "$DS" "${FILES[@]}" <<'PY' 2>&1 | tee "$LOGS/stress_${DS}_${GROUP}_check.log"
import os
import sys

import yaml

import nesybench  # sets TF_USE_LEGACY_KERAS before tensorflow is imported
from nesybench.logic.edit import load_run_kb

from nesybench.data import load_dataset

dataset = sys.argv[1]
train = load_dataset(dataset, "train")
errors = []
neural = False


def fail(msg):
    errors.append(msg)
    print("KO  " + msg)


for path in sys.argv[2:]:
    cfg = yaml.safe_load(open(path))
    if cfg.get("dataset") != dataset:
        fail(f"{path}: dataset {cfg.get('dataset')!r}")
    if cfg.get("evaluate_test"):
        print(f"!!  {path}: evaluate_test gia' true")
    for key in ("kb", "c5_kb"):
        if cfg.get(key) and not os.path.exists(cfg[key]):
            fail(f"{path}: manca {cfg[key]}")
    if cfg.get("kb"):
        try:
            load_run_kb(cfg, train)
        except Exception as exc:
            fail(f"{path}: KB non applicabile: {type(exc).__name__}: {exc}")
    neural = neural or cfg.get("model") in ("mlp_ce", "ltn_sup", "ltn_full")
print(f"ok  config      {len(sys.argv) - 2} file letti")

if neural:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("ok  GPU         " + ", ".join(g.name for g in gpus))
    elif os.environ.get("NESY_ALLOW_CPU") == "1":
        print("!!  nessuna GPU, NESY_ALLOW_CPU=1: si prosegue su CPU")
    else:
        fail("nessuna GPU NVIDIA visibile a TensorFlow")

ROWS = {
    "nslkdd": {"train": 125973, "test": 22544},
    "cicids2017": {"train": 1715341},
    "cicddos2019": {"train": 340560, "test": 79083},
}
for part, rows in ROWS[dataset].items():
    n = len(train if part == "train" else load_dataset(dataset, part))
    print(f"{'ok ' if n == rows else 'KO '} dati {dataset} {part}: {n} righe")
    if n != rows:
        errors.append(f"{dataset} {part}: {n} righe, attese {rows}")

print("controllo KO" if errors else "controllo OK")
sys.exit(1 if errors else 0)
PY
then
    die "controllo fallito, le run non partono (log: $LOGS/stress_${DS}_${GROUP}_check.log)"
fi

FAILED=0
IDS=()
for CFG in "${FILES[@]}"; do
    RUN_ID="$("$PY" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CFG")"
    SEEDS="$("$PY" -c "import sys,yaml;print(*yaml.safe_load(open(sys.argv[1]))['seeds'])" "$CFG")"
    IDS+=("$RUN_ID")
    done_n="$(find "$RES" -maxdepth 1 -name "${RUN_ID}_seed*.json" | wc -l)"
    echo "== $RUN_ID (seed gia' fatti $done_n)"
    PIDS=(); PSEED=()
    for seed in $SEEDS; do
        while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
        "$PY" -m nesybench.run --config "$CFG" --evaluate-test --resume --seeds "$seed" \
            > "$LOGS/stress_${RUN_ID}_s${seed}.log" 2> "$LOGS/stress_${RUN_ID}_s${seed}.err" &
        PIDS+=($!); PSEED+=("$seed")
    done
    for k in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$k]}"; then
            echo "ERRORE: $RUN_ID seed ${PSEED[$k]} fallito, ultime righe:" >&2
            tail -n 10 "$LOGS/stress_${RUN_ID}_s${PSEED[$k]}.err" >&2
            FAILED=1
        fi
    done
done
trap 'echo; echo "Interrotto: rilanciare lo stesso comando per riprendere."; exit 130' INT TERM

echo "== verifica dei risultati"
if ! "$PY" - "$RES" "${FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

res, *configs = sys.argv[1:]
bad = []
for path in configs:
    cfg = yaml.safe_load(open(path))
    for seed in cfg["seeds"]:
        f = Path(res) / f"{cfg['run_id']}_seed{seed}.json"
        if not f.exists():
            bad.append(f"{cfg['run_id']} seed {seed}: manca {f.name}")
            continue
        rec = json.loads(f.read_text())
        if "test" not in rec["metrics"]:
            bad.append(f"{cfg['run_id']} seed {seed}: parte test non valutata")
        elif rec.get("git_dirty"):
            print(f"!!  {cfg['run_id']} seed {seed}: git_dirty")
print("\n".join("KO  " + b for b in bad) if bad else "verifica OK")
sys.exit(1 if bad else 0)
PY
then
    FAILED=1
fi

SUFFIX=""
[ "$FAILED" -eq 0 ] || SUFFIX="_incompleto"
OUT="risultati_stress_${DS}_${GROUP}${SUFFIX}.tar.gz"
LIST="$(mktemp)"
for RUN_ID in "${IDS[@]}"; do
    find "$RES" -maxdepth 1 -type f -name "${RUN_ID}_seed*.json" >> "$LIST"
    find "$LOGS" -maxdepth 1 -type f -name "stress_${RUN_ID}_*" >> "$LIST"
done
find "$LOGS" -maxdepth 1 -type f -name "stress_${DS}_${GROUP}_check.log" >> "$LIST"
tar -czf "$OUT" -T "$LIST" && rm -f "$LIST"
sha256sum "$OUT"

if [ "$FAILED" -ne 0 ]; then
    echo "Stress NON completo. Rilanciare lo stesso comando per riprendere; in ogni caso rimandare $OUT."
    exit 1
fi
rm -f "risultati_stress_${DS}_${GROUP}_incompleto.tar.gz"
echo "Stress $GROUP completato. Da rimandare: $OUT"
