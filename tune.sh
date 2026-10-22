#!/usr/bin/env bash
# Tune one NSL-KDD neural arm: check, search, finalize, archive.
#   bash tune.sh <arm> [workers]
# arm: mlp_noweights | mlp_weights | ltn_sup | ltn_full. workers: processes for this arm (default 1).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
ROOT="$PWD"

ARMS="mlp_noweights mlp_weights ltn_sup ltn_full"
ARM="${1:-}"
WORKERS="${2:-1}"
die() { echo "ERRORE: $*" >&2; exit 1; }

case " $ARMS " in
    *" $ARM "*) ;;
    *) echo "Uso: bash tune.sh <braccio> [processi]"; echo "Bracci: $ARMS"; exit 2 ;;
esac
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || die "processi: serve un intero >= 1 (ricevuto: $WORKERS)"
PY="$(command -v python || command -v python3)" || die "python non trovato (attivare il venv)"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_FORCE_GPU_ALLOW_GROWTH=true   # processes of different arms share the GPU

CFG="configs/tuning/nslkdd_${ARM}.yaml"
ID="tune_nslkdd_${ARM}"
RES="results/tuning"
LOGS="logs"
[ -f "$CFG" ] || die "manca $CFG"
mkdir -p "$RES" "$LOGS"

# one run per arm at a time
if command -v flock >/dev/null; then
    exec 9> "$RES/.$ID.lock"
    flock -n 9 || die "il braccio $ARM e' gia' in esecuzione"
fi

echo "== controllo rapido ($ARM)"
if ! "$PY" - "$CFG" "$ID" <<'PY' 2>&1 | tee "$LOGS/tune_${ARM}_check.log"
import importlib
import os
import platform
import sys
from pathlib import Path

import nesybench  # sets TF_USE_LEGACY_KERAS before tensorflow is imported

errors = []


def fail(msg):
    errors.append(msg)
    print("KO  " + msg)


print(f"python      {platform.python_version()}  {platform.platform()}")
for name in ("numpy", "pandas", "sklearn", "scipy", "yaml", "pyarrow", "xgboost",
             "tensorflow", "tf_keras", "ltn"):
    try:
        module = importlib.import_module(name)
        print(f"ok  {name:11} {getattr(module, '__version__', '')}")
    except Exception as exc:
        fail(f"libreria {name}: {type(exc).__name__}: {exc}")
if errors:
    sys.exit(1)

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    names = [tf.config.experimental.get_device_details(g).get("device_name", g.name) for g in gpus]
    print("ok  GPU         " + ", ".join(names))
elif os.environ.get("NESY_ALLOW_CPU") == "1":
    print("!!  nessuna GPU, NESY_ALLOW_CPU=1: si prosegue su CPU")
else:
    fail("nessuna GPU NVIDIA visibile a TensorFlow (driver o CUDA mancanti?)")

try:
    import ltn

    x = ltn.Variable("x", np.random.rand(8, 3).astype("float32"))
    P = ltn.Predicate.MLP([3], hidden_layer_sizes=(4,))
    forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
    print(f"ok  ltn         formula di prova {float(forall(x, P(x)).tensor.numpy()):.3f}")
except Exception as exc:
    fail(f"ltn: {type(exc).__name__}: {exc}")

from nesybench.data import load_dataset

for part, rows in (("train", 125973), ("test", 22544)):
    try:
        n = len(load_dataset("nslkdd", part))
        print(f"{'ok ' if n == rows else 'KO '} dati nslkdd {part}: {n} righe")
        if n != rows:
            errors.append(f"nslkdd {part}: {n} righe, attese {rows}")
    except Exception as exc:
        fail(f"dati nslkdd {part}: {type(exc).__name__}: {exc}")

from nesybench import tune

try:
    tuning = tune.load_tuning(sys.argv[1])
    if tuning["tuning_id"] != sys.argv[2]:
        fail(f"tuning_id {tuning['tuning_id']}, atteso {sys.argv[2]}")
    tune.tune(tuning, "unused", dry_run=True, log=lambda *_: None)
    print(f"ok  config      {tuning['trials']} prove x {len(tuning['seeds'])} seed")
except Exception as exc:
    fail(f"config {sys.argv[1]}: {type(exc).__name__}: {exc}")
else:
    import yaml

    kb = yaml.safe_load(Path(tuning["base"]).read_text()).get("kb")
    if kb and not Path(kb).exists():
        fail(f"manca {kb}")

print("controllo KO" if errors else "controllo OK")
sys.exit(1 if errors else 0)
PY
then
    die "controllo fallito, il tuning non parte (log: $LOGS/tune_${ARM}_check.log)"
fi

TRIALS="$(awk '/^trials:/ {print $2}' "$CFG")"
count() { find "$RES" -maxdepth 1 -name "${ID}_trial*.json" | wc -l; }
echo "== tuning $ARM: $WORKERS processi, prove gia' fatte $(count)/$TRIALS"

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null; echo; echo "Interrotto: rilanciare lo stesso comando per riprendere."; exit 130' INT TERM
if [ "$WORKERS" -eq 1 ]; then
    "$PY" -m nesybench.tune --tuning "$CFG" > "$LOGS/tune_${ARM}_w0.log" 2> "$LOGS/tune_${ARM}_w0.err" &
    PIDS+=($!)
else
    for i in $(seq 0 $((WORKERS - 1))); do
        "$PY" -m nesybench.tune --tuning "$CFG" --worker "$i" --workers "$WORKERS" \
            > "$LOGS/tune_${ARM}_w$i.log" 2> "$LOGS/tune_${ARM}_w$i.err" &
        PIDS+=($!)
    done
fi

last=-1
while :; do
    alive=0
    for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    n="$(count)"
    if [ "$n" != "$last" ]; then echo "$(date +%T) prove finite $n/$TRIALS"; last="$n"; fi
    [ "$alive" -eq 0 ] && break
    sleep 20
done

FAILED=0
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then
        echo "ERRORE: processo $k fallito, ultime righe:" >&2
        tail -n 15 "$LOGS/tune_${ARM}_w$k.err" >&2
        FAILED=1
    fi
done

if [ "$FAILED" -eq 0 ]; then
    if "$PY" -m nesybench.tune --tuning "$CFG" --finalize \
        > "$LOGS/tune_${ARM}_finalize.log" 2> "$LOGS/tune_${ARM}_finalize.err"; then
        tail -n 1 "$LOGS/tune_${ARM}_finalize.log"
    else
        echo "ERRORE: finalize fallito, ultime righe:" >&2
        tail -n 15 "$LOGS/tune_${ARM}_finalize.err" >&2
        FAILED=1
    fi
fi

# archive of this arm only
OUT="risultati_${ARM}.tar.gz"
if [ "$FAILED" -eq 0 ]; then
    rm -f "risultati_${ARM}_incompleto.tar.gz"
else
    OUT="risultati_${ARM}_incompleto.tar.gz"
fi
LIST="$(mktemp)"
find "$RES" -maxdepth 1 -type f \( -name "$ID.json" -o -name "${ID}_*" \) > "$LIST"
find "$LOGS" -maxdepth 1 -type f -name "tune_${ARM}_*" >> "$LIST"
tar -czf "$OUT" -T "$LIST" && rm -f "$LIST"
sha256sum "$OUT"

if [ "$FAILED" -ne 0 ]; then
    echo "Tuning NON completato. Rilanciare lo stesso comando per riprendere; in ogni caso rimandare $OUT."
    exit 1
fi
echo "Tuning $ARM completato. Da rimandare: $OUT"
