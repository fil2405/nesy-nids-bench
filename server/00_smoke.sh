#!/usr/bin/env bash
# Controllo rapido: GPU, librerie, un modello per tipo su NSL-KDD, commit leggibile nel JSON.
# Uso: bash server/00_smoke.sh      (SMOKE_TESTS=1 per lanciare anche pytest)
source "$(dirname "$0")/common.sh"
OUT="$SRV/smoke"; rm -rf "$OUT"; mkdir -p "$OUT"

echo "== ambiente"; python scripts/check_env.py || die "check_env.py fallito"
if command -v nvidia-smi >/dev/null && nvidia-smi -L; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "nvidia-smi non disponibile"
fi
echo "cpu: $(nproc)  ram: $(free -g 2>/dev/null | awk '/Mem:/{print $2" GB"}')  disco libero: $(df -h . | awk 'NR==2{print $4}')"
[ "${SMOKE_TESTS:-0}" = 1 ] && { python -m pytest -q || die "pytest fallito"; }

echo "== un modello per tipo, NSL-KDD, seed 0, solo validation"
for cfg in nslkdd_mlp_noweights nslkdd_ltn_sup nslkdd_ltn_full nslkdd_rf nslkdd_xgb nslkdd_rules; do
    echo "$(date +%T) $cfg"
    python -m nesybench.run --config "configs/$cfg.yaml" --seeds 0 --results-dir "$OUT" \
        > "$LOGS/smoke_$cfg.log" 2>&1 || fail "smoke $cfg (vedi $LOGS/smoke_$cfg.log)"
done

echo "== controllo dei risultati"
python - "$OUT" <<'PY' || die "smoke: risultati non validi"
import glob, json, math, sys
files = sorted(glob.glob(sys.argv[1] + "/*_seed0.json"))
bad = 0
for f in files:
    r = json.load(open(f))
    f1 = r["metrics"]["val"]["macro_f1"]
    ok = math.isfinite(f1) and r["git_commit"] != "unknown" and r["git_dirty"] is False
    bad += not ok
    print(f"{'ok ' if ok else 'KO '} {r['run_id']:34} val macro-F1 {f1:.4f}  fit {r['timings']['fit_s']:.1f} s  "
          f"commit {r['git_commit']} dirty {r['git_dirty']}")
print(f"{len(files)} file, {bad} da controllare")
sys.exit(1 if bad or len(files) < 6 else 0)
PY
echo "smoke OK"
