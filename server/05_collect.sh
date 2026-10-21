#!/usr/bin/env bash
# Impacchetta quello che torna indietro: results/ (JSON, best.yaml, final_configs), log, progress.
# Uso: bash server/05_collect.sh NOME_FASE      -> results/server/out/NOME_FASE_<data>.tar.gz
# Non include pesi dei modelli ne' dati: solo risultati e log.
source "$(dirname "$0")/common.sh"
TAG="${1:?nome fase, per esempio tune_cicids2017}"
OUT="$SRV/out"; mkdir -p "$OUT"
progress
ARCH="$OUT/${TAG}_$(date +%Y%m%d_%H%M).tar.gz"
{
    echo "commit: $(git rev-parse HEAD 2>/dev/null)  dirty: $([ -z "$(git status --porcelain 2>/dev/null)" ] && echo no || echo SI)"
    echo "macchina: $(uname -a)"; nvidia-smi -L 2>/dev/null
    python server/plan.py audit | tail -1
} > "$SRV/MANIFEST.txt"
tar -czf "$ARCH" --exclude='results/server/out' --exclude='results/server/smoke' --exclude='results/server/scaling/*/P*/results' \
    results server/plan.yaml 2>/dev/null
sha256sum "$ARCH" | tee "$ARCH.sha256"
ls -lh "$ARCH"
