#!/usr/bin/env bash
# Funzioni comuni degli script di server/. Si usa con: source "$(dirname "$0")/common.sh"
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TF_FORCE_GPU_ALLOW_GROWTH=true   # senza, il primo processo si prende tutta la VRAM
export PYTHONUNBUFFERED=1

SRV="$ROOT/results/server"             # in results/, ignorata da git: non sporca il checkout
LOGS="$SRV/logs"
FAILED="$SRV/failed.txt"
mkdir -p "$LOGS"
[ -s "$FAILED" ] && mv "$FAILED" "$FAILED.$(date +%s)"   # ogni lancio riparte pulito, lo storico resta

# processi in parallelo per tipo di braccio; il valore giusto esce da 01_scaling.sh
JOBS_NEURAL="${NESY_JOBS_NEURAL:-1}"
JOBS_CLASSIC="${NESY_JOBS_CLASSIC:-1}"

die() { echo "ERRORE: $*" >&2; exit 1; }
fail() { echo "$(date +%F_%T) $*" | tee -a "$FAILED" >&2; }

jobs_for() {   # jobs_for ARM
    case "$(python server/plan.py kind "$1")" in
        neural) echo "$JOBS_NEURAL" ;;
        *)      echo "$JOBS_CLASSIC" ;;
    esac
}

require_clean() {   # una run e' citabile solo con git_dirty falso
    git rev-parse --git-dir >/dev/null 2>&1 || die "non e' un checkout git: le run non sarebbero citabili"
    [ -z "$(git status --porcelain)" ] || { git status --short >&2; die "checkout sporco: commit o ripristina prima"; }
}

arms_or_default() {   # arms_or_default DS PHASE [ARM...]
    local ds=$1 phase=$2; shift 2
    if [ $# -gt 0 ]; then printf '%s\n' "$@"; else python server/plan.py list "$ds" "$phase"; fi
}

progress() { python server/plan.py progress > "$SRV/progress.txt" 2>&1; }
