#!/usr/bin/env bash
# All NSL-KDD tuning arms in sequence. Finished runs are reused on restart.
set -euo pipefail
cd "$(dirname "$0")/.."
for arm in mlp_noweights mlp_weights ltn_sup ltn_full rf xgb; do
    python -m nesybench.tune --tuning "configs/tuning/nslkdd_${arm}.yaml"
done
