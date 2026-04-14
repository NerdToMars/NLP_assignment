#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHON_BIN="$PWD/.venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$PWD/outputs_new_final_clean"
LOG="$OUT/gpu_cuda1_light_${TS}.log"

mkdir -p "$OUT"

./run_models.sh \
    deberta_baseline deberta_focal deberta_definition \
    deberta_multitask deberta_synthetic_curriculum \
  -- \
    --device cuda:1 --output-dir "$OUT" \
    --lr 2e-5 --lr 5e-5 \
    --batch-size 8 --gradient-accumulation-steps 4 >"$LOG" 2>&1
