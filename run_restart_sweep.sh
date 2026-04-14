#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHON_BIN="$PWD/.venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$PWD/outputs_new_final_clean"
LOG8="$OUT/gpu_cuda8_${TS}.log"
LOG0="$OUT/gpu_cuda0_${TS}.log"

mkdir -p "$OUT"

# Heavy runs on cuda:8 (48 GB free): R-Drop double-pass + recall-boost + combined.
# batch_size 4 with grad_accum 8 keeps effective batch near 32 and halves activation memory.
(./run_models.sh \
    rdrop_s42 rdrop_s123 recall_boost_s42 recall_boost_s123 \
    deberta_combined deberta_combined_no_synth \
  -- \
    --device cuda:8 --output-dir "$OUT" \
    --lr 2e-5 --lr 5e-5 \
    --batch-size 4 --gradient-accumulation-steps 8) >"$LOG8" 2>&1 &
PID8=$!

# Lighter single-pass DeBERTa variants on cuda:0 (27 GB free alongside a 20 GB user).
# Presets default to batch_size 30; drop to 16 with grad_accum 2 to stay comfortably under 27 GB.
(./run_models.sh \
    deberta_baseline deberta_focal deberta_definition \
    deberta_multitask deberta_synthetic_curriculum \
  -- \
    --device cuda:0 --output-dir "$OUT" \
    --lr 2e-5 --lr 5e-5 \
    --batch-size 16 --gradient-accumulation-steps 2) >"$LOG0" 2>&1 &
PID0=$!

echo "cuda:8 PID=$PID8 log=$LOG8"
echo "cuda:0 PID=$PID0 log=$LOG0"

wait "$PID8"
S8=$?
wait "$PID0"
S0=$?

echo "cuda:8 exit=$S8"
echo "cuda:0 exit=$S0"
