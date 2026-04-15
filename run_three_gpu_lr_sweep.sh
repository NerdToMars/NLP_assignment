#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU0_DEVICE="${GPU0_DEVICE:-cuda:0}"
GPU1_DEVICE="${GPU1_DEVICE:-cuda:1}"
GPU2_DEVICE="${GPU2_DEVICE:-cuda:2}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs_new_final_clean}"

# DeBERTa-based families want 2e-5 / 5e-5; BiLSTM wants 5e-4; GLiNER wants 5e-6.
# Families are split across three GPUs to roughly balance wall time.
# GPU0 takes the R-Drop/FGM/recall-boost-s42 line.
# GPU1 takes the s123 line and the core ablations that are cheap.
# GPU2 takes BiLSTM + GLiNER + the heavy "all four combined" families.

DEBERTA_LR_VALUES=(2e-5 5e-5)
BILSTM_LR_VALUES=(5e-4)
GLINER_LR_VALUES=(5e-6)

GPU0_EXPERIMENTS=(
  fgm_swa_s42
  rdrop_s42
  recall_boost_s42
  deberta_multitask
  deberta_definition
)

GPU1_EXPERIMENTS=(
  rdrop_s123
  recall_boost_s123
  deberta_baseline
  deberta_focal
  deberta_synthetic_curriculum
)

GPU2_EXPERIMENTS=(
  deberta_combined
  deberta_combined_no_synth
)

# Single-LR experiments that live on GPU2 after the deberta queue.
GPU2_BILSTM="bilstm"
GPU2_GLINER="gliner_finetune"

main() {
  mkdir -p "${OUTPUT_DIR}"

  local -a deberta_lr_args=()
  for lr in "${DEBERTA_LR_VALUES[@]}"; do deberta_lr_args+=(--lr "$lr"); done
  local -a bilstm_lr_args=()
  for lr in "${BILSTM_LR_VALUES[@]}"; do bilstm_lr_args+=(--lr "$lr"); done
  local -a gliner_lr_args=()
  for lr in "${GLINER_LR_VALUES[@]}"; do gliner_lr_args+=(--lr "$lr"); done

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local gpu0_log="${OUTPUT_DIR}/gpu0_3sweep_${timestamp}.log"
  local gpu1_log="${OUTPUT_DIR}/gpu1_3sweep_${timestamp}.log"
  local gpu2_log="${OUTPUT_DIR}/gpu2_3sweep_${timestamp}.log"
  local plan_log="${OUTPUT_DIR}/three_gpu_plan_${timestamp}.log"

  {
    echo "Three-GPU sweep plan"
    echo "Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "DeBERTa LRs: ${DEBERTA_LR_VALUES[*]}"
    echo "BiLSTM LRs: ${BILSTM_LR_VALUES[*]}"
    echo "GLiNER LRs: ${GLINER_LR_VALUES[*]}"
    echo "GPU0 ${GPU0_DEVICE}: ${GPU0_EXPERIMENTS[*]}"
    echo "GPU1 ${GPU1_DEVICE}: ${GPU1_EXPERIMENTS[*]}"
    echo "GPU2 ${GPU2_DEVICE}: ${GPU2_EXPERIMENTS[*]} + ${GPU2_BILSTM} + ${GPU2_GLINER}"
  } | tee "${plan_log}"

  (
    cd "${SCRIPT_DIR}"
    ./run_models.sh "${GPU0_EXPERIMENTS[@]}" -- \
      --device "${GPU0_DEVICE}" \
      --output-dir "${OUTPUT_DIR}" \
      "${deberta_lr_args[@]}"
  ) >"${gpu0_log}" 2>&1 &
  local pid0=$!

  (
    cd "${SCRIPT_DIR}"
    ./run_models.sh "${GPU1_EXPERIMENTS[@]}" -- \
      --device "${GPU1_DEVICE}" \
      --output-dir "${OUTPUT_DIR}" \
      "${deberta_lr_args[@]}"
  ) >"${gpu1_log}" 2>&1 &
  local pid1=$!

  (
    cd "${SCRIPT_DIR}"
    ./run_models.sh "${GPU2_EXPERIMENTS[@]}" -- \
      --device "${GPU2_DEVICE}" \
      --output-dir "${OUTPUT_DIR}" \
      "${deberta_lr_args[@]}"
    ./run_models.sh "${GPU2_BILSTM}" -- \
      --device "${GPU2_DEVICE}" \
      --output-dir "${OUTPUT_DIR}" \
      "${bilstm_lr_args[@]}"
    ./run_models.sh "${GPU2_GLINER}" -- \
      --device "${GPU2_DEVICE}" \
      --output-dir "${OUTPUT_DIR}" \
      "${gliner_lr_args[@]}"
  ) >"${gpu2_log}" 2>&1 &
  local pid2=$!

  echo "Started GPU queues: ${pid0} ${pid1} ${pid2}"
  echo "Logs:"
  echo "  ${gpu0_log}"
  echo "  ${gpu1_log}"
  echo "  ${gpu2_log}"

  local status=0
  wait "${pid0}" || { echo "GPU0 queue failed"; status=1; }
  wait "${pid1}" || { echo "GPU1 queue failed"; status=1; }
  wait "${pid2}" || { echo "GPU2 queue failed"; status=1; }
  exit "${status}"
}

main "$@"
