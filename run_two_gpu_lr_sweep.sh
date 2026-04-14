#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU0_DEVICE="cuda:0"
GPU1_DEVICE="cuda:1"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
DRY_RUN=0
INCLUDE_EXTRA_PIPELINES=0

MODEL_LR_VALUES=(5e-4 2e-5 5e-5)
GLINER_LR_VALUES=(5e-6 1e-5 2e-5)
CUSTOM_MODEL_LR_VALUES=0
CUSTOM_GLINER_LR_VALUES=0

# Split to keep the longest-running jobs distributed across both GPUs.
GPU0_EXPERIMENTS=(
  rdrop_s42
  fgm_swa_s42
  recall_boost_s42
  deberta_multitask
  deberta_combined
  deberta_definition
)
GPU0_GLINER_EXPERIMENT="gliner_finetune"

GPU1_EXPERIMENTS=(
  rdrop_s123
  recall_boost_s123
  bilstm
  deberta_baseline
  deberta_focal
  deberta_synthetic_curriculum
  deberta_combined_no_synth
)

EXTRA_GPU0_EXPERIMENTS=(
  two_step_impact_pipeline
)

EXTRA_GPU1_EXPERIMENTS=(
  hierarchical_deberta
  sentence_token_hierarchy
)

usage() {
  cat <<'EOF'
Usage:
  ./run_two_gpu_lr_sweep.sh
  ./run_two_gpu_lr_sweep.sh --dry-run
  ./run_two_gpu_lr_sweep.sh --include-extra-pipelines
  ./run_two_gpu_lr_sweep.sh --output-dir /path/to/outputs
  ./run_two_gpu_lr_sweep.sh --gpu0-device cuda:0 --gpu1-device cuda:1
  ./run_two_gpu_lr_sweep.sh --lr 5e-4 --lr 2e-5 --lr 5e-5
  ./run_two_gpu_lr_sweep.sh --gliner-lr 5e-6 --gliner-lr 1e-5 --gliner-lr 2e-5
  ./run_two_gpu_lr_sweep.sh --output-dir /path/to/outputs -- --backbone socbert
  ./run_two_gpu_lr_sweep.sh -- --early-stopping-patience 5 --early-stopping-min-delta 0.001
  ./run_two_gpu_lr_sweep.sh -- --epochs 20

Behavior:
  - Launches two sequential experiment queues in parallel, one per GPU.
  - Applies one LR sweep to the main trainable presets and a separate safer LR sweep to GLiNER fine-tuning.
  - --include-extra-pipelines also adds hierarchical_deberta, sentence_token_hierarchy, and two_step_impact_pipeline.
  - Writes launcher logs to the output directory.
  - Excludes evaluation-only presets such as gliner, gliner_inference, and ensemble because they do not use optimizer learning rates.
EOF
}

print_queue_plan() {
  local label="$1"
  local device="$2"
  shift 2
  local -a experiments=("$@")
  local run_index=0

  echo "${label} plan on ${device}:"
  for experiment in "${experiments[@]}"; do
    for lr in "${MODEL_LR_VALUES[@]}"; do
      run_index=$((run_index + 1))
      echo "  [${run_index}] ${experiment} (lr=${lr})"
    done
  done
}

print_gliner_plan() {
  local label="$1"
  local device="$2"
  local experiment="$3"
  local run_index="$4"

  echo "${label} GLiNER plan on ${device}:"
  for lr in "${GLINER_LR_VALUES[@]}"; do
    run_index=$((run_index + 1))
    echo "  [${run_index}] ${experiment} (lr=${lr})"
  done
}

main() {
  local -a forwarded_args=()
  local -a model_lr_args=()
  local -a gliner_lr_args=()
  local -a output_args=()
  local forwarded_args_text=""
  local timestamp
  local gpu0_log
  local gpu1_log
  local plan_log
  local pid0
  local pid1
  local status=0
  local gpu0_main_runs=0
  local gpu0_total_runs=0
  local gpu1_total_runs=0
  local -a gpu0_experiments=("${GPU0_EXPERIMENTS[@]}")
  local -a gpu1_experiments=("${GPU1_EXPERIMENTS[@]}")

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --include-extra-pipelines)
        INCLUDE_EXTRA_PIPELINES=1
        shift
        ;;
      --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --gpu0-device)
        GPU0_DEVICE="$2"
        shift 2
        ;;
      --gpu1-device)
        GPU1_DEVICE="$2"
        shift 2
        ;;
      --lr)
        if [[ ${CUSTOM_MODEL_LR_VALUES} -eq 0 ]]; then
          MODEL_LR_VALUES=()
          CUSTOM_MODEL_LR_VALUES=1
        fi
        MODEL_LR_VALUES+=("$2")
        shift 2
        ;;
      --gliner-lr)
        if [[ ${CUSTOM_GLINER_LR_VALUES} -eq 0 ]]; then
          GLINER_LR_VALUES=()
          CUSTOM_GLINER_LR_VALUES=1
        fi
        GLINER_LR_VALUES+=("$2")
        shift 2
        ;;
      --)
        shift
        forwarded_args=("$@")
        break
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  if [[ ${#MODEL_LR_VALUES[@]} -eq 0 ]]; then
    echo "At least one model --lr value is required." >&2
    exit 1
  fi

  if [[ ${#GLINER_LR_VALUES[@]} -eq 0 ]]; then
    echo "At least one --gliner-lr value is required." >&2
    exit 1
  fi

  if [[ ${INCLUDE_EXTRA_PIPELINES} -eq 1 ]]; then
    gpu0_experiments+=("${EXTRA_GPU0_EXPERIMENTS[@]}")
    gpu1_experiments+=("${EXTRA_GPU1_EXPERIMENTS[@]}")
  fi

  mkdir -p "${OUTPUT_DIR}"

  for lr in "${MODEL_LR_VALUES[@]}"; do
    model_lr_args+=(--lr "$lr")
  done

  for lr in "${GLINER_LR_VALUES[@]}"; do
    gliner_lr_args+=(--lr "$lr")
  done

  if [[ -n "${OUTPUT_DIR}" ]]; then
    output_args=(--output-dir "$OUTPUT_DIR")
  fi

  if [[ ${DRY_RUN} -eq 1 ]]; then
    forwarded_args+=(--dry-run)
  fi

  if [[ ${#forwarded_args[@]} -gt 0 ]]; then
    printf -v forwarded_args_text '%q ' "${forwarded_args[@]}"
    forwarded_args_text="${forwarded_args_text% }"
  fi

  timestamp="$(date +%Y%m%d_%H%M%S)"
  gpu0_log="${OUTPUT_DIR}/gpu0_lr_sweep_${timestamp}.log"
  gpu1_log="${OUTPUT_DIR}/gpu1_lr_sweep_${timestamp}.log"
  plan_log="${OUTPUT_DIR}/two_gpu_lr_sweep_plan_${timestamp}.log"
  gpu0_main_runs=$((${#gpu0_experiments[@]} * ${#MODEL_LR_VALUES[@]}))
  gpu0_total_runs=$((gpu0_main_runs + ${#GLINER_LR_VALUES[@]}))
  gpu1_total_runs=$((${#gpu1_experiments[@]} * ${#MODEL_LR_VALUES[@]}))

  echo "Launching two-GPU LR sweep"
  echo "  Main model LR values: ${MODEL_LR_VALUES[*]}"
  echo "  GLiNER LR values: ${GLINER_LR_VALUES[*]}"
  echo "  GPU 0 device: ${GPU0_DEVICE}"
  echo "  GPU 1 device: ${GPU1_DEVICE}"
  echo "  GPU 0 experiments: ${gpu0_experiments[*]}"
  echo "  GPU 0 GLiNER experiment: ${GPU0_GLINER_EXPERIMENT}"
  echo "  GPU 1 experiments: ${gpu1_experiments[*]}"
  echo "  GPU 0 log: ${gpu0_log}"
  echo "  GPU 1 log: ${gpu1_log}"
  echo "  Plan log: ${plan_log}"
  echo "  Evaluation-only presets skipped: gliner gliner_inference ensemble"
  if [[ -n "${forwarded_args_text}" ]]; then
    echo "  Forwarded args: ${forwarded_args_text}"
  fi

  {
    echo "Two-GPU LR sweep plan"
    echo "Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "Main model LR values: ${MODEL_LR_VALUES[*]}"
    echo "GLiNER LR values: ${GLINER_LR_VALUES[*]}"
    echo "GPU 0 device: ${GPU0_DEVICE}"
    echo "GPU 1 device: ${GPU1_DEVICE}"
    if [[ -n "${forwarded_args_text}" ]]; then
      echo "Forwarded args: ${forwarded_args_text}"
    fi
    echo
    echo "GPU 0 total planned runs: ${gpu0_total_runs}"
    print_queue_plan "GPU 0" "${GPU0_DEVICE}" "${gpu0_experiments[@]}"
    print_gliner_plan "GPU 0" "${GPU0_DEVICE}" "${GPU0_GLINER_EXPERIMENT}" "${gpu0_main_runs}"
    echo
    echo "GPU 1 total planned runs: ${gpu1_total_runs}"
    print_queue_plan "GPU 1" "${GPU1_DEVICE}" "${gpu1_experiments[@]}"
  } | tee "${plan_log}"

  (
    cd "${SCRIPT_DIR}"
    ./run_models.sh "${gpu0_experiments[@]}" -- \
      --device "${GPU0_DEVICE}" \
      "${output_args[@]}" \
      "${model_lr_args[@]}" \
      "${forwarded_args[@]}"

    ./run_models.sh "${GPU0_GLINER_EXPERIMENT}" -- \
      --device "${GPU0_DEVICE}" \
      "${output_args[@]}" \
      "${gliner_lr_args[@]}" \
      "${forwarded_args[@]}"
  ) >"${gpu0_log}" 2>&1 &
  pid0=$!

  (
    cd "${SCRIPT_DIR}"
    ./run_models.sh "${gpu1_experiments[@]}" -- \
      --device "${GPU1_DEVICE}" \
      "${output_args[@]}" \
      "${model_lr_args[@]}" \
      "${forwarded_args[@]}"
  ) >"${gpu1_log}" 2>&1 &
  pid1=$!

  echo "Started GPU queues with PIDs ${pid0} and ${pid1}"
  echo "Monitor progress with:"
  echo "  tail -f ${gpu0_log}"
  echo "  tail -f ${gpu1_log}"

  if ! wait "${pid0}"; then
    echo "GPU 0 queue failed. See ${gpu0_log}" >&2
    status=1
  fi

  if ! wait "${pid1}"; then
    echo "GPU 1 queue failed. See ${gpu1_log}" >&2
    status=1
  fi

  if [[ ${status} -ne 0 ]]; then
    exit "${status}"
  fi

  echo "Two-GPU LR sweep completed successfully."
}

main "$@"
