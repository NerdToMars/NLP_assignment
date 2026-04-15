#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_ROOT="${SCRIPT_DIR}/outputs_reddit_backbones"
GPU0_DEVICE="cuda:0"
GPU1_DEVICE="cuda:1"
DRY_RUN=0
INCLUDE_GATED=0
INCLUDE_EXTRA_PIPELINES=0
SKIP_EXISTING=0

MODEL_LR_VALUES=(5e-4 2e-5 5e-5)
CUSTOM_MODEL_LR_VALUES=0

DEFAULT_BACKBONES=(
  socbert
  stress_roberta
)

GATED_BACKBONES=(
  mental_bert
  mental_roberta
)

GPU0_EXPERIMENTS=(
  rdrop_s42
  fgm_swa_s42
  recall_boost_s42
  deberta_multitask
  deberta_combined
  deberta_definition
)

GPU1_EXPERIMENTS=(
  rdrop_s123
  recall_boost_s123
  deberta_baseline
  deberta_focal
  deberta_synthetic_curriculum
  deberta_combined_no_synth
  hierarchical_deberta
)

EXTRA_GPU0_EXPERIMENTS=(
  two_step_impact_pipeline
)

EXTRA_GPU1_EXPERIMENTS=(
  sentence_token_hierarchy
)

usage() {
  cat <<'EOF'
Usage:
  ./run_reddit_backbone_sweeps.sh
  ./run_reddit_backbone_sweeps.sh socbert stress_roberta
  ./run_reddit_backbone_sweeps.sh --include-gated
  ./run_reddit_backbone_sweeps.sh --include-extra-pipelines
  ./run_reddit_backbone_sweeps.sh --skip-existing
  ./run_reddit_backbone_sweeps.sh --output-root /path/to/output_root
  ./run_reddit_backbone_sweeps.sh --lr 5e-4 --lr 2e-5 --lr 5e-5
  ./run_reddit_backbone_sweeps.sh --gpu0-device cuda:0 --gpu1-device cuda:1
  ./run_reddit_backbone_sweeps.sh -- --enable-preprocessing

Behavior:
  - Runs only the transformer / DeBERTa-style experiment family on each selected backbone.
  - --include-extra-pipelines also adds two_step_impact_pipeline and sentence_token_hierarchy.
  - --skip-existing skips runs whose expected best checkpoint / result artifacts already exist.
  - Uses two sequential queues in parallel, one per GPU.
  - Writes each backbone's artifacts into its own output directory.
  - Defaults to the non-gated backbones: socbert stress_roberta
  - --include-gated also adds: mental_bert mental_roberta
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

run_one_backbone() {
  local backbone="$1"
  shift
  local -a forwarded_args=("$@")
  local -a model_lr_args=()
  local -a gpu0_experiments=("${GPU0_EXPERIMENTS[@]}")
  local -a gpu1_experiments=("${GPU1_EXPERIMENTS[@]}")
  local timestamp
  local output_dir
  local gpu0_log
  local gpu1_log
  local plan_log
  local pid0
  local pid1
  local status=0

  output_dir="${OUTPUT_ROOT}/${backbone}"
  mkdir -p "${output_dir}"

  if [[ ${INCLUDE_EXTRA_PIPELINES} -eq 1 ]]; then
    gpu0_experiments+=("${EXTRA_GPU0_EXPERIMENTS[@]}")
    gpu1_experiments+=("${EXTRA_GPU1_EXPERIMENTS[@]}")
  fi

  for lr in "${MODEL_LR_VALUES[@]}"; do
    model_lr_args+=(--lr "$lr")
  done

  if [[ ${DRY_RUN} -eq 1 ]]; then
    forwarded_args+=(--dry-run)
  fi

  if [[ ${SKIP_EXISTING} -eq 1 ]]; then
    forwarded_args+=(--skip-existing)
  fi

  timestamp="$(date +%Y%m%d_%H%M%S)"
  gpu0_log="${output_dir}/gpu0_${backbone}_${timestamp}.log"
  gpu1_log="${output_dir}/gpu1_${backbone}_${timestamp}.log"
  plan_log="${output_dir}/backbone_plan_${timestamp}.log"

  echo "================================================================"
  echo "Running backbone: ${backbone}"
  echo "  Output dir: ${output_dir}"
  echo "  GPU 0 device: ${GPU0_DEVICE}"
  echo "  GPU 1 device: ${GPU1_DEVICE}"
  echo "  LR values: ${MODEL_LR_VALUES[*]}"
  echo "  GPU 0 experiments: ${gpu0_experiments[*]}"
  echo "  GPU 1 experiments: ${gpu1_experiments[*]}"
  echo "  GPU 0 log: ${gpu0_log}"
  echo "  GPU 1 log: ${gpu1_log}"
  echo "  Plan log: ${plan_log}"
  echo "================================================================"

  {
    echo "Reddit backbone sweep plan"
    echo "Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Backbone: ${backbone}"
    echo "Output dir: ${output_dir}"
    echo "GPU 0 device: ${GPU0_DEVICE}"
    echo "GPU 1 device: ${GPU1_DEVICE}"
    echo "LR values: ${MODEL_LR_VALUES[*]}"
    echo
    print_queue_plan "GPU 0" "${GPU0_DEVICE}" "${gpu0_experiments[@]}"
    echo
    print_queue_plan "GPU 1" "${GPU1_DEVICE}" "${gpu1_experiments[@]}"
  } | tee "${plan_log}"

  (
    cd "${SCRIPT_DIR}"
    ./run_models.sh "${gpu0_experiments[@]}" -- \
      --device "${GPU0_DEVICE}" \
      --output-dir "${output_dir}" \
      --backbone "${backbone}" \
      "${model_lr_args[@]}" \
      "${forwarded_args[@]}"
  ) >"${gpu0_log}" 2>&1 &
  pid0=$!

  (
    cd "${SCRIPT_DIR}"
    ./run_models.sh "${gpu1_experiments[@]}" -- \
      --device "${GPU1_DEVICE}" \
      --output-dir "${output_dir}" \
      --backbone "${backbone}" \
      "${model_lr_args[@]}" \
      "${forwarded_args[@]}"
  ) >"${gpu1_log}" 2>&1 &
  pid1=$!

  echo "Started GPU queues for ${backbone} with PIDs ${pid0} and ${pid1}"
  echo "Monitor progress with:"
  echo "  tail -f ${gpu0_log}"
  echo "  tail -f ${gpu1_log}"

  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?

  if [[ ${status} -ne 0 ]]; then
    echo "Backbone sweep failed for ${backbone}. Check:"
    echo "  ${gpu0_log}"
    echo "  ${gpu1_log}"
    return "${status}"
  fi

  echo "Backbone sweep completed for ${backbone}."
  echo
}

main() {
  local -a backbones=()
  local -a forwarded_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --output-root)
        OUTPUT_ROOT="$2"
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
      --include-gated)
        INCLUDE_GATED=1
        shift
        ;;
      --include-extra-pipelines)
        INCLUDE_EXTRA_PIPELINES=1
        shift
        ;;
      --skip-existing)
        SKIP_EXISTING=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --lr)
        if [[ ${CUSTOM_MODEL_LR_VALUES} -eq 0 ]]; then
          MODEL_LR_VALUES=()
          CUSTOM_MODEL_LR_VALUES=1
        fi
        MODEL_LR_VALUES+=("$2")
        shift 2
        ;;
      --)
        shift
        forwarded_args=("$@")
        break
        ;;
      *)
        backbones+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#MODEL_LR_VALUES[@]} -eq 0 ]]; then
    echo "At least one --lr value is required." >&2
    exit 1
  fi

  if [[ ${#backbones[@]} -eq 0 ]]; then
    backbones=("${DEFAULT_BACKBONES[@]}")
    if [[ ${INCLUDE_GATED} -eq 1 ]]; then
      backbones+=("${GATED_BACKBONES[@]}")
    fi
  fi

  mkdir -p "${OUTPUT_ROOT}"

  echo "Reddit / mental-health backbone sweep"
  echo "  Output root: ${OUTPUT_ROOT}"
  echo "  GPU 0 device: ${GPU0_DEVICE}"
  echo "  GPU 1 device: ${GPU1_DEVICE}"
  echo "  Backbones: ${backbones[*]}"
  echo "  LR values: ${MODEL_LR_VALUES[*]}"
  echo

  for backbone in "${backbones[@]}"; do
    run_one_backbone "${backbone}" "${forwarded_args[@]}"
  done
}

main "$@"
