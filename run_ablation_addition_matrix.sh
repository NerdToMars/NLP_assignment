#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
else
  echo "No Python interpreter found. Set PYTHON_BIN to override." >&2
  exit 1
fi

OUTPUT_DIR="${SCRIPT_DIR}/outputs_ablation_matrix"
GPU0_DEVICE="cuda:0"
GPU1_DEVICE="cuda:1"
DRY_RUN=0
INCLUDE_ADDITIONS=1

BACKBONE=""
MODEL_NAME=""
MODEL_SUFFIX=""

MODEL_LR_VALUES=(5e-4 2e-5 5e-5)
GLINER_LR_VALUES=(5e-6 1e-5 2e-5)
CUSTOM_MODEL_LR_VALUES=0
CUSTOM_GLINER_LR_VALUES=0

usage() {
  cat <<'EOF'
Usage:
  ./run_ablation_addition_matrix.sh
  ./run_ablation_addition_matrix.sh --dry-run
  ./run_ablation_addition_matrix.sh --output-dir /path/to/outputs
  ./run_ablation_addition_matrix.sh --backbone socbert
  ./run_ablation_addition_matrix.sh --model-name microsoft/deberta-v3-base
  ./run_ablation_addition_matrix.sh --lr 5e-4 --lr 2e-5 --lr 5e-5
  ./run_ablation_addition_matrix.sh --gliner-lr 5e-6 --gliner-lr 1e-5 --gliner-lr 2e-5
  ./run_ablation_addition_matrix.sh --combinations-only
  ./run_ablation_addition_matrix.sh -- --enable-preprocessing

Behavior:
  - Runs all 16 combinations of the DeBERTa ablation toggles:
      focal, definition prompting, multitask, synthetic+curriculum
  - By default also runs the transformer-based architecture additions:
      hierarchical_deberta, two_step_impact_pipeline, sentence_token_hierarchy
  - When no backbone/model override is provided, it also runs span_nested_gliner.
  - Splits the queued jobs across two GPUs and gives each job its own LR sweep.
EOF
}

sanitize_suffix() {
  local raw="$1"
  raw="${raw##*/}"
  raw="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  raw="$(printf '%s' "${raw}" | tr '/.:-' '_' | tr -cs '[:alnum:]_' '_')"
  raw="${raw##_}"
  raw="${raw%%_}"
  printf '%s' "${raw}"
}

append_spec() {
  local spec="$1"
  if (( ${#ALL_SPECS[@]} % 2 == 0 )); then
    GPU0_SPECS+=("${spec}")
  else
    GPU1_SPECS+=("${spec}")
  fi
  ALL_SPECS+=("${spec}")
}

build_combo_name() {
  local mask="$1"
  local -a parts=()

  (( mask & 1 )) && parts+=(focal)
  (( mask & 2 )) && parts+=(definition)
  (( mask & 4 )) && parts+=(multitask)
  (( mask & 8 )) && parts+=(synthcurr)

  if [[ ${#parts[@]} -eq 0 ]]; then
    parts=(base)
  fi

  local joined
  joined="$(IFS=_; echo "${parts[*]}")"
  printf 'matrix_%s%s' "${joined}" "${MODEL_SUFFIX}"
}

build_spec_plan_line() {
  local spec="$1"
  local type name focal definition multitask synthcurr preset
  IFS='|' read -r type name focal definition multitask synthcurr preset <<< "${spec}"

  if [[ "${type}" == "combo" ]]; then
    local toggles=()
    [[ "${focal}" == "1" ]] && toggles+=(focal)
    [[ "${definition}" == "1" ]] && toggles+=(definition)
    [[ "${multitask}" == "1" ]] && toggles+=(multitask)
    [[ "${synthcurr}" == "1" ]] && toggles+=(synthetic+curriculum)
    if [[ ${#toggles[@]} -eq 0 ]]; then
      toggles=(baseline)
    fi
    printf '%s => deberta_baseline [%s]' "${name}" "$(IFS=', '; echo "${toggles[*]}")"
    return
  fi

  printf '%s => %s' "${name}" "${preset}"
}

run_spec() {
  local spec="$1"
  local device="$2"
  local type name focal definition multitask synthcurr preset
  local -a cmd=()
  local -a lr_args=()

  IFS='|' read -r type name focal definition multitask synthcurr preset <<< "${spec}"

  cmd=(
    "${PYTHON_CMD[@]}"
    run_experiments.py
    run
    --device "${device}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${name}"
  )

  if [[ "${type}" == "combo" ]]; then
    cmd+=(--experiment deberta_baseline)
    for lr in "${MODEL_LR_VALUES[@]}"; do
      lr_args+=(--lr "${lr}")
    done
    cmd+=("${lr_args[@]}")
    [[ "${focal}" == "1" ]] && cmd+=(--use-focal-loss)
    [[ "${definition}" == "1" ]] && cmd+=(--definition-prompting)
    [[ "${multitask}" == "1" ]] && cmd+=(--use-multitask)
    if [[ "${synthcurr}" == "1" ]]; then
      cmd+=(--use-synthetic --use-curriculum)
    fi
    if [[ -n "${BACKBONE}" ]]; then
      cmd+=(--backbone "${BACKBONE}")
    elif [[ -n "${MODEL_NAME}" ]]; then
      cmd+=(--model-name "${MODEL_NAME}")
    fi
  else
    cmd+=(--experiment "${preset}")
    if [[ "${preset}" == "span_nested_gliner" ]]; then
      for lr in "${GLINER_LR_VALUES[@]}"; do
        lr_args+=(--lr "${lr}")
      done
    else
      for lr in "${MODEL_LR_VALUES[@]}"; do
        lr_args+=(--lr "${lr}")
      done
      if [[ -n "${BACKBONE}" ]]; then
        cmd+=(--backbone "${BACKBONE}")
      elif [[ -n "${MODEL_NAME}" ]]; then
        cmd+=(--model-name "${MODEL_NAME}")
      fi
    fi
    cmd+=("${lr_args[@]}")
  fi

  if [[ ${DRY_RUN} -eq 1 ]]; then
    cmd+=(--dry-run)
  fi
  if [[ ${#FORWARDED_ARGS[@]} -gt 0 ]]; then
    cmd+=("${FORWARDED_ARGS[@]}")
  fi

  "${cmd[@]}"
}

run_queue() {
  local queue_name="$1"
  local device="$2"
  shift 2
  local -a specs=("$@")

  for spec in "${specs[@]}"; do
    echo
    echo "================================================================"
    echo "${queue_name} on ${device}: $(build_spec_plan_line "${spec}")"
    echo "================================================================"
    run_spec "${spec}" "${device}"
  done
}

main() {
  FORWARDED_ARGS=()
  ALL_SPECS=()
  GPU0_SPECS=()
  GPU1_SPECS=()

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
      --backbone)
        BACKBONE="$2"
        shift 2
        ;;
      --model-name)
        MODEL_NAME="$2"
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
      --combinations-only)
        INCLUDE_ADDITIONS=0
        shift
        ;;
      --)
        shift
        FORWARDED_ARGS=("$@")
        break
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  if [[ -n "${BACKBONE}" && -n "${MODEL_NAME}" ]]; then
    echo "Use either --backbone or --model-name, not both." >&2
    exit 1
  fi

  if [[ ${#MODEL_LR_VALUES[@]} -eq 0 ]]; then
    echo "At least one --lr value is required." >&2
    exit 1
  fi

  if [[ ${#GLINER_LR_VALUES[@]} -eq 0 ]]; then
    echo "At least one --gliner-lr value is required." >&2
    exit 1
  fi

  if [[ -n "${BACKBONE}" ]]; then
    MODEL_SUFFIX="_${BACKBONE}"
  elif [[ -n "${MODEL_NAME}" ]]; then
    MODEL_SUFFIX="_$(sanitize_suffix "${MODEL_NAME}")"
  fi

  mkdir -p "${OUTPUT_DIR}"

  local mask
  for mask in $(seq 0 15); do
    local combo_name
    combo_name="$(build_combo_name "${mask}")"
    append_spec "combo|${combo_name}|$(( (mask & 1) != 0 ))|$(( (mask & 2) != 0 ))|$(( (mask & 4) != 0 ))|$(( (mask & 8) != 0 ))|"
  done

  if [[ ${INCLUDE_ADDITIONS} -eq 1 ]]; then
    append_spec "preset|matrix_hierarchical${MODEL_SUFFIX}|0|0|0|0|hierarchical_deberta"
    append_spec "preset|matrix_two_step${MODEL_SUFFIX}|0|0|0|0|two_step_impact_pipeline"
    append_spec "preset|matrix_sentence_token${MODEL_SUFFIX}|0|0|0|0|sentence_token_hierarchy"
    if [[ -z "${BACKBONE}" && -z "${MODEL_NAME}" ]]; then
      append_spec "preset|matrix_span_nested_gliner|0|0|0|0|span_nested_gliner"
    fi
  fi

  local timestamp
  local plan_log
  local gpu0_log
  local gpu1_log
  local pid0
  local pid1
  local status=0
  local forwarded_args_text=""

  if [[ ${#FORWARDED_ARGS[@]} -gt 0 ]]; then
    printf -v forwarded_args_text '%q ' "${FORWARDED_ARGS[@]}"
    forwarded_args_text="${forwarded_args_text% }"
  fi

  timestamp="$(date +%Y%m%d_%H%M%S)"
  plan_log="${OUTPUT_DIR}/ablation_addition_matrix_plan_${timestamp}.log"
  gpu0_log="${OUTPUT_DIR}/ablation_addition_matrix_gpu0_${timestamp}.log"
  gpu1_log="${OUTPUT_DIR}/ablation_addition_matrix_gpu1_${timestamp}.log"

  echo "Ablation + addition matrix launcher"
  echo "  Output dir: ${OUTPUT_DIR}"
  echo "  GPU 0 device: ${GPU0_DEVICE}"
  echo "  GPU 1 device: ${GPU1_DEVICE}"
  echo "  Backbone: ${BACKBONE:-<default>}"
  echo "  Model name override: ${MODEL_NAME:-<default>}"
  echo "  Main LR values: ${MODEL_LR_VALUES[*]}"
  if [[ ${INCLUDE_ADDITIONS} -eq 1 ]]; then
    echo "  GLiNER LR values: ${GLINER_LR_VALUES[*]}"
  fi
  echo "  Total queued jobs: ${#ALL_SPECS[@]}"
  echo "  GPU 0 jobs: ${#GPU0_SPECS[@]}"
  echo "  GPU 1 jobs: ${#GPU1_SPECS[@]}"
  echo "  GPU 0 log: ${gpu0_log}"
  echo "  GPU 1 log: ${gpu1_log}"
  echo "  Plan log: ${plan_log}"
  if [[ -n "${forwarded_args_text}" ]]; then
    echo "  Forwarded args: ${forwarded_args_text}"
  fi

  {
    echo "Ablation + addition matrix plan"
    echo "Generated at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "GPU 0 device: ${GPU0_DEVICE}"
    echo "GPU 1 device: ${GPU1_DEVICE}"
    echo "Backbone: ${BACKBONE:-<default>}"
    echo "Model name override: ${MODEL_NAME:-<default>}"
    echo "Main LR values: ${MODEL_LR_VALUES[*]}"
    if [[ ${INCLUDE_ADDITIONS} -eq 1 ]]; then
      echo "GLiNER LR values: ${GLINER_LR_VALUES[*]}"
    fi
    if [[ -n "${forwarded_args_text}" ]]; then
      echo "Forwarded args: ${forwarded_args_text}"
    fi
    echo
    echo "GPU 0 queue:"
    for spec in "${GPU0_SPECS[@]}"; do
      echo "  - $(build_spec_plan_line "${spec}")"
    done
    echo
    echo "GPU 1 queue:"
    for spec in "${GPU1_SPECS[@]}"; do
      echo "  - $(build_spec_plan_line "${spec}")"
    done
  } | tee "${plan_log}"

  (
    cd "${SCRIPT_DIR}"
    run_queue "GPU 0 queue" "${GPU0_DEVICE}" "${GPU0_SPECS[@]}"
  ) >"${gpu0_log}" 2>&1 &
  pid0=$!

  (
    cd "${SCRIPT_DIR}"
    run_queue "GPU 1 queue" "${GPU1_DEVICE}" "${GPU1_SPECS[@]}"
  ) >"${gpu1_log}" 2>&1 &
  pid1=$!

  echo "Started matrix queues with PIDs ${pid0} and ${pid1}"
  echo "Monitor progress with:"
  echo "  tail -f ${gpu0_log}"
  echo "  tail -f ${gpu1_log}"

  if ! wait "${pid0}"; then
    echo "GPU 0 matrix queue failed. See ${gpu0_log}" >&2
    status=1
  fi

  if ! wait "${pid1}"; then
    echo "GPU 1 matrix queue failed. See ${gpu1_log}" >&2
    status=1
  fi

  if [[ ${status} -ne 0 ]]; then
    exit "${status}"
  fi

  echo "Ablation + addition matrix completed successfully."
}

main "$@"
