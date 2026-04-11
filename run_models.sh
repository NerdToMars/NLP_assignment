#!/usr/bin/env bash

set -euo pipefail

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

CORE_EXPERIMENTS=(
  bilstm
  deberta_baseline
  deberta_focal
  deberta_definition
  deberta_multitask
  deberta_synthetic_curriculum
  deberta_combined
  deberta_combined_no_synth
  gliner
  gliner_finetune
)

ADVANCED_EXPERIMENTS=(
  recall_boost_s42
  recall_boost_s123
  rdrop_s42
  rdrop_s123
  fgm_swa_s42
)

usage() {
  cat <<'EOF'
Usage:
  ./run_models.sh
  ./run_models.sh core
  ./run_models.sh advanced
  ./run_models.sh all
  ./run_models.sh ensemble -- --output-dir /path/to/checkpoints
  ./run_models.sh deberta_baseline deberta_multitask -- --device cuda:0 --epochs 5
  ./run_models.sh deberta_baseline -- --lr 1e-5 --lr 2e-5 --lr 3e-5
  ./run_models.sh advanced -- --early-stopping-patience 5 --early-stopping-min-delta 0.001
  ./run_models.sh hierarchical_deberta -- --device cuda:0 --threshold 0.5
  ./run_models.sh model_soup -- --source-experiment deberta_baseline_lr2em05 --checkpoint-limit 5
  ./run_models.sh ensemble_search -- --source-experiment recall_boost_ow02_s42_lr2em05 --source-experiment fgm05_swa_s42_lr2em05 --min-models 2 --max-models 5
  ./run_models.sh list

Behavior:
  - Runs each experiment in a fresh Python process to release GPU memory between jobs.
  - Pass extra CLI flags to run_experiments.py after a standalone -- separator.
  - By default, the script runs the core training and evaluation presets.
  - Repeating --lr runs the same experiment once per learning rate and writes separate logs.
EOF
}

main() {
  local -a experiments=()
  local -a extra_args=()
  local extra_args_text=""

  if [[ $# -eq 0 ]]; then
    experiments=("${CORE_EXPERIMENTS[@]}")
  else
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      list)
        "${PYTHON_CMD[@]}" run_experiments.py list
        exit 0
        ;;
      core)
        experiments=("${CORE_EXPERIMENTS[@]}")
        shift
        ;;
      advanced)
        experiments=("${ADVANCED_EXPERIMENTS[@]}")
        shift
        ;;
      all)
        experiments=("${CORE_EXPERIMENTS[@]}" "${ADVANCED_EXPERIMENTS[@]}")
        shift
        ;;
      *)
        while [[ $# -gt 0 ]]; do
          if [[ "$1" == "--" ]]; then
            shift
            break
          fi
          experiments+=("$1")
          shift
        done
        ;;
    esac
  fi

  if [[ ${#experiments[@]} -eq 0 ]]; then
    experiments=("${CORE_EXPERIMENTS[@]}")
  fi

  if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--" ]]; then
      shift
    fi
    extra_args=("$@")
  fi

  if [[ ${#extra_args[@]} -gt 0 ]]; then
    printf -v extra_args_text '%q ' "${extra_args[@]}"
    extra_args_text="${extra_args_text% }"
  fi

  local total="${#experiments[@]}"
  local index=0

  echo "Run queue summary"
  echo "  Total experiments: ${total}"
  echo "  Experiments: ${experiments[*]}"
  if [[ -n "${extra_args_text}" ]]; then
    echo "  Extra args: ${extra_args_text}"
  fi

  for experiment in "${experiments[@]}"; do
    index=$((index + 1))
    echo
    echo "================================================================"
    echo "[$index/$total] Running ${experiment}"
    echo "================================================================"
    "${PYTHON_CMD[@]}" run_experiments.py run --experiment "${experiment}" "${extra_args[@]}"

    if command -v nvidia-smi >/dev/null 2>&1; then
      echo
      echo "GPU memory snapshot after ${experiment}:"
      nvidia-smi \
        --query-gpu="index,name,memory.used,memory.total" \
        --format="csv,noheader" || true
    fi
  done
}

main "$@"
