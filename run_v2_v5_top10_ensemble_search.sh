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

V2_DIR="${SCRIPT_DIR}/outputs_new_final_v2"
V5_DIR="${SCRIPT_DIR}/outputs_new_final_v5"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_cross_version_ensembles"
DEVICE="cuda:0"
EXPERIMENT_NAME="ensemble_search_v2_v5_top7_relaxed_strict"
TOP_RELAXED=7
TOP_STRICT=7
CHECKPOINT_LIMIT=1
VOTE_METHOD="majority_vote"
SAVE_COMBINATION_FILES=0
SKIP_EXISTING=0
DRY_RUN=0
FORWARDED_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./run_v2_v5_top10_ensemble_search.sh
  ./run_v2_v5_top10_ensemble_search.sh --dry-run
  ./run_v2_v5_top10_ensemble_search.sh --device cuda:1
  ./run_v2_v5_top10_ensemble_search.sh --output-dir /path/to/output_dir
  ./run_v2_v5_top10_ensemble_search.sh --skip-existing

Behavior:
  - Selects the top relaxed-F1 and strict-F1 runs from:
      outputs_new_final_v2  (preprocessing disabled)
      outputs_new_final_v5  (preprocessing enabled)
  - Uses the union of those candidate pools for ensemble search.
  - Searches explicit ensemble sizes:
      2, 3, 4, 5, 7
  - Forces majority_vote so hierarchical / staged pipelines can be included.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --v2-dir)
      V2_DIR="$2"
      shift 2
      ;;
    --v5-dir)
      V5_DIR="$2"
      shift 2
      ;;
    --experiment-name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --top-relaxed)
      TOP_RELAXED="$2"
      shift 2
      ;;
    --top-strict)
      TOP_STRICT="$2"
      shift 2
      ;;
    --checkpoint-limit)
      CHECKPOINT_LIMIT="$2"
      shift 2
      ;;
    --save-combination-files)
      SAVE_COMBINATION_FILES=1
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

mkdir -p "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_CMD[@]}"
  run_experiments.py
  run
  --experiment ensemble_search
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
  --experiment-name "${EXPERIMENT_NAME}"
  --vote-method "${VOTE_METHOD}"
  --candidate-output-dir "${V2_DIR}::disabled"
  --candidate-output-dir "${V5_DIR}::enabled"
  --top-relaxed-per-output "${TOP_RELAXED}"
  --top-strict-per-output "${TOP_STRICT}"
  --checkpoint-limit "${CHECKPOINT_LIMIT}"
  --ensemble-size 2
  --ensemble-size 3
  --ensemble-size 4
  --ensemble-size 5
  --ensemble-size 7
)

if [[ ${SAVE_COMBINATION_FILES} -eq 1 ]]; then
  cmd+=(--save-combination-files)
fi

if [[ ${SKIP_EXISTING} -eq 1 ]]; then
  cmd+=(--skip-existing)
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  cmd+=(--dry-run)
fi

if [[ ${#FORWARDED_ARGS[@]} -gt 0 ]]; then
  cmd+=("${FORWARDED_ARGS[@]}")
fi

echo "Running cross-version ensemble search"
echo "  V2 dir: ${V2_DIR} (preprocessing disabled)"
echo "  V5 dir: ${V5_DIR} (preprocessing enabled)"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Device: ${DEVICE}"
echo "  Top relaxed per output: ${TOP_RELAXED}"
echo "  Top strict per output: ${TOP_STRICT}"
echo "  Ensemble sizes: 2 3 4 5 7"
echo

(cd "${SCRIPT_DIR}" && "${cmd[@]}")
