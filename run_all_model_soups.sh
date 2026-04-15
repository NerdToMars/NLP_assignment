#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="cuda:0"
CHECKPOINT_LIMIT=5
OUTPUT_DIR=""
DATA_DIR=""
DRY_RUN=0

usage() {
  cat <<EOF
Usage: ./run_all_model_soups.sh --output-dir DIR --data-dir DIR [options]

Create one model soup per compatible trained experiment found under:
  <output-dir>/checkpoints/*/topk_summary.json

Supported model types:
  deberta, deberta_multitask, deberta_crf, deberta_crf_multitask, bilstm_crf

Excluded automatically:
  GLiNER, hierarchical classifier runs, hierarchical NER runs, and BiLSTM-CRF

Examples:
  ./run_all_model_soups.sh \\
    --output-dir /home/ismail/Documents/NLP_assignment/outputs_new_final_v2 \\
    --data-dir /home/ismail/Documents/NLP_assignment/SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2/dataset

  ./run_all_model_soups.sh \\
    --output-dir /home/ismail/Documents/NLP_assignment/outputs_new_final_clean \\
    --data-dir /home/ismail/Documents/NLP_assignment/SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2/dataset/cleaned

Options:
  --device DEVICE            CUDA device to use. Default: cuda:0
  --checkpoint-limit N       Top-k checkpoints to average per experiment. Default: 5
  --python BIN               Python interpreter. Default: \$PYTHON_BIN or python3
  --dry-run                  Print the commands without running them
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --checkpoint-limit)
      CHECKPOINT_LIMIT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" || -z "$DATA_DIR" ]]; then
  echo "--output-dir and --data-dir are required." >&2
  usage >&2
  exit 1
fi

if [[ ! -d "$OUTPUT_DIR/checkpoints" ]]; then
  echo "Missing checkpoints directory: $OUTPUT_DIR/checkpoints" >&2
  exit 1
fi

SUPPORTED_TYPES="deberta,deberta_multitask,deberta_crf,deberta_crf_multitask,bilstm_crf"

mapfile -t EXPERIMENTS < <(
  "$PYTHON_BIN" - <<'PY' "$OUTPUT_DIR" "$CHECKPOINT_LIMIT" "$SUPPORTED_TYPES"
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1]).resolve()
checkpoint_limit = int(sys.argv[2])
supported_types = set(sys.argv[3].split(","))

eligible = []
skipped = []

for summary_path in sorted((output_dir / "checkpoints").glob("*/topk_summary.json")):
    data = json.loads(summary_path.read_text())
    experiment_name = data.get("experiment_name", summary_path.parent.name)
    metadata = data.get("metadata", {})
    model_type = metadata.get("model_type")
    checkpoints = data.get("checkpoints", [])

    if experiment_name.startswith("hierarchical_deberta"):
        skipped.append((experiment_name, model_type or "unknown", "hierarchical runs are excluded from batch soup"))
        continue
    if model_type not in supported_types:
        skipped.append((experiment_name, model_type or "unknown", "unsupported model_type"))
        continue
    if model_type == "bilstm_crf":
        skipped.append((experiment_name, model_type, "requires the original saved word2idx mapping"))
        continue
    if len(checkpoints) < 2:
        skipped.append((experiment_name, model_type, f"only {len(checkpoints)} checkpoint(s)"))
        continue

    eligible.append(experiment_name)

print("Eligible experiments:", file=sys.stderr)
for name in eligible:
    print(f"  {name}", file=sys.stderr)

if skipped:
    print("Skipped experiments:", file=sys.stderr)
    for name, model_type, reason in skipped:
        print(f"  {name} [{model_type}] - {reason}", file=sys.stderr)

for name in eligible:
    print(name)
PY
)

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  echo "No compatible experiments found for model soup in $OUTPUT_DIR" >&2
  exit 1
fi

echo
echo "Running one soup per compatible experiment"
echo "Output dir : $OUTPUT_DIR"
echo "Data dir   : $DATA_DIR"
echo "Device     : $DEVICE"
echo "Top-k      : $CHECKPOINT_LIMIT"
echo "Count      : ${#EXPERIMENTS[@]}"
echo

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  SOUP_NAME="${EXPERIMENT}_soup"
  CMD=(
    "$PYTHON_BIN" "$ROOT_DIR/run_experiments.py" run
    --experiment model_soup
    --output-dir "$OUTPUT_DIR"
    --data-dir "$DATA_DIR"
    --device "$DEVICE"
    --checkpoint-limit "$CHECKPOINT_LIMIT"
    --source-experiment "$EXPERIMENT"
    --experiment-name "$SOUP_NAME"
  )

  echo "============================================================"
  echo "Soup source : $EXPERIMENT"
  echo "Soup target : $SOUP_NAME"
  echo "============================================================"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${CMD[@]}"
    printf '\n\n'
  else
    "${CMD[@]}"
    echo
  fi
done
