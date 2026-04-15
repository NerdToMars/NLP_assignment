#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$ROOT_DIR/run_all_model_soups.sh" \
  --output-dir "$ROOT_DIR/outputs_new_final_clean" \
  --data-dir "$ROOT_DIR/SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2/dataset/cleaned" \
  "$@"
