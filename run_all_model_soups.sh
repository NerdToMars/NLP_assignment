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

DEVICE="cuda:0"
CHECKPOINT_LIMITS=(3 5)
CUSTOM_CHECKPOINT_LIMITS=0
SKIP_EXISTING=0
DRY_RUN=0
SEARCH_ROOTS=("${SCRIPT_DIR}")
FORWARDED_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./run_all_model_soups.sh
  ./run_all_model_soups.sh --dry-run
  ./run_all_model_soups.sh --skip-existing
  ./run_all_model_soups.sh --device cuda:1
  ./run_all_model_soups.sh --checkpoint-limit 3 --checkpoint-limit 5
  ./run_all_model_soups.sh --search-root /path/to/output_dir
  ./run_all_model_soups.sh --search-root outputs_new_final_v2 --search-root outputs_new_final_v5
  ./run_all_model_soups.sh -- --batch-size 16

Behavior:
  - Recursively discovers training runs that have top-k checkpoint summaries.
  - Builds one model soup per supported source experiment and checkpoint limit.
  - Writes each soup back into the same output root as its source checkpoints.
  - Defaults to checkpoint limits 3 and 5.
  - Currently soups only these model types:
      deberta
      deberta_multitask
      deberta_crf
      deberta_crf_multitask
  - Skips unsupported families such as GLiNER, sentence-token hierarchy,
    two-step classifier/extractor submodels, sentence classifiers, and BiLSTM-CRF.
EOF
}

main() {
  local -a search_roots=("${SEARCH_ROOTS[@]}")
  local -a supported_entries=()
  local -a skipped_entries=()
  local -a cmd=()
  local line=""
  local status=""
  local output_root=""
  local source_experiment=""
  local model_type=""
  local detail=""
  local experiment_name=""
  local index=0
  local total=0
  local checkpoint_limit=""

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
      --checkpoint-limit)
        if [[ ${CUSTOM_CHECKPOINT_LIMITS} -eq 0 ]]; then
          CHECKPOINT_LIMITS=()
          CUSTOM_CHECKPOINT_LIMITS=1
        fi
        CHECKPOINT_LIMITS+=("$2")
        shift 2
        ;;
      --search-root)
        if [[ ${#search_roots[@]} -eq 1 && "${search_roots[0]}" == "${SCRIPT_DIR}" ]]; then
          search_roots=()
        fi
        search_roots+=("$2")
        shift 2
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

  if [[ ${#CHECKPOINT_LIMITS[@]} -eq 0 ]]; then
    echo "At least one --checkpoint-limit value is required." >&2
    exit 1
  fi

  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    IFS=$'\t' read -r status output_root source_experiment model_type detail <<<"${line}"
    if [[ "${status}" == "SUPPORTED" ]]; then
      supported_entries+=("${output_root}"$'\t'"${source_experiment}"$'\t'"${model_type}"$'\t'"${detail}")
    else
      skipped_entries+=("${output_root}"$'\t'"${source_experiment}"$'\t'"${model_type}"$'\t'"${detail}")
    fi
  done < <(
    "${PYTHON_CMD[@]}" - "${search_roots[@]}" <<'PY'
import json
import sys
from pathlib import Path

supported_types = {
    "deberta",
    "deberta_multitask",
    "deberta_crf",
    "deberta_crf_multitask",
}

seen = set()
records = []
for root_arg in sys.argv[1:]:
    root = Path(root_arg).expanduser().resolve()
    if not root.exists():
        records.append(("SKIPPED", str(root), "-", "-", "search root does not exist"))
        continue
    for summary_path in sorted(root.rglob("topk_summary.json")):
        if summary_path.name != "topk_summary.json":
            continue
        if len(summary_path.parts) < 3 or summary_path.parent.parent.name != "checkpoints":
            continue
        summary_path = summary_path.resolve()
        if summary_path in seen:
            continue
        seen.add(summary_path)

        output_root = summary_path.parent.parent.parent
        source_experiment = summary_path.parent.name
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:  # pragma: no cover - defensive logging in shell helper
            records.append(("SKIPPED", str(output_root), source_experiment, "-", f"invalid summary JSON: {exc}"))
            continue

        metadata = summary.get("metadata", {})
        model_type = metadata.get("model_type", "-")
        model_name = metadata.get("model_name", "-")

        if model_type in supported_types:
            records.append(("SUPPORTED", str(output_root), source_experiment, model_type, model_name))
        elif model_type == "bilstm_crf":
            records.append(
                (
                    "SKIPPED",
                    str(output_root),
                    source_experiment,
                    model_type,
                    "bilstm_crf soup disabled because vocab alignment is unsafe",
                )
            )
        else:
            records.append(
                (
                    "SKIPPED",
                    str(output_root),
                    source_experiment,
                    model_type,
                    "unsupported model_type for current model_soup",
                )
            )

for record in sorted(records, key=lambda item: (item[0], item[1], item[2], item[3], item[4])):
    print("\t".join(record))
PY
  )

  total=$((${#supported_entries[@]} * ${#CHECKPOINT_LIMITS[@]}))

  echo "Discovered ${total} soupable source experiments."
  echo "  Search roots: ${search_roots[*]}"
  echo "  Device: ${DEVICE}"
  echo "  Checkpoint limits: ${CHECKPOINT_LIMITS[*]}"
  if [[ ${SKIP_EXISTING} -eq 1 ]]; then
    echo "  Skip existing: yes"
  else
    echo "  Skip existing: no"
  fi
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  Dry run: yes"
  else
    echo "  Dry run: no"
  fi
  echo

  if [[ ${#skipped_entries[@]} -gt 0 ]]; then
    echo "Skipped unsupported or unsafe source experiments:"
    for line in "${skipped_entries[@]}"; do
      IFS=$'\t' read -r output_root source_experiment model_type detail <<<"${line}"
      echo "  - ${source_experiment} [${model_type}] in ${output_root}: ${detail}"
    done
    echo
  fi

  if [[ ${total} -eq 0 ]]; then
    echo "No soupable source experiments were found."
    exit 0
  fi

  echo "Soup plan:"
  for line in "${supported_entries[@]}"; do
    IFS=$'\t' read -r output_root source_experiment model_type detail <<<"${line}"
    for checkpoint_limit in "${CHECKPOINT_LIMITS[@]}"; do
      echo "  - ${source_experiment} [${model_type}] in ${output_root} (top ${checkpoint_limit})"
    done
  done
  echo

  for line in "${supported_entries[@]}"; do
    IFS=$'\t' read -r output_root source_experiment model_type detail <<<"${line}"
    for checkpoint_limit in "${CHECKPOINT_LIMITS[@]}"; do
      index=$((index + 1))
      experiment_name="${source_experiment}_top${checkpoint_limit}_soup"

      cmd=(
        "${PYTHON_CMD[@]}"
        run_experiments.py
        run
        --experiment model_soup
        --output-dir "${output_root}"
        --source-experiment "${source_experiment}"
        --checkpoint-limit "${checkpoint_limit}"
        --experiment-name "${experiment_name}"
        --device "${DEVICE}"
      )

      if [[ ${SKIP_EXISTING} -eq 1 ]]; then
        cmd+=(--skip-existing)
      fi

      if [[ ${#FORWARDED_ARGS[@]} -gt 0 ]]; then
        cmd+=("${FORWARDED_ARGS[@]}")
      fi

      echo "[${index}/${total}] ${source_experiment} -> ${experiment_name}"
      echo "  Output root: ${output_root}"
      echo "  Model type: ${model_type}"
      echo "  Checkpoint limit: ${checkpoint_limit}"

      if [[ ${DRY_RUN} -eq 1 ]]; then
        printf '  Command:'
        printf ' %q' "${cmd[@]}"
        printf '\n\n'
        continue
      fi

      (cd "${SCRIPT_DIR}" && "${cmd[@]}")
      echo
    done
  done
}

main "$@"
