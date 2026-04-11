#!/usr/bin/env python3
"""Create cleaned dataset files with duplicate text groups removed."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.preprocessing import drop_duplicate_text_rows_from_rows, summarize_duplicate_groups_from_rows


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2" / "dataset"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "cleaned"


@dataclass
class CleaningSummary:
    dataset: str
    input_path: str
    output_path: str
    original_rows: int
    remaining_rows: int
    dropped_rows: int
    duplicate_groups: int
    exact_duplicate_groups: int
    conflicting_duplicate_groups: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write cleaned dataset CSVs with every duplicated text group removed. "
            "The original files are left unchanged."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the raw dataset CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write cleaned CSV files into.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=["new_train_data.csv", "new_dev_data.csv"],
        help="Dataset CSV filenames to clean.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional explicit path for the cleaning summary JSON.",
    )
    return parser.parse_args()


def clean_one_file(input_path: Path, output_path: Path) -> CleaningSummary:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            parsed = dict(row)
            parsed["tokens"] = ast.literal_eval(parsed["tokens"].strip())
            parsed["ner_tags"] = ast.literal_eval(parsed["ner_tags"].strip())
            if "labels" in parsed and parsed["labels"] is not None:
                parsed["labels"] = ast.literal_eval(parsed["labels"].strip())
            rows.append(parsed)

    duplicate_summary = summarize_duplicate_groups_from_rows(rows)
    cleaned_rows, drop_summary = drop_duplicate_text_rows_from_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in cleaned_rows:
            writer.writerow(
                {
                    key: row[key]
                    for key in fieldnames
                }
            )

    return CleaningSummary(
        dataset=input_path.name,
        input_path=str(input_path.resolve()),
        output_path=str(output_path.resolve()),
        original_rows=drop_summary["original_rows"],
        remaining_rows=drop_summary["remaining_rows"],
        dropped_rows=drop_summary["dropped_rows"],
        duplicate_groups=duplicate_summary["duplicate_groups"],
        exact_duplicate_groups=duplicate_summary["exact_duplicate_groups"],
        conflicting_duplicate_groups=duplicate_summary["conflicting_duplicate_groups"],
    )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    summaries: list[CleaningSummary] = []
    for filename in args.files:
        input_path = input_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {input_path}")

        output_path = output_dir / filename
        summary = clean_one_file(input_path, output_path)
        summaries.append(summary)
        print(
            f"{summary.dataset}: kept {summary.remaining_rows}/{summary.original_rows} rows "
            f"(dropped {summary.dropped_rows} rows across {summary.duplicate_groups} duplicate group(s))"
        )

    summary_path = args.summary_json.resolve() if args.summary_json else output_dir / "cleaning_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps([asdict(item) for item in summaries], indent=2), encoding="utf-8")
    print(f"Saved cleaning summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
