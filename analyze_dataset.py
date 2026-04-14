#!/usr/bin/env python3
"""Summarize Reddit Impacts dataset splits for quick quality checks."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_DATASET_DIR = Path("SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2/dataset")
DEFAULT_FILES = (
    DEFAULT_DATASET_DIR / "new_train_data.csv",
    DEFAULT_DATASET_DIR / "new_dev_data.csv",
)
METADATA_MARKERS = {"submission_title", "submission_subreddit"}
MOJIBAKE_MARKERS = ("â€™", "â€œ", "â€", "Ã", "�")
PLACEHOLDER_LABELS = {"", "''", "_", "O"}


@dataclass
class DatasetSummary:
    dataset: str
    total_rows: int
    entity_rows: int
    entity_pct: float
    all_o_rows: int
    all_o_pct: float
    clinical_rows: int
    social_rows: int
    both_rows: int
    metadata_rows: int
    metadata_pct: float
    mojibake_rows: int
    quote_artifact_rows: int
    duplicate_text_groups: int
    extra_duplicate_texts: int
    conflicting_duplicate_groups: int
    unique_ids: int
    nonunique_id_groups: int
    length_mismatches: int
    token_len_min: int
    token_len_max: int
    token_len_mean: float
    token_len_median: float
    entity_row_type_breakdown: dict[str, int]
    top_entity_tags: list[tuple[str, int]]
    label_values_top: list[tuple[str, int]]
    label_placeholders: dict[str, int]
    top_duplicate_ids: list[tuple[str, int]]
    sample_conflicts: list[dict[str, object]]
    sample_metadata_rows: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Reddit Impacts CSV splits. "
            "If no paths are provided, analyzes the default train and dev files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="CSV files to analyze.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="Optional path to save the summaries as JSON.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="How many conflict and metadata samples to print per dataset.",
    )
    parser.add_argument(
        "--top-tags",
        type=int,
        default=12,
        help="How many top entity and label counts to display.",
    )
    return parser.parse_args()


def parse_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, list):
            return parsed
    raise TypeError(f"Unsupported list value: {type(value)}")


def round_pct(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 2) if denominator else 0.0


def safe_mean(values: list[int]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def safe_median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize_csv(path: Path, show_samples: int, top_tags: int) -> DatasetSummary:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    id_counter: Counter[str] = Counter()
    text_counter: Counter[str] = Counter()
    text_to_ner: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)
    labels_values_counter: Counter[str] = Counter()
    label_placeholders: Counter[str] = Counter()
    entity_token_counter: Counter[str] = Counter()
    entity_row_type_breakdown: Counter[str] = Counter()
    token_lengths: list[int] = []
    conflicting_duplicate_groups: list[dict[str, object]] = []
    metadata_examples: list[str] = []

    entity_rows = 0
    all_o_rows = 0
    clinical_rows = 0
    social_rows = 0
    both_rows = 0
    metadata_rows = 0
    mojibake_rows = 0
    quote_artifact_rows = 0
    length_mismatches = 0

    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        tokens = parse_list(row.get("tokens"))
        labels = parse_list(row.get("labels"))
        ner_tags = parse_list(row.get("ner_tags"))
        identifier = str(row.get("ID", ""))
        normalized_rows.append(
            {
                "id": identifier,
                "tokens": tokens,
                "labels": labels,
                "ner_tags": ner_tags,
            }
        )

    for row in normalized_rows:
        identifier = row["id"]
        tokens = row["tokens"]
        labels = row["labels"]
        ner_tags = row["ner_tags"]
        text = " ".join(str(token) for token in tokens)

        id_counter[identifier] += 1
        text_counter[text] += 1
        text_to_ner[text].add(tuple(str(tag) for tag in ner_tags))
        token_lengths.append(len(tokens))

        if not (len(tokens) == len(labels) == len(ner_tags)):
            length_mismatches += 1

        row_has_entity = any(tag != "O" for tag in ner_tags)
        if row_has_entity:
            entity_rows += 1
        else:
            all_o_rows += 1

        clinical = any("ClinicalImpacts" in str(tag) for tag in ner_tags)
        social = any("SocialImpacts" in str(tag) for tag in ner_tags)
        if clinical:
            clinical_rows += 1
        if social:
            social_rows += 1
        if clinical and social:
            both_rows += 1

        if clinical and social:
            entity_row_type_breakdown["both"] += 1
        elif clinical:
            entity_row_type_breakdown["clinical_only"] += 1
        elif social:
            entity_row_type_breakdown["social_only"] += 1
        else:
            entity_row_type_breakdown["none"] += 1

        if any(str(token) in METADATA_MARKERS for token in tokens):
            metadata_rows += 1
            if len(metadata_examples) < show_samples:
                metadata_examples.append(text[:260])

        if any(marker in text for marker in MOJIBAKE_MARKERS):
            mojibake_rows += 1

        if '""' in text:
            quote_artifact_rows += 1

        for tag in ner_tags:
            tag = str(tag)
            if tag != "O":
                entity_token_counter[tag] += 1

        for label in labels:
            label = str(label)
            labels_values_counter[label] += 1
            if label in PLACEHOLDER_LABELS:
                label_placeholders[label] += 1

    duplicate_text_groups = sum(1 for count in text_counter.values() if count > 1)
    extra_duplicate_texts = sum(count - 1 for count in text_counter.values() if count > 1)
    nonunique_id_groups = sum(1 for count in id_counter.values() if count > 1)

    for text, tagsets in text_to_ner.items():
        if len(tagsets) > 1:
            conflicting_duplicate_groups.append(
                {
                    "text": text[:260],
                    "count": text_counter[text],
                    "distinct_tagsets": len(tagsets),
                }
            )

    return DatasetSummary(
        dataset=str(path),
        total_rows=len(normalized_rows),
        entity_rows=entity_rows,
        entity_pct=round_pct(entity_rows, len(normalized_rows)),
        all_o_rows=all_o_rows,
        all_o_pct=round_pct(all_o_rows, len(normalized_rows)),
        clinical_rows=clinical_rows,
        social_rows=social_rows,
        both_rows=both_rows,
        metadata_rows=metadata_rows,
        metadata_pct=round_pct(metadata_rows, len(normalized_rows)),
        mojibake_rows=mojibake_rows,
        quote_artifact_rows=quote_artifact_rows,
        duplicate_text_groups=duplicate_text_groups,
        extra_duplicate_texts=extra_duplicate_texts,
        conflicting_duplicate_groups=len(conflicting_duplicate_groups),
        unique_ids=len(id_counter),
        nonunique_id_groups=nonunique_id_groups,
        length_mismatches=length_mismatches,
        token_len_min=min(token_lengths) if token_lengths else 0,
        token_len_max=max(token_lengths) if token_lengths else 0,
        token_len_mean=safe_mean(token_lengths),
        token_len_median=safe_median(token_lengths),
        entity_row_type_breakdown=dict(entity_row_type_breakdown),
        top_entity_tags=entity_token_counter.most_common(top_tags),
        label_values_top=labels_values_counter.most_common(top_tags),
        label_placeholders=dict(label_placeholders),
        top_duplicate_ids=id_counter.most_common(10),
        sample_conflicts=conflicting_duplicate_groups[:show_samples],
        sample_metadata_rows=metadata_examples,
    )


def print_summary(summary: DatasetSummary) -> None:
    print(f"\n=== {summary.dataset} ===")
    print(f"Rows: {summary.total_rows}")
    print(
        "Entity rows: "
        f"{summary.entity_rows} ({summary.entity_pct:.2f}%) | "
        f"All-O rows: {summary.all_o_rows} ({summary.all_o_pct:.2f}%)"
    )
    print(
        "Clinical rows: "
        f"{summary.clinical_rows} | Social rows: {summary.social_rows} | Both: {summary.both_rows}"
    )
    print(
        "Metadata rows: "
        f"{summary.metadata_rows} ({summary.metadata_pct:.2f}%) | "
        f"Mojibake rows: {summary.mojibake_rows} | "
        f'Quote-artifact rows: {summary.quote_artifact_rows}'
    )
    print(
        "Duplicate text groups: "
        f"{summary.duplicate_text_groups} | "
        f"Extra duplicate texts: {summary.extra_duplicate_texts} | "
        f"Conflicting duplicate groups: {summary.conflicting_duplicate_groups}"
    )
    print(
        "Unique IDs: "
        f"{summary.unique_ids} | Non-unique ID groups: {summary.nonunique_id_groups} | "
        f"Length mismatches: {summary.length_mismatches}"
    )
    print(
        "Token length: "
        f"min={summary.token_len_min}, "
        f"max={summary.token_len_max}, "
        f"mean={summary.token_len_mean:.2f}, "
        f"median={summary.token_len_median:.2f}"
    )
    print(f"Entity row breakdown: {summary.entity_row_type_breakdown}")
    print(f"Top entity tags: {summary.top_entity_tags}")
    print(f"Top label values: {summary.label_values_top}")
    print(f"Label placeholders: {summary.label_placeholders}")

    if summary.sample_conflicts:
        print("Sample conflicting duplicates:")
        for item in summary.sample_conflicts:
            print(
                f"  count={item['count']} tagsets={item['distinct_tagsets']} text={item['text']}"
            )

    if summary.sample_metadata_rows:
        print("Sample metadata rows:")
        for text in summary.sample_metadata_rows:
            print(f"  {text}")


def print_comparison(summaries: list[DatasetSummary]) -> None:
    if len(summaries) < 2:
        return

    print("\n=== Comparison ===")
    header = (
        f"{'dataset':<40} {'rows':>6} {'entity%':>8} {'all-O%':>8} "
        f"{'clinical':>9} {'social':>8} {'meta%':>8} {'dupes':>7}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        dataset_name = Path(summary.dataset).name
        print(
            f"{dataset_name:<40} "
            f"{summary.total_rows:>6} "
            f"{summary.entity_pct:>8.2f} "
            f"{summary.all_o_pct:>8.2f} "
            f"{summary.clinical_rows:>9} "
            f"{summary.social_rows:>8} "
            f"{summary.metadata_pct:>8.2f} "
            f"{summary.duplicate_text_groups:>7}"
        )


def main() -> int:
    args = parse_args()
    paths = args.paths or list(DEFAULT_FILES)

    summaries: list[DatasetSummary] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        summaries.append(summarize_csv(path, args.show_samples, args.top_tags))

    for summary in summaries:
        print_summary(summary)
    print_comparison(summaries)

    if args.json_path is not None:
        payload = [asdict(summary) for summary in summaries]
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON summary to {args.json_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
