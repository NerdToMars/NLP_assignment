"""Reusable dataset cleanup and runtime preprocessing utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


METADATA_MARKERS = {"submission_title", "submission_subreddit"}
USER_PREFIXES = ("u/", "/u/")
URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)
MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€\"": "-",
    "â€“": "-",
    "â€”": "-",
    "Ã©": "e",
}


@dataclass(frozen=True)
class RuntimePreprocessingConfig:
    """Safe token-aligned preprocessing switches for runtime loading."""

    normalize_encoding_artifacts: bool = True
    remove_metadata_markers: bool = True
    replace_user_mentions: bool = True
    replace_urls: bool = True


DEFAULT_RUNTIME_PREPROCESSING = RuntimePreprocessingConfig()


def normalize_token_artifacts(token: str) -> str:
    """Fix the token-level mojibake artifacts observed in the train split."""
    normalized = token
    for source, target in MOJIBAKE_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


def preprocess_token(token: str, config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING) -> str:
    """Apply safe token-only replacements that do not change sequence length."""
    normalized = normalize_token_artifacts(token) if config.normalize_encoding_artifacts else token

    if config.replace_user_mentions and normalized.startswith(USER_PREFIXES):
        return "[USER]"
    if config.replace_urls and URL_PATTERN.match(normalized):
        return "[URL]"
    return normalized


def preprocess_tokens(
    tokens: list[str],
    config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING,
) -> list[str]:
    """Apply token-only runtime preprocessing."""
    return [preprocess_token(str(token), config=config) for token in tokens]


def preprocess_labeled_row(
    tokens: list[Any],
    ner_tags: list[Any],
    labels: list[Any] | None = None,
    config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING,
) -> tuple[list[str], list[str], list[str] | None]:
    """Apply safe row-level preprocessing while preserving token/tag alignment."""
    processed_tokens: list[str] = []
    processed_tags: list[str] = []
    processed_labels: list[str] | None = [] if labels is not None else None

    for index, raw_token in enumerate(tokens):
        token = str(raw_token)
        if config.remove_metadata_markers and token in METADATA_MARKERS:
            continue

        processed_tokens.append(preprocess_token(token, config=config))
        processed_tags.append(str(ner_tags[index]))
        if processed_labels is not None:
            processed_labels.append(str(labels[index]))

    if not processed_tokens:
        fallback_tokens = preprocess_tokens([str(token) for token in tokens], config=config)
        fallback_tags = [str(tag) for tag in ner_tags]
        fallback_labels = [str(label) for label in labels] if labels is not None else None
        return fallback_tokens, fallback_tags, fallback_labels

    return processed_tokens, processed_tags, processed_labels


def apply_runtime_preprocessing(
    df,
    config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING,
) -> Any:
    """Apply safe preprocessing to a dataframe loaded from the task CSV files."""
    working = df.copy()
    has_labels = "labels" in working.columns

    processed = working.apply(
        lambda row: preprocess_labeled_row(
            row["tokens"],
            row["ner_tags"],
            row["labels"] if has_labels else None,
            config=config,
        ),
        axis=1,
    )

    working["tokens"] = processed.apply(lambda item: item[0])
    working["ner_tags"] = processed.apply(lambda item: item[1])
    if has_labels:
        working["labels"] = processed.apply(lambda item: item[2])
    return working


def _token_key(tokens: list[Any]) -> tuple[str, ...]:
    return tuple(str(token) for token in tokens)


def summarize_duplicate_groups_from_rows(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Report how many duplicate groups are exact duplicates versus conflicting labels."""
    grouped: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
    counts: dict[tuple[str, ...], int] = {}

    for row in rows:
        key = _token_key(row["tokens"])
        counts[key] = counts.get(key, 0) + 1
        grouped.setdefault(key, set()).add(tuple(str(tag) for tag in row["ner_tags"]))

    duplicate_keys = [key for key, count in counts.items() if count > 1]
    exact_duplicate_groups = 0
    conflicting_duplicate_groups = 0
    for key in duplicate_keys:
        if len(grouped[key]) > 1:
            conflicting_duplicate_groups += 1
        else:
            exact_duplicate_groups += 1

    return {
        "duplicate_groups": len(duplicate_keys),
        "exact_duplicate_groups": exact_duplicate_groups,
        "conflicting_duplicate_groups": conflicting_duplicate_groups,
    }


def drop_duplicate_text_rows_from_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop every row that belongs to a duplicated text group."""
    counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = _token_key(row["tokens"])
        counts[key] = counts.get(key, 0) + 1

    cleaned_rows = [row for row in rows if counts[_token_key(row["tokens"])] == 1]
    duplicate_groups = sum(1 for count in counts.values() if count > 1)
    dropped_rows = len(rows) - len(cleaned_rows)
    stats = {
        "original_rows": int(len(rows)),
        "duplicate_groups": int(duplicate_groups),
        "dropped_rows": int(dropped_rows),
        "remaining_rows": int(len(cleaned_rows)),
    }
    return cleaned_rows, stats


def drop_all_duplicate_text_rows(df) -> tuple[Any, dict[str, int]]:
    """Drop every row that belongs to a duplicated text group."""
    rows = [row.to_dict() for _, row in df.iterrows()]
    cleaned_rows, stats = drop_duplicate_text_rows_from_rows(rows)
    cleaned = df.iloc[0:0].copy()
    if cleaned_rows:
        cleaned = cleaned.from_records(cleaned_rows)
    return cleaned.reset_index(drop=True), stats


def summarize_duplicate_groups(df) -> dict[str, int]:
    """Report how many duplicate groups are exact duplicates versus conflicting labels."""
    rows = [row.to_dict() for _, row in df.iterrows()]
    return summarize_duplicate_groups_from_rows(rows)
