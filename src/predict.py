"""Prediction utilities for running trained checkpoints on CSV datasets."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .bilstm_crf import BiLSTMCRF
from .deberta_crf import DeBERTaCRF, DeBERTaCRFMultiTask
from .deberta_ner import DeBERTaNER, DeBERTaNERMultiTask
from .preprocessing import (
    DEFAULT_RUNTIME_PREPROCESSING,
    RuntimePreprocessingConfig,
    preprocess_tokens as runtime_preprocess_tokens,
    preprocess_tokens_with_alignment as runtime_preprocess_tokens_with_alignment,
    restore_forced_o_predictions,
)


LABEL2ID = {
    "O": 0,
    "B-ClinicalImpacts": 1,
    "I-ClinicalImpacts": 2,
    "B-SocialImpacts": 3,
    "I-SocialImpacts": 4,
}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)
DEFAULT_WINDOW_OVERLAP = 96

CLINICAL_DEFINITION = (
    "Clinical Impacts are physical or psychological consequences of substance use "
    "personally experienced by the post author, such as overdose, withdrawal, "
    "addiction, depression, anxiety, hospitalization, or medical treatment."
)
SOCIAL_DEFINITION = (
    "Social Impacts are social, occupational, legal, or relational consequences of "
    "substance use personally experienced by the post author, such as job loss, "
    "arrest, criminal charges, relationship breakdown, financial hardship, or homelessness."
)
ENTITY_DEFINITION_PREFIX = f"{CLINICAL_DEFINITION} {SOCIAL_DEFINITION}"

MODEL_TYPES = {
    "deberta": DeBERTaNER,
    "deberta_multitask": DeBERTaNERMultiTask,
    "deberta_crf": DeBERTaCRF,
    "deberta_crf_multitask": DeBERTaCRFMultiTask,
}


def apply_bio_repair(pred_tags: list[str]) -> list[str]:
    """Fix invalid BIO sequences."""

    repaired = list(pred_tags)
    current_entity = None
    for index, tag in enumerate(repaired):
        if tag.startswith("B-"):
            current_entity = tag[2:]
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if current_entity != entity_type:
                repaired[index] = f"B-{entity_type}"
                current_entity = entity_type
        else:
            current_entity = None
    return repaired


def preprocess_tokens(
    tokens: list[str],
    enable_preprocessing: bool = False,
    preprocessing_config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING,
) -> list[str]:
    """Apply optional runtime preprocessing for standalone prediction jobs."""

    if not enable_preprocessing:
        return [str(token) for token in tokens]
    return runtime_preprocess_tokens(tokens, config=preprocessing_config)


def _prepare_inference_tokens(
    tokens: list[str],
    enable_preprocessing: bool = False,
    preprocessing_config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING,
) -> dict[str, object]:
    """Prepare model tokens while keeping alignment back to the original token list."""

    original_tokens = [str(token) for token in tokens]
    if not enable_preprocessing:
        return {
            "original_tokens": original_tokens,
            "model_tokens": list(original_tokens),
            "encoding_tokens": list(original_tokens),
            "kept_indices": list(range(len(original_tokens))),
        }

    original_tokens, model_tokens, kept_indices = runtime_preprocess_tokens_with_alignment(
        tokens,
        config=preprocessing_config,
    )
    encoding_tokens = model_tokens if model_tokens else ["[EMPTY]"]
    return {
        "original_tokens": original_tokens,
        "model_tokens": model_tokens,
        "encoding_tokens": encoding_tokens,
        "kept_indices": kept_indices,
    }


def _restore_word_probabilities(
    num_original_tokens: int,
    kept_indices: list[int],
    word_probs: np.ndarray,
) -> np.ndarray:
    """Restore skipped positions as one-hot O probabilities."""

    restored = np.zeros((num_original_tokens, NUM_LABELS), dtype=np.float32)
    restored[:, LABEL2ID["O"]] = 1.0

    for cleaned_index, original_index in enumerate(kept_indices):
        if cleaned_index >= word_probs.shape[0]:
            break
        if 0 <= original_index < num_original_tokens:
            restored[original_index] = word_probs[cleaned_index]

    return restored


def parse_list_col(value):
    """Parse list-literal CSV fields."""

    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value.strip())
    raise TypeError(f"Unsupported type for list column: {type(value)}")


def load_inference_rows(path: str | Path) -> list[dict[str, object]]:
    """Load a plain CSV with ID/tokens for inference."""

    csv_path = Path(path).resolve()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Inference CSV is empty: {csv_path}")

        required = {"ID", "tokens"}
        missing = required - set(reader.fieldnames)
        if missing:
            formatted = ", ".join(sorted(missing))
            raise ValueError(f"Inference CSV is missing required column(s): {formatted}")

        rows = []
        for row in reader:
            rows.append(
                {
                    "ID": row["ID"],
                    "tokens": parse_list_col(row["tokens"]),
                }
            )
    return rows


def load_prediction_rows(path: str | Path) -> list[dict[str, object]]:
    """Load a CSV for prediction, keeping gold tags when they are present."""

    csv_path = Path(path).resolve()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Prediction CSV is empty: {csv_path}")

        required = {"ID", "tokens"}
        missing = required - set(reader.fieldnames)
        if missing:
            formatted = ", ".join(sorted(missing))
            raise ValueError(f"Prediction CSV is missing required column(s): {formatted}")

        has_gold = "ner_tags" in reader.fieldnames
        rows = []
        for row in reader:
            record = {
                "ID": row["ID"],
                "tokens": parse_list_col(row["tokens"]),
            }
            if has_gold and row.get("ner_tags"):
                record["ner_tags"] = parse_list_col(row["ner_tags"])
            rows.append(record)
    return rows


def load_training_rows(
    path: str | Path,
    enable_preprocessing: bool = False,
    preprocessing_config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING,
) -> list[dict[str, object]]:
    """Load training rows for rebuilding the BiLSTM vocabulary."""

    csv_path = Path(path).resolve()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Training CSV is empty: {csv_path}")

        rows = []
        for row in reader:
            prepared = _prepare_inference_tokens(
                parse_list_col(row["tokens"]),
                enable_preprocessing=enable_preprocessing,
                preprocessing_config=preprocessing_config,
            )
            rows.append(
                {
                    "ID": row["ID"],
                    "tokens": list(prepared["model_tokens"]),
                }
            )
    return rows


def write_submission(output_csv: str | Path, ids: list[str], predictions: list[list[str]]) -> Path:
    """Write predictions in sample-submission format."""

    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "predicted_ner_tags"])
        writer.writeheader()
        for sample_id, pred_tags in zip(ids, predictions):
            writer.writerow(
                {
                    "ID": sample_id,
                    "predicted_ner_tags": str(pred_tags),
                }
            )

    return output_path


class InferenceNERDataset(Dataset):
    """Transformer inference dataset without label requirements."""

    def __init__(
        self,
        rows: list[dict[str, object]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        definition_prompting: bool = False,
        enable_preprocessing: bool = False,
        preprocessing_config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING,
    ) -> None:
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.definition_prompting = definition_prompting

        for row in rows:
            prepared = _prepare_inference_tokens(
                list(row["tokens"]),
                enable_preprocessing=enable_preprocessing,
                preprocessing_config=preprocessing_config,
            )
            model_tokens = list(prepared["model_tokens"])
            encoding_tokens = list(prepared["encoding_tokens"])
            original_tokens = list(prepared["original_tokens"])
            kept_indices = list(prepared["kept_indices"])
            if definition_prompting:
                encoding = self._encode_with_definition(encoding_tokens)
            else:
                encoding = self._encode(encoding_tokens)

            encoding["id"] = str(row["ID"])
            encoding["raw_tokens"] = original_tokens
            encoding["model_tokens"] = model_tokens
            encoding["kept_indices"] = kept_indices
            encoding["raw_tags"] = ["O"] * len(original_tokens)
            self.samples.append(encoding)

    def _encode(self, tokens: list[str]) -> dict[str, object]:
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "word_ids": encoding.word_ids(batch_index=0),
        }

    def _encode_with_definition(self, tokens: list[str]) -> dict[str, object]:
        def_tokens = self.tokenizer.tokenize(ENTITY_DEFINITION_PREFIX)
        text_encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length - len(def_tokens) - 2,
            add_special_tokens=False,
        )

        def_input_ids = self.tokenizer.convert_tokens_to_ids(def_tokens)
        cls_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        input_ids = [cls_id] + def_input_ids + [sep_id] + text_encoding["input_ids"] + [sep_id]
        def_len = 1 + len(def_input_ids) + 1
        pad_len = self.max_length - len(input_ids)

        input_ids = input_ids + [self.tokenizer.pad_token_id or 0] * pad_len
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len
        word_ids = [None] * def_len + text_encoding.word_ids(batch_index=0) + [None] * (1 + pad_len)

        return {
            "input_ids": torch.tensor(input_ids[: self.max_length], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[: self.max_length], dtype=torch.long),
            "word_ids": word_ids[: self.max_length],
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
        }

    def get_full_sample(self, index: int) -> dict[str, object]:
        return self.samples[index]


class InferenceSentenceDataset(Dataset):
    """Sentence-level inference dataset for the hierarchical classifier."""

    def __init__(
        self,
        rows: list[dict[str, object]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        enable_preprocessing: bool = False,
        preprocessing_config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING,
    ) -> None:
        self.samples: list[dict[str, object]] = []

        for row in rows:
            prepared = _prepare_inference_tokens(
                list(row["tokens"]),
                enable_preprocessing=enable_preprocessing,
                preprocessing_config=preprocessing_config,
            )
            encoding_tokens = list(prepared["encoding_tokens"])
            encoding = tokenizer(
                encoding_tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.samples.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "tokens": list(prepared["original_tokens"]),
                    "model_tokens": list(prepared["model_tokens"]),
                    "kept_indices": list(prepared["kept_indices"]),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
        }


class InferenceBiLSTMDataset(Dataset):
    """BiLSTM-CRF inference dataset without label requirements."""

    def __init__(
        self,
        rows: list[dict[str, object]],
        word2idx: dict[str, int],
        max_length: int = 256,
        enable_preprocessing: bool = False,
        preprocessing_config: RuntimePreprocessingConfig = DEFAULT_RUNTIME_PREPROCESSING,
    ) -> None:
        self.samples = []
        self.max_length = max_length

        for row in rows:
            prepared = _prepare_inference_tokens(
                list(row["tokens"]),
                enable_preprocessing=enable_preprocessing,
                preprocessing_config=preprocessing_config,
            )
            model_tokens = list(prepared["model_tokens"])
            encoding_tokens = list(prepared["encoding_tokens"])
            token_ids = [word2idx.get(token.lower(), word2idx.get("<UNK>", 1)) for token in encoding_tokens]
            length = min(len(model_tokens), max_length)
            padded_ids = token_ids[:max_length] + [0] * max(0, max_length - len(token_ids[:max_length]))
            self.samples.append(
                {
                    "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                    "length": length,
                    "id": str(row["ID"]),
                    "raw_tokens": list(prepared["original_tokens"])[:max_length],
                    "model_tokens": model_tokens[:max_length],
                    "kept_indices": [index for index in prepared["kept_indices"] if index < max_length],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample = self.samples[index]
        return {
            "input_ids": sample["input_ids"],
            "length": sample["length"],
        }


def build_vocab(rows: list[dict[str, object]], min_freq: int = 1) -> dict[str, int]:
    """Rebuild the word vocabulary used by the BiLSTM-CRF baseline."""

    from collections import Counter

    counter = Counter()
    for row in rows:
        for token in row["tokens"]:
            counter[str(token).lower()] += 1

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)
    return word2idx


def load_glove_embeddings(glove_path: str | Path, word2idx: dict[str, int], dim: int = 300) -> torch.Tensor:
    """Load GloVe vectors for the BiLSTM-CRF baseline."""

    import numpy as np

    embeddings = np.random.normal(0, 0.1, (len(word2idx), dim)).astype(np.float32)
    embeddings[0] = 0

    glove_file = Path(glove_path).resolve()
    found = 0
    with glove_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"GloVe: found {found}/{len(word2idx)} words")
    return torch.tensor(embeddings)


def _normalize_window_overlap(window_overlap: int | None) -> int:
    if window_overlap is None:
        return DEFAULT_WINDOW_OVERLAP
    return max(0, int(window_overlap))


def _count_transformer_input_tokens(
    tokenizer: AutoTokenizer,
    tokens: list[str],
    definition_prompting: bool = False,
) -> int:
    if definition_prompting:
        def_tokens = tokenizer.tokenize(ENTITY_DEFINITION_PREFIX)
        text_encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=False,
            add_special_tokens=False,
        )
        return 1 + len(def_tokens) + 1 + len(text_encoding["input_ids"]) + 1

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=False,
    )
    return len(encoding["input_ids"])


def _split_token_windows(
    tokens: list[str],
    max_items: int,
    overlap: int,
    count_fn,
    window_size: int | None = None,
) -> list[tuple[int, int]]:
    if not tokens:
        return []

    normalized_overlap = max(0, int(overlap))
    max_window_items = max(1, int(window_size)) if window_size is not None else max_items
    windows: list[tuple[int, int]] = []
    start = 0

    while start < len(tokens):
        upper = min(len(tokens), start + max_window_items)
        low = start + 1
        best_end = start + 1

        while low <= upper:
            mid = (low + upper) // 2
            if count_fn(tokens[start:mid]) <= max_items:
                best_end = mid
                low = mid + 1
            else:
                upper = mid - 1

        windows.append((start, best_end))
        if best_end >= len(tokens):
            break

        next_start = max(best_end - normalized_overlap, start + 1)
        start = next_start

    return windows


def _encode_transformer_window(
    tokenizer: AutoTokenizer,
    tokens: list[str],
    max_length: int = 512,
    definition_prompting: bool = False,
) -> dict[str, object]:
    if definition_prompting:
        def_tokens = tokenizer.tokenize(ENTITY_DEFINITION_PREFIX)
        text_encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length - len(def_tokens) - 2,
            add_special_tokens=False,
        )

        def_input_ids = tokenizer.convert_tokens_to_ids(def_tokens)
        cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id
        sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id

        input_ids = [cls_id] + def_input_ids + [sep_id] + text_encoding["input_ids"] + [sep_id]
        def_len = 1 + len(def_input_ids) + 1
        pad_len = max_length - len(input_ids)

        input_ids = input_ids + [tokenizer.pad_token_id or 0] * pad_len
        attention_mask = [1] * (max_length - pad_len) + [0] * pad_len
        word_ids = [None] * def_len + text_encoding.word_ids(batch_index=0) + [None] * (1 + pad_len)

        return {
            "input_ids": torch.tensor(input_ids[:max_length], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:max_length], dtype=torch.long),
            "word_ids": word_ids[:max_length],
        }

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "word_ids": encoding.word_ids(batch_index=0),
    }


def _average_window_probabilities(
    num_tokens: int,
    window_probabilities: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    averaged = np.zeros((num_tokens, NUM_LABELS), dtype=np.float32)
    counts = np.zeros((num_tokens, 1), dtype=np.float32)

    for start, end, probs in window_probabilities:
        averaged[start:end] += probs
        counts[start:end] += 1.0

    covered = counts.squeeze(-1) > 0
    averaged[covered] /= counts[covered]
    averaged[~covered, LABEL2ID["O"]] = 1.0
    return averaged


def _load_transformer_model_for_inference(
    checkpoint_path: str | Path,
    model_type: str,
    model_name: str,
    device: str,
    metadata: dict[str, object] | None = None,
):
    metadata = metadata or {}
    model_cls = MODEL_TYPES[model_type]
    if model_type == "deberta_crf":
        model = model_cls(
            model_name=model_name,
            num_labels=NUM_LABELS,
            use_lstm=bool(metadata.get("use_lstm", False)),
        )
    else:
        model = model_cls(model_name=model_name, num_labels=NUM_LABELS)

    model.load_state_dict(torch.load(Path(checkpoint_path).resolve(), map_location="cpu"))
    model = model.to(device)
    model.eval()
    return model


def _predict_transformer_sample_probs(
    model,
    tokenizer: AutoTokenizer,
    prepared: dict[str, object],
    device: str,
    batch_size: int,
    max_length: int = 512,
    definition_prompting: bool = False,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> np.ndarray:
    original_tokens = list(prepared["original_tokens"])
    model_tokens = list(prepared["model_tokens"])
    kept_indices = list(prepared["kept_indices"])

    if not model_tokens:
        return _restore_word_probabilities(len(original_tokens), kept_indices, np.zeros((0, NUM_LABELS), dtype=np.float32))

    overlap = _normalize_window_overlap(window_overlap)
    windows = _split_token_windows(
        model_tokens,
        max_items=max_length,
        overlap=overlap,
        count_fn=lambda chunk: _count_transformer_input_tokens(tokenizer, chunk, definition_prompting),
        window_size=window_size,
    )

    window_records: list[dict[str, object]] = []
    for start, end in windows:
        encoding = _encode_transformer_window(
            tokenizer,
            model_tokens[start:end],
            max_length=max_length,
            definition_prompting=definition_prompting,
        )
        window_records.append(
            {
                "start": start,
                "end": end,
                "num_words": end - start,
                **encoding,
            }
        )

    window_probabilities: list[tuple[int, int, np.ndarray]] = []
    with torch.no_grad():
        for batch_start in range(0, len(window_records), batch_size):
            batch_records = window_records[batch_start : batch_start + batch_size]
            input_ids = torch.stack([record["input_ids"] for record in batch_records]).to(device)
            attention_mask = torch.stack([record["attention_mask"] for record in batch_records]).to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            token_scores = outputs["logits"].detach().cpu().float().numpy()

            for batch_index, record in enumerate(batch_records):
                word_scores = _first_subword_scores(
                    token_scores[batch_index],
                    list(record["word_ids"]),
                    int(record["num_words"]),
                )
                window_probabilities.append(
                    (
                        int(record["start"]),
                        int(record["end"]),
                        _softmax(word_scores),
                    )
                )

    word_probabilities = _average_window_probabilities(len(model_tokens), window_probabilities)
    return _restore_word_probabilities(len(original_tokens), kept_indices, word_probabilities)


def _predict_bilstm_sample_probs(
    model,
    word2idx: dict[str, int],
    prepared: dict[str, object],
    device: str,
    batch_size: int,
    max_length: int = 256,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> np.ndarray:
    original_tokens = list(prepared["original_tokens"])
    model_tokens = list(prepared["model_tokens"])
    kept_indices = list(prepared["kept_indices"])

    if not model_tokens:
        return _restore_word_probabilities(len(original_tokens), kept_indices, np.zeros((0, NUM_LABELS), dtype=np.float32))

    overlap = _normalize_window_overlap(window_overlap)
    windows = _split_token_windows(
        model_tokens,
        max_items=max_length,
        overlap=overlap,
        count_fn=len,
        window_size=window_size,
    )

    window_records: list[dict[str, object]] = []
    for start, end in windows:
        window_tokens = model_tokens[start:end]
        token_ids = [word2idx.get(token.lower(), word2idx.get("<UNK>", 1)) for token in window_tokens]
        padded_ids = token_ids + [0] * max(0, max_length - len(token_ids))
        window_records.append(
            {
                "start": start,
                "end": end,
                "length": len(window_tokens),
                "input_ids": torch.tensor(padded_ids[:max_length], dtype=torch.long),
            }
        )

    window_probabilities: list[tuple[int, int, np.ndarray]] = []
    with torch.no_grad():
        for batch_start in range(0, len(window_records), batch_size):
            batch_records = window_records[batch_start : batch_start + batch_size]
            input_ids = torch.stack([record["input_ids"] for record in batch_records]).to(device)
            outputs = model(input_ids)
            emissions = outputs["emissions"].detach().cpu().float().numpy()

            for batch_index, record in enumerate(batch_records):
                length = int(record["length"])
                window_probabilities.append(
                    (
                        int(record["start"]),
                        int(record["end"]),
                        _softmax(emissions[batch_index][:length]),
                    )
                )

    word_probabilities = _average_window_probabilities(len(model_tokens), window_probabilities)
    return _restore_word_probabilities(len(original_tokens), kept_indices, word_probabilities)


def _entities_to_token_spans(tokens: list[str], entities: list[dict[str, object]]) -> list[tuple[int, int, str, float]]:
    char_to_token: dict[int, int] = {}
    char_pos = 0
    for token_index, token in enumerate(tokens):
        for _ in token:
            char_to_token[char_pos] = token_index
            char_pos += 1
        char_to_token[char_pos] = token_index
        char_pos += 1

    spans: list[tuple[int, int, str, float]] = []
    for entity in entities:
        label = str(entity.get("label", ""))
        if label not in {"ClinicalImpacts", "SocialImpacts"}:
            continue

        start_char = int(entity.get("start", -1))
        end_char = int(entity.get("end", -1))
        start_token = char_to_token.get(start_char)
        end_token = char_to_token.get(end_char - 1)
        if start_token is None or end_token is None or start_token > end_token:
            continue

        spans.append((start_token, end_token, label, float(entity.get("score", 0.0))))

    return spans


def _render_scored_token_spans(num_tokens: int, spans: list[tuple[int, int, str, float]]) -> list[str]:
    deduped: dict[tuple[int, int, str], float] = {}
    for start, end, label, score in spans:
        key = (start, end, label)
        deduped[key] = max(score, deduped.get(key, float("-inf")))

    occupied = [False] * num_tokens
    tags = ["O"] * num_tokens
    ranked_spans = sorted(
        [(start, end, label, score) for (start, end, label), score in deduped.items()],
        key=lambda item: (-item[3], item[0], item[1], item[2]),
    )

    for start, end, label, _ in ranked_spans:
        if start < 0 or end >= num_tokens or start > end:
            continue
        if any(occupied[position] for position in range(start, end + 1)):
            continue

        tags[start] = f"B-{label}"
        for position in range(start + 1, end + 1):
            tags[position] = f"I-{label}"
        for position in range(start, end + 1):
            occupied[position] = True

    return apply_bio_repair(tags)


def _predict_gliner_sample_tags(
    model,
    prepared: dict[str, object],
    threshold: float,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> list[str]:
    original_tokens = list(prepared["original_tokens"])
    model_tokens = list(prepared["model_tokens"])
    kept_indices = list(prepared["kept_indices"])

    if not model_tokens:
        return restore_forced_o_predictions(original_tokens, kept_indices, [])

    from .gliner_finetune import ENTITY_LABELS

    max_length = int(getattr(model.config, "max_len", 384))
    overlap = _normalize_window_overlap(window_overlap)
    windows = _split_token_windows(
        model_tokens,
        max_items=max_length,
        overlap=overlap,
        count_fn=len,
        window_size=window_size,
    )

    merged_spans: list[tuple[int, int, str, float]] = []
    for start, end in windows:
        try:
            window_entities = model.predict_entities(
                " ".join(model_tokens[start:end]),
                ENTITY_LABELS,
                threshold=threshold,
            )
        except Exception:
            window_entities = []

        for local_start, local_end, label, score in _entities_to_token_spans(model_tokens[start:end], window_entities):
            merged_spans.append((start + local_start, start + local_end, label, score))

    cleaned_predictions = _render_scored_token_spans(len(model_tokens), merged_spans)
    return restore_forced_o_predictions(
        original_tokens,
        kept_indices,
        cleaned_predictions,
    )


def _predict_sentence_labels_with_windowing(
    model,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, object]],
    batch_size: int,
    device: str,
    threshold: float,
    enable_preprocessing: bool = False,
    max_length: int = 512,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> np.ndarray:
    overlap = _normalize_window_overlap(window_overlap)
    all_predictions: list[np.ndarray] = []

    with torch.no_grad():
        for row in rows:
            prepared = _prepare_inference_tokens(
                list(row["tokens"]),
                enable_preprocessing=enable_preprocessing,
            )
            model_tokens = list(prepared["model_tokens"])
            if not model_tokens:
                all_predictions.append(np.zeros(2, dtype=np.int64))
                continue

            windows = _split_token_windows(
                model_tokens,
                max_items=max_length,
                overlap=overlap,
                count_fn=lambda chunk: _count_transformer_input_tokens(tokenizer, chunk, False),
                window_size=window_size,
            )
            window_records = [
                _encode_transformer_window(tokenizer, model_tokens[start:end], max_length=max_length)
                for start, end in windows
            ]

            sentence_probs = np.zeros(2, dtype=np.float32)
            for batch_start in range(0, len(window_records), batch_size):
                batch_records = window_records[batch_start : batch_start + batch_size]
                input_ids = torch.stack([record["input_ids"] for record in batch_records]).to(device)
                attention_mask = torch.stack([record["attention_mask"] for record in batch_records]).to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                probs = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
                sentence_probs = np.maximum(sentence_probs, probs.max(axis=0))

            all_predictions.append((sentence_probs >= threshold).astype(np.int64))

    return np.stack(all_predictions) if all_predictions else np.zeros((0, 2), dtype=np.int64)


def _decode_dataset_predictions(dataset: InferenceNERDataset, batch_predictions: list[torch.Tensor]) -> list[list[str]]:
    """Map model token predictions back to word-level BIO tags."""

    decoded: list[list[str]] = []
    sample_index = 0

    for prediction_batch in batch_predictions:
        pred_array = prediction_batch.cpu().numpy()
        for batch_index in range(pred_array.shape[0]):
            if sample_index >= len(dataset.samples):
                break

            sample = dataset.get_full_sample(sample_index)
            word_preds: dict[int, str] = {}
            for token_index, word_id in enumerate(sample["word_ids"]):
                if word_id is not None and word_id not in word_preds:
                    word_preds[word_id] = ID2LABEL[int(pred_array[batch_index][token_index])]

            cleaned_pred_tags = [
                word_preds.get(word_index, "O")
                for word_index in range(len(sample["kept_indices"]))
            ]
            restored_pred_tags = restore_forced_o_predictions(
                sample["raw_tokens"],
                sample["kept_indices"],
                cleaned_pred_tags,
            )
            decoded.append(apply_bio_repair(restored_pred_tags))
            sample_index += 1

    return decoded


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def _first_subword_scores(token_scores: np.ndarray, word_ids: list[int | None], num_words: int) -> np.ndarray:
    word_scores: dict[int, np.ndarray] = {}
    for token_index, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_scores and word_id < num_words:
            word_scores[word_id] = token_scores[token_index]

    return np.stack(
        [
            word_scores.get(word_index, np.zeros(NUM_LABELS, dtype=np.float32))
            for word_index in range(num_words)
        ]
    ).astype(np.float32)


def _decode_single_candidate_probs(probabilities: list[np.ndarray]) -> list[list[str]]:
    predictions: list[list[str]] = []
    for sample_probs in probabilities:
        pred_ids = sample_probs.argmax(axis=-1)
        pred_tags = [ID2LABEL[int(pred_id)] for pred_id in pred_ids]
        predictions.append(apply_bio_repair(pred_tags))
    return predictions


def _decode_average_probs(probability_sets: list[list[np.ndarray]]) -> list[list[str]]:
    combined_predictions: list[list[str]] = []
    num_samples = len(probability_sets[0])

    for sample_index in range(num_samples):
        avg_probs = np.mean([candidate_probs[sample_index] for candidate_probs in probability_sets], axis=0)
        pred_ids = avg_probs.argmax(axis=-1)
        pred_tags = [ID2LABEL[int(pred_id)] for pred_id in pred_ids]
        combined_predictions.append(apply_bio_repair(pred_tags))

    return combined_predictions


def _majority_vote_predictions(prediction_sets: list[list[list[str]]]) -> list[list[str]]:
    combined_predictions: list[list[str]] = []
    num_samples = len(prediction_sets[0])

    for sample_index in range(num_samples):
        sample_predictions = [predictions[sample_index] for predictions in prediction_sets]
        sequence_length = max(len(prediction) for prediction in sample_predictions)
        voted_tags: list[str] = []

        for token_index in range(sequence_length):
            vote_counts: dict[str, int] = {}
            for prediction in sample_predictions:
                tag = prediction[token_index] if token_index < len(prediction) else "O"
                vote_counts[tag] = vote_counts.get(tag, 0) + 1
            selected_tag = max(
                vote_counts.items(),
                key=lambda item: (item[1], item[0] != "O", item[0]),
            )[0]
            voted_tags.append(selected_tag)

        combined_predictions.append(apply_bio_repair(voted_tags))

    return combined_predictions


def _collect_transformer_test_probs(
    candidate: dict[str, object],
    rows: list[dict[str, object]],
    batch_size: int,
    device: str,
    enable_preprocessing: bool = False,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> list[np.ndarray]:
    model_type = str(candidate["model_type"])
    model_name = str(candidate.get("model_name") or "microsoft/deberta-v3-large")
    metadata = dict(candidate.get("metadata") or {})
    definition_prompting = bool(metadata.get("definition_prompting", False))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_transformer_model_for_inference(
        checkpoint_path=str(candidate["path"]),
        model_type=model_type,
        model_name=model_name,
        device=device,
        metadata=metadata,
    )

    all_probs: list[np.ndarray] = []
    for row in rows:
        prepared = _prepare_inference_tokens(
            list(row["tokens"]),
            enable_preprocessing=enable_preprocessing,
        )
        all_probs.append(
            _predict_transformer_sample_probs(
                model,
                tokenizer,
                prepared,
                device=device,
                batch_size=batch_size,
                max_length=512,
                definition_prompting=definition_prompting,
                window_overlap=window_overlap,
                window_size=window_size,
            )
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_probs


def _collect_bilstm_test_probs(
    candidate: dict[str, object],
    rows: list[dict[str, object]],
    batch_size: int,
    device: str,
    data_dir: str | Path,
    glove_path: str | Path,
    enable_preprocessing: bool = False,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> list[np.ndarray]:
    train_rows = load_training_rows(
        Path(data_dir).resolve() / "new_train_data.csv",
        enable_preprocessing=enable_preprocessing,
    )
    word2idx = build_vocab(train_rows)
    embeddings = load_glove_embeddings(glove_path, word2idx, dim=300)

    model = BiLSTMCRF(
        vocab_size=len(word2idx),
        embedding_dim=300,
        hidden_dim=256,
        num_tags=NUM_LABELS,
        num_layers=2,
        dropout=0.5,
        pretrained_embeddings=embeddings,
    )
    model.load_state_dict(torch.load(Path(str(candidate["path"])).resolve(), map_location="cpu"))
    model = model.to(device)
    model.eval()

    all_probs: list[np.ndarray] = []
    for row in rows:
        prepared = _prepare_inference_tokens(
            list(row["tokens"]),
            enable_preprocessing=enable_preprocessing,
        )
        all_probs.append(
            _predict_bilstm_sample_probs(
                model,
                word2idx,
                prepared,
                device=device,
                batch_size=batch_size,
                max_length=256,
                window_overlap=window_overlap,
                window_size=window_size,
            )
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_probs


def _predict_sentence_labels_inference(
    model,
    dataset: InferenceSentenceDataset,
    batch_size: int,
    device: str,
    threshold: float,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_preds.append((probs >= threshold).astype(np.int64))

    return np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 2), dtype=np.int64)


def _collect_hierarchical_test_predictions(
    candidate: dict[str, object],
    rows: list[dict[str, object]],
    batch_size: int,
    device: str,
    enable_preprocessing: bool = False,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> list[list[str]]:
    from .hierarchical import (
        _load_sentence_classifier,
        _mask_predicted_tags,
    )

    model_name = str(candidate.get("model_name") or "microsoft/deberta-v3-large")
    metadata = dict(candidate.get("metadata") or {})
    threshold = float(metadata.get("threshold", 0.5))
    classifier_checkpoint = Path(str(metadata["classifier_checkpoint"])).resolve()
    ner_checkpoint = Path(str(metadata["ner_checkpoint"])).resolve()

    sentence_tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier_model = _load_sentence_classifier(classifier_checkpoint, model_name, device)
    sentence_preds = _predict_sentence_labels_with_windowing(
        classifier_model,
        sentence_tokenizer,
        rows,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
        enable_preprocessing=enable_preprocessing,
        window_overlap=window_overlap,
        window_size=window_size,
    )

    prepared_rows = [
        _prepare_inference_tokens(
            list(row["tokens"]),
            enable_preprocessing=enable_preprocessing,
        )
        for row in rows
    ]
    all_predictions = [["O"] * len(prepared["original_tokens"]) for prepared in prepared_rows]

    positive_indices = [
        index
        for index, pred in enumerate(sentence_preds)
        if int(pred.sum()) > 0 and prepared_rows[index]["kept_indices"]
    ]
    if positive_indices:
        ner_model = _load_transformer_model_for_inference(
            checkpoint_path=ner_checkpoint,
            model_type="deberta_multitask",
            model_name=model_name,
            device=device,
        )
        ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
        for sample_index in positive_indices:
            sample_pred = _decode_single_candidate_probs(
                [
                    _predict_transformer_sample_probs(
                        ner_model,
                        ner_tokenizer,
                        prepared_rows[sample_index],
                        device=device,
                        batch_size=batch_size,
                        max_length=512,
                        definition_prompting=False,
                        window_overlap=window_overlap,
                        window_size=window_size,
                    )
                ]
            )[0]
            allow_clinical = bool(sentence_preds[sample_index][0])
            allow_social = bool(sentence_preds[sample_index][1])
            all_predictions[sample_index] = _mask_predicted_tags(sample_pred, allow_clinical, allow_social)
        del ner_model

    del classifier_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_predictions


def _resolve_ensemble_selection(
    search_results_path: str | Path,
    best_size: int | None = None,
    selection_metric: str = "relaxed_f1",
) -> tuple[dict[str, object], dict[str, object]]:
    results_path = Path(search_results_path).resolve()
    with results_path.open("r", encoding="utf-8") as handle:
        search_results = json.load(handle)

    if selection_metric not in {"relaxed_f1", "strict_f1"}:
        raise ValueError("selection_metric must be either 'relaxed_f1' or 'strict_f1'.")

    if best_size is None:
        selection_name = f"best_overall_by_{selection_metric}"
        saved_selection = search_results.get(selection_name)
        if saved_selection is not None:
            selection = dict(saved_selection)
        else:
            all_results = search_results.get("all_results", [])
            if not all_results:
                raise ValueError(f"No all_results entries found in {results_path}.")
            selection = dict(
                max(
                    all_results,
                    key=lambda record: float(record.get(selection_metric, float("-inf"))),
                )
            )
    else:
        target_size = int(best_size)
        selection_name = f"best_size_{target_size}_by_{selection_metric}"
        saved_by_size = search_results.get(f"best_by_size_by_{selection_metric}", {})
        saved_selection = saved_by_size.get(str(target_size))
        if saved_selection is not None:
            selection = dict(saved_selection)
        else:
            all_results = search_results.get("all_results", [])
            if not all_results:
                raise ValueError(f"No all_results entries found in {results_path}.")
            candidates = [record for record in all_results if int(record.get("num_models", -1)) == target_size]
            if not candidates:
                raise ValueError(
                    f"No ensemble combinations of size {target_size} were found in {results_path}."
                )
            selection = dict(
                max(
                    candidates,
                    key=lambda record: float(record.get(selection_metric, float("-inf"))),
                )
            )

    candidates_by_name = {
        candidate["name"]: candidate
        for candidate in search_results.get("candidates", [])
    }
    selected_candidates = []
    missing = []
    for name in selection["models"]:
        candidate = candidates_by_name.get(name)
        if candidate is None:
            missing.append(name)
        else:
            selected_candidates.append(candidate)

    if missing:
        formatted = ", ".join(missing)
        raise ValueError(f"The search results file is missing candidate metadata for: {formatted}")

    return {
        "results_path": str(results_path),
        "selection_name": selection_name,
        "vote_method": selection["vote_method"],
        "selected_record": selection,
        "selected_candidates": selected_candidates,
    }, search_results


def predict_transformer_ner(
    input_csv: str | Path,
    output_csv: str | Path,
    checkpoint_path: str | Path,
    model_type: str = "deberta",
    model_name: str = "microsoft/deberta-v3-large",
    batch_size: int = 8,
    device: str = "cuda:0",
    definition_prompting: bool = False,
    enable_preprocessing: bool = False,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> dict[str, str | int]:
    """Run a saved transformer NER checkpoint on a CSV and write sample-submission output."""

    if model_type not in MODEL_TYPES:
        supported = ", ".join(sorted(MODEL_TYPES))
        raise ValueError(f"Unsupported model_type '{model_type}'. Supported values: {supported}")

    input_path = Path(input_csv).resolve()
    output_path = Path(output_csv).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("\n" + "=" * 60)
    print("  Prediction Job")
    print("=" * 60)
    print(f"  Input CSV: {input_path}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Model type: {model_type}")
    print(f"  Model name: {model_name}")
    print(f"  Device: {device}")
    print(f"  Output CSV: {output_path}")
    print(f"  Long-window overlap: {_normalize_window_overlap(window_overlap)}")
    if window_size is not None:
        print(f"  Long-window size cap: {window_size}")
    print("=" * 60 + "\n")

    rows = load_inference_rows(input_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_transformer_model_for_inference(
        checkpoint_path=checkpoint,
        model_type=model_type,
        model_name=model_name,
        device=device,
        metadata={"use_lstm": False},
    )
    decoded_predictions = _decode_single_candidate_probs(
        [
            _predict_transformer_sample_probs(
                model,
                tokenizer,
                _prepare_inference_tokens(
                    list(row["tokens"]),
                    enable_preprocessing=enable_preprocessing,
                ),
                device=device,
                batch_size=batch_size,
                max_length=512,
                definition_prompting=definition_prompting,
                window_overlap=window_overlap,
                window_size=window_size,
            )
            for row in rows
        ]
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path = write_submission(output_path, [str(row["ID"]) for row in rows], decoded_predictions)
    print(f"Saved predictions to {output_path}")
    return {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "checkpoint_path": str(checkpoint),
        "rows": len(decoded_predictions),
    }


def predict_bilstm_crf(
    input_csv: str | Path,
    output_csv: str | Path,
    checkpoint_path: str | Path,
    data_dir: str | Path,
    glove_path: str | Path,
    batch_size: int = 32,
    device: str = "cuda:0",
    enable_preprocessing: bool = False,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> dict[str, str | int]:
    """Run a saved BiLSTM-CRF checkpoint on a CSV and write sample-submission output."""

    input_path = Path(input_csv).resolve()
    output_path = Path(output_csv).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    train_rows = load_training_rows(
        Path(data_dir).resolve() / "new_train_data.csv",
        enable_preprocessing=enable_preprocessing,
    )
    word2idx = build_vocab(train_rows)
    embeddings = load_glove_embeddings(glove_path, word2idx, dim=300)

    rows = load_inference_rows(input_path)

    model = BiLSTMCRF(
        vocab_size=len(word2idx),
        embedding_dim=300,
        hidden_dim=256,
        num_tags=NUM_LABELS,
        num_layers=2,
        dropout=0.5,
        pretrained_embeddings=embeddings,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    decoded_predictions = _decode_single_candidate_probs(
        [
            _predict_bilstm_sample_probs(
                model,
                word2idx,
                _prepare_inference_tokens(
                    list(row["tokens"]),
                    enable_preprocessing=enable_preprocessing,
                ),
                device=device,
                batch_size=batch_size,
                max_length=256,
                window_overlap=window_overlap,
                window_size=window_size,
            )
            for row in rows
        ]
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path = write_submission(output_path, [str(row["ID"]) for row in rows], decoded_predictions)
    print(f"Saved predictions to {output_path}")
    return {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "checkpoint_path": str(checkpoint),
        "rows": len(decoded_predictions),
    }


def predict_ensemble_from_search_results(
    input_csv: str | Path,
    output_csv: str | Path,
    search_results_path: str | Path,
    best_size: int | None = None,
    selection_metric: str = "relaxed_f1",
    batch_size: int = 8,
    device: str = "cuda:0",
    data_dir: str | Path = ".",
    glove_path: str | Path = "glove.6B.300d.txt",
    enable_preprocessing: bool = False,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> dict[str, str | int | float | list[str]]:
    """Run a saved ensemble-search selection on a CSV and write sample-submission output."""

    resolved, _ = _resolve_ensemble_selection(
        search_results_path,
        best_size=best_size,
        selection_metric=selection_metric,
    )
    selected_record = dict(resolved["selected_record"])
    selected_candidates = list(resolved["selected_candidates"])
    vote_method = str(resolved["vote_method"])

    input_path = Path(input_csv).resolve()
    output_path = Path(output_csv).resolve()
    rows = load_inference_rows(input_path)

    print("\n" + "=" * 60)
    print("  Ensemble Prediction Job")
    print("=" * 60)
    print(f"  Input CSV: {input_path}")
    print(f"  Search results: {resolved['results_path']}")
    print(f"  Selected entry: {resolved['selection_name']}")
    print(f"  Selection metric: {selection_metric}")
    print(f"  Vote method: {vote_method}")
    print(f"  Models: {', '.join(selected_record['models'])}")
    print(f"  Device: {device}")
    print(f"  Output CSV: {output_path}")
    print("=" * 60 + "\n")

    candidate_probabilities: dict[str, list[np.ndarray]] = {}
    candidate_predictions: dict[str, list[list[str]]] = {}

    for candidate in selected_candidates:
        model_type = str(candidate["model_type"])
        if model_type == "hierarchical_deberta":
            candidate_predictions[str(candidate["name"])] = _collect_hierarchical_test_predictions(
                candidate,
                rows,
                batch_size=batch_size,
                device=device,
                enable_preprocessing=enable_preprocessing,
                window_overlap=window_overlap,
                window_size=window_size,
            )
        elif model_type == "bilstm_crf":
            probs = _collect_bilstm_test_probs(
                candidate,
                rows,
                batch_size=batch_size,
                device=device,
                data_dir=data_dir,
                glove_path=glove_path,
                enable_preprocessing=enable_preprocessing,
                window_overlap=window_overlap,
                window_size=window_size,
            )
            if vote_method == "probability_average":
                candidate_probabilities[str(candidate["name"])] = probs
            else:
                candidate_predictions[str(candidate["name"])] = _decode_single_candidate_probs(probs)
        else:
            probs = _collect_transformer_test_probs(
                candidate,
                rows,
                batch_size=batch_size,
                device=device,
                enable_preprocessing=enable_preprocessing,
                window_overlap=window_overlap,
                window_size=window_size,
            )
            if vote_method == "probability_average":
                candidate_probabilities[str(candidate["name"])] = probs
            else:
                candidate_predictions[str(candidate["name"])] = _decode_single_candidate_probs(probs)

    if vote_method == "probability_average":
        predictions = _decode_average_probs(
            [candidate_probabilities[name] for name in selected_record["models"]]
        )
    elif vote_method == "majority_vote":
        predictions = _majority_vote_predictions(
            [candidate_predictions[name] for name in selected_record["models"]]
        )
    else:
        raise ValueError(f"Unsupported vote_method in search results: {vote_method}")

    output_path = write_submission(output_path, [str(row["ID"]) for row in rows], predictions)
    print(f"Saved ensemble predictions to {output_path}")
    return {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "search_results_path": str(Path(search_results_path).resolve()),
        "selection_name": str(resolved["selection_name"]),
        "selection_metric": selection_metric,
        "vote_method": vote_method,
        "rows": len(predictions),
        "models": list(selected_record["models"]),
        "dev_relaxed_f1": float(selected_record["relaxed_f1"]),
        "dev_strict_f1": float(selected_record["strict_f1"]),
    }


def predict_gliner_ner(
    input_csv: str | Path,
    output_csv: str | Path,
    model_dir: str | Path,
    threshold: float = 0.4,
    device: str = "cuda:0",
    metrics_output: str | Path | None = None,
    enable_preprocessing: bool = False,
    window_overlap: int | None = None,
    window_size: int | None = None,
) -> dict[str, object]:
    """Run a fine-tuned GLiNER model on a CSV and optionally score it when gold tags are present."""

    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError("gliner is not installed in the active environment.") from exc

    from .evaluation import evaluate_ner

    input_path = Path(input_csv).resolve()
    output_path = Path(output_csv).resolve()
    model_dir_path = Path(model_dir).resolve()

    if not model_dir_path.exists():
        raise FileNotFoundError(f"GLiNER model directory not found: {model_dir_path}")

    rows = load_prediction_rows(input_path)
    model = GLiNER.from_pretrained(str(model_dir_path))
    model = model.to(device)

    ids: list[str] = []
    predictions: list[list[str]] = []
    gold_tags: list[list[str]] = []
    has_gold = True

    print("\n" + "=" * 60)
    print("  GLiNER Prediction Job")
    print("=" * 60)
    print(f"  Input CSV: {input_path}")
    print(f"  Model dir: {model_dir_path}")
    print(f"  Threshold: {threshold}")
    print(f"  Device: {device}")
    print(f"  Output CSV: {output_path}")
    print(f"  Long-window overlap: {_normalize_window_overlap(window_overlap)}")
    if window_size is not None:
        print(f"  Long-window size cap: {window_size}")
    print("=" * 60 + "\n")

    for row in rows:
        prepared = _prepare_inference_tokens(
            list(row["tokens"]),
            enable_preprocessing=enable_preprocessing,
        )
        ids.append(str(row["ID"]))
        predictions.append(
            _predict_gliner_sample_tags(
                model,
                prepared,
                threshold=threshold,
                window_overlap=window_overlap,
                window_size=window_size,
            )
        )

        if "ner_tags" in row:
            gold_tags.append(list(row["ner_tags"]))
        else:
            has_gold = False

    output_path = write_submission(output_path, ids, predictions)
    result: dict[str, object] = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "model_dir": str(model_dir_path),
        "threshold": threshold,
        "device": device,
        "rows": len(predictions),
    }

    print(f"Saved GLiNER predictions to {output_path}")

    if has_gold and gold_tags:
        metrics = evaluate_ner(gold_tags, predictions, print_report=True)
        metrics_output_path = (
            Path(metrics_output).resolve()
            if metrics_output is not None
            else output_path.with_name(f"{output_path.stem}_metrics.json")
        )
        with metrics_output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        result["metrics_output"] = str(metrics_output_path)
        result["metrics"] = metrics
        print(f"Saved GLiNER metrics to {metrics_output_path}")

    return result
