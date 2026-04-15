"""Additional hierarchical / span-oriented impact extraction experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .checkpoints import TopKCheckpointManager
from .data import ID2LABEL, LABEL2ID, NERDataset, load_dataframe, preprocess_tokens
from .deberta_ner import DeBERTaNER
from .evaluation import evaluate_ner
from .gliner_finetune import finetune_gliner
from .train import (
    _clear_torch_memory,
    _early_stopping_status,
    _init_early_stopping,
    _maybe_save_strict_best_state_dict,
    _update_early_stopping,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2" / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-large"

BINARY_LABEL2ID = {"O": 0, "B-Impact": 1, "I-Impact": 2}
BINARY_ID2LABEL = {value: key for key, value in BINARY_LABEL2ID.items()}
SPAN_TYPE_LABEL2ID = {"ClinicalImpacts": 0, "SocialImpacts": 1}
SPAN_TYPE_ID2LABEL = {value: key for key, value in SPAN_TYPE_LABEL2ID.items()}


def _typed_spans_from_tags(tags: list[str]) -> list[tuple[int, int, str]]:
    """Extract typed spans from BIO tags, tolerating malformed I- starts."""

    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    current_type: str | None = None

    for index, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type is not None and start is not None:
                spans.append((start, index - 1, current_type))
            current_type = tag[2:]
            start = index
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if current_type != tag_type or start is None:
                if current_type is not None and start is not None:
                    spans.append((start, index - 1, current_type))
                current_type = tag_type
                start = index
        else:
            if current_type is not None and start is not None:
                spans.append((start, index - 1, current_type))
            current_type = None
            start = None

    if current_type is not None and start is not None:
        spans.append((start, len(tags) - 1, current_type))
    return spans


def _collapse_to_binary_tags(tags: list[str]) -> list[str]:
    return [
        "O"
        if tag == "O"
        else "B-Impact"
        if tag.startswith("B-")
        else "I-Impact"
        for tag in tags
    ]


def _repair_bio_tags(tags: list[str]) -> list[str]:
    """Repair invalid BIO transitions with the same label vocabulary."""

    repaired: list[str] = []
    active_type: str | None = None

    for tag in tags:
        if tag == "O":
            repaired.append("O")
            active_type = None
            continue

        prefix, entity_type = tag.split("-", maxsplit=1)
        if prefix == "B":
            repaired.append(tag)
            active_type = entity_type
            continue

        if active_type == entity_type:
            repaired.append(tag)
        else:
            repaired.append(f"B-{entity_type}")
            active_type = entity_type

    return repaired


def _exact_span_f1(gold_tags: list[list[str]], pred_tags: list[list[str]]) -> dict[str, float]:
    """Compute exact span F1 while ignoring entity types."""

    gold_spans = [set((start, end) for start, end, _ in _typed_spans_from_tags(tags)) for tags in gold_tags]
    pred_spans = [set((start, end) for start, end, _ in _typed_spans_from_tags(tags)) for tags in pred_tags]

    true_positive = sum(len(gold & pred) for gold, pred in zip(gold_spans, pred_spans))
    total_gold = sum(len(item) for item in gold_spans)
    total_pred = sum(len(item) for item in pred_spans)

    precision = true_positive / total_pred if total_pred else 0.0
    recall = true_positive / total_gold if total_gold else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"span_precision": precision, "span_recall": recall, "span_f1": f1}


def _render_typed_spans(num_tokens: int, spans: list[tuple[int, int, str]]) -> list[str]:
    tags = ["O"] * num_tokens
    for start, end, label in sorted(spans, key=lambda item: (item[0], item[1])):
        if start < 0 or end >= num_tokens or start > end:
            continue
        if any(tags[position] != "O" for position in range(start, end + 1)):
            continue
        tags[start] = f"B-{label}"
        for position in range(start + 1, end + 1):
            tags[position] = f"I-{label}"
    return tags


def _mask_typed_tags(pred_tags: list[str], allow_clinical: bool, allow_social: bool) -> list[str]:
    masked = []
    for tag in pred_tags:
        if "ClinicalImpacts" in tag and not allow_clinical:
            masked.append("O")
        elif "SocialImpacts" in tag and not allow_social:
            masked.append("O")
        else:
            masked.append(tag)
    return _repair_bio_tags(masked)


def _sentence_targets_from_labels(labels_batch: torch.Tensor) -> torch.Tensor:
    """Derive [has_clinical, has_social, no_impact] labels from token labels."""

    sentence_labels = []
    for labels in labels_batch:
        valid = labels[labels != -100]
        has_clinical = bool(((valid == LABEL2ID["B-ClinicalImpacts"]) | (valid == LABEL2ID["I-ClinicalImpacts"])).any())
        has_social = bool(((valid == LABEL2ID["B-SocialImpacts"]) | (valid == LABEL2ID["I-SocialImpacts"])).any())
        no_impact = not has_clinical and not has_social
        sentence_labels.append([float(has_clinical), float(has_social), float(no_impact)])
    return torch.tensor(sentence_labels, dtype=torch.float32, device=labels_batch.device)


def _multilabel_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    per_label = {}
    micro_tp = micro_fp = micro_fn = 0
    label_names = ("has_clinical", "has_social", "no_impact")

    for index, name in enumerate(label_names):
        gold_col = gold[:, index].astype(bool)
        pred_col = pred[:, index].astype(bool)
        tp = int(np.logical_and(gold_col, pred_col).sum())
        fp = int(np.logical_and(~gold_col, pred_col).sum())
        fn = int(np.logical_and(gold_col, ~pred_col).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        per_label[f"{name}_precision"] = precision
        per_label[f"{name}_recall"] = recall
        per_label[f"{name}_f1"] = f1
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    per_label["sentence_micro_precision"] = micro_precision
    per_label["sentence_micro_recall"] = micro_recall
    per_label["sentence_micro_f1"] = (
        (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    per_label["sentence_subset_accuracy"] = float(np.mean(np.all(gold == pred, axis=1)))
    return per_label


class BinaryImpactNERDataset(Dataset):
    """Token classification dataset for binary impact boundary detection."""

    def __init__(self, df, tokenizer, max_length: int = 512):
        self.samples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for _, row in df.iterrows():
            tokens = preprocess_tokens(row["tokens"])
            binary_tags = _collapse_to_binary_tags(row["ner_tags"])
            label_ids = [BINARY_LABEL2ID[tag] for tag in binary_tags]
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            word_ids = encoding.word_ids(batch_index=0)
            aligned_labels = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)
                elif word_id != previous_word_id:
                    aligned_labels.append(label_ids[word_id])
                else:
                    label_id = label_ids[word_id]
                    if BINARY_ID2LABEL[label_id].startswith("B-"):
                        aligned_labels.append(BINARY_LABEL2ID["I-Impact"])
                    else:
                        aligned_labels.append(label_id)
                previous_word_id = word_id

            self.samples.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "labels": torch.tensor(aligned_labels, dtype=torch.long),
                    "word_ids": word_ids,
                    "raw_tokens": tokens,
                    "raw_binary_tags": binary_tags,
                    "raw_typed_tags": row["ner_tags"],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }

    def get_full_sample(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class SpanTypeDataset(Dataset):
    """Sequence classification dataset for classifying extracted impact spans."""

    def __init__(self, df, tokenizer, max_length: int = 256):
        self.samples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for _, row in df.iterrows():
            tokens = preprocess_tokens(row["tokens"])
            spans = _typed_spans_from_tags(row["ner_tags"])
            for start, end, label in spans:
                marked_tokens = (
                    tokens[:start]
                    + ["[IMPACT_START]"]
                    + tokens[start : end + 1]
                    + ["[IMPACT_END]"]
                    + tokens[end + 1 :]
                )
                text = " ".join(marked_tokens)
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.samples.append(
                    {
                        "input_ids": encoding["input_ids"].squeeze(0),
                        "attention_mask": encoding["attention_mask"].squeeze(0),
                        "labels": torch.tensor(SPAN_TYPE_LABEL2ID[label], dtype=torch.long),
                        "tokens": tokens,
                        "span": (start, end),
                        "span_label": label,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }


class ImpactSpanClassifier(nn.Module):
    """Sentence classifier that predicts whether a marked span is clinical or social."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0].float())
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


class SentenceTokenHierarchyNER(nn.Module):
    """Joint sentence-level and token-level model with sentence-context biasing."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_token_labels: int = 5,
        num_sentence_labels: int = 3,
        dropout: float = 0.1,
        aux_weight: float = 0.3,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.sentence_head = nn.Linear(self.config.hidden_size, num_sentence_labels)
        self.sentence_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.token_classifier = nn.Linear(self.config.hidden_size, num_token_labels)
        self.aux_weight = aux_weight
        self.sentence_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, labels=None, sentence_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state.float()
        cls_output = self.dropout(sequence_output[:, 0])
        sentence_logits = self.sentence_head(cls_output)

        contextual_bias = torch.tanh(self.sentence_projection(cls_output)).unsqueeze(1)
        fused_sequence = self.dropout(sequence_output + contextual_bias)
        token_logits = self.token_classifier(fused_sequence)

        loss = None
        if labels is not None:
            token_loss = F.cross_entropy(
                token_logits.view(-1, token_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = token_loss
            if sentence_labels is not None:
                sentence_loss = self.sentence_loss_fn(sentence_logits, sentence_labels.float())
                loss = loss + self.aux_weight * sentence_loss

        return {"loss": loss, "logits": token_logits, "sentence_logits": sentence_logits}


def _evaluate_binary_extractor(model, dataset, dataloader, device: str) -> dict[str, float]:
    model.eval()
    all_gold: list[list[str]] = []
    all_pred: list[list[str]] = []
    total_loss = 0.0
    num_batches = 0
    sample_ptr = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
            )
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                num_batches += 1

            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            for batch_index in range(preds.shape[0]):
                sample = dataset.get_full_sample(sample_ptr)
                word_preds = {}
                for token_index, word_id in enumerate(sample["word_ids"]):
                    if word_id is not None and word_id not in word_preds:
                        word_preds[word_id] = BINARY_ID2LABEL[preds[batch_index][token_index]]
                pred_tags = [
                    word_preds.get(word_index, "O")
                    for word_index in range(len(sample["raw_binary_tags"]))
                ]
                pred_tags = _repair_bio_tags(pred_tags)
                all_gold.append(sample["raw_binary_tags"])
                all_pred.append(pred_tags)
                sample_ptr += 1

    metrics = _exact_span_f1(all_gold, all_pred)
    metrics["dev_loss"] = total_loss / num_batches if num_batches else float("nan")
    return metrics


def _evaluate_span_classifier(model, dataloader, device: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_gold = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
            )
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                num_batches += 1
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy().tolist()
            all_pred.extend(preds)
            all_gold.extend(batch["labels"].cpu().numpy().tolist())

    if not all_gold:
        return {"accuracy": 0.0, "macro_f1": 0.0, "dev_loss": float("nan")}

    accuracy = float(np.mean(np.asarray(all_gold) == np.asarray(all_pred)))
    f1_scores = []
    for label_id in sorted(SPAN_TYPE_ID2LABEL):
        gold_positive = np.asarray(all_gold) == label_id
        pred_positive = np.asarray(all_pred) == label_id
        tp = int(np.logical_and(gold_positive, pred_positive).sum())
        fp = int(np.logical_and(~gold_positive, pred_positive).sum())
        fn = int(np.logical_and(gold_positive, ~pred_positive).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "dev_loss": total_loss / num_batches if num_batches else float("nan"),
    }


def _classify_spans_for_sentence(
    model: ImpactSpanClassifier,
    tokenizer,
    tokens: list[str],
    spans: list[tuple[int, int]],
    device: str,
    max_length: int = 256,
) -> list[str]:
    if not spans:
        return []

    marked_texts = []
    for start, end in spans:
        marked_tokens = (
            tokens[:start]
            + ["[IMPACT_START]"]
            + tokens[start : end + 1]
            + ["[IMPACT_END]"]
            + tokens[end + 1 :]
        )
        marked_texts.append(" ".join(marked_tokens))

    encoding = tokenizer(
        marked_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    batch_device = {key: value.to(device) for key, value in encoding.items()}
    with torch.no_grad():
        outputs = model(
            input_ids=batch_device["input_ids"],
            attention_mask=batch_device["attention_mask"],
        )
    preds = outputs["logits"].argmax(dim=-1).cpu().tolist()
    return [SPAN_TYPE_ID2LABEL[pred] for pred in preds]


def _evaluate_two_step_pipeline(
    extractor_checkpoint: Path,
    classifier_checkpoint: Path,
    model_name: str,
    batch_size: int,
    device: str,
    data_dir: str,
) -> dict[str, Any]:
    dev_df = load_dataframe(Path(data_dir).resolve() / "new_dev_data.csv")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    extractor_dataset = BinaryImpactNERDataset(dev_df, tokenizer)
    extractor_loader = DataLoader(extractor_dataset, batch_size=batch_size)
    extractor_model = DeBERTaNER(model_name=model_name, num_labels=len(BINARY_LABEL2ID)).to(device)
    extractor_model.load_state_dict(torch.load(extractor_checkpoint, map_location=device))
    extractor_model.eval()

    classifier_model = ImpactSpanClassifier(model_name=model_name).to(device)
    classifier_model.load_state_dict(torch.load(classifier_checkpoint, map_location=device))
    classifier_model.eval()

    all_gold: list[list[str]] = []
    all_pred: list[list[str]] = []
    sample_ptr = 0

    with torch.no_grad():
        for batch in extractor_loader:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = extractor_model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()

            for batch_index in range(preds.shape[0]):
                sample = extractor_dataset.get_full_sample(sample_ptr)
                word_preds = {}
                for token_index, word_id in enumerate(sample["word_ids"]):
                    if word_id is not None and word_id not in word_preds:
                        word_preds[word_id] = BINARY_ID2LABEL[preds[batch_index][token_index]]
                pred_binary_tags = [
                    word_preds.get(word_index, "O")
                    for word_index in range(len(sample["raw_binary_tags"]))
                ]
                pred_binary_tags = _repair_bio_tags(pred_binary_tags)
                pred_spans = [(start, end) for start, end, _ in _typed_spans_from_tags(pred_binary_tags)]
                span_labels = _classify_spans_for_sentence(
                    classifier_model,
                    tokenizer,
                    sample["raw_tokens"],
                    pred_spans,
                    device=device,
                )
                typed_spans = [
                    (start, end, label)
                    for (start, end), label in zip(pred_spans, span_labels)
                ]
                all_gold.append(sample["raw_typed_tags"])
                all_pred.append(_render_typed_spans(len(sample["raw_typed_tags"]), typed_spans))
                sample_ptr += 1

    metrics = evaluate_ner(all_gold, all_pred, print_report=True)
    metrics["predicted_spans"] = int(
        sum(len(_typed_spans_from_tags(tags)) for tags in all_pred)
    )
    return metrics


def _evaluate_sentence_token_hierarchy(
    model: SentenceTokenHierarchyNER,
    dataset: NERDataset,
    dataloader: DataLoader,
    device: str,
    sentence_threshold: float = 0.5,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    all_gold_tags: list[list[str]] = []
    all_pred_tags: list[list[str]] = []
    all_sentence_gold = []
    all_sentence_pred = []
    total_loss = 0.0
    num_batches = 0
    sample_ptr = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            sentence_labels = _sentence_targets_from_labels(batch_device["labels"])
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
                sentence_labels=sentence_labels,
            )
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                num_batches += 1

            token_preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            sentence_probs = torch.sigmoid(outputs["sentence_logits"]).cpu().numpy()
            sentence_pred = (sentence_probs >= sentence_threshold).astype(np.int64)
            sentence_gold = sentence_labels.detach().cpu().numpy().astype(np.int64)

            for batch_index in range(token_preds.shape[0]):
                sample = dataset.get_full_sample(sample_ptr)
                word_preds = {}
                for token_index, word_id in enumerate(sample["word_ids"]):
                    if word_id is not None and word_id not in word_preds:
                        word_preds[word_id] = ID2LABEL[token_preds[batch_index][token_index]]

                pred_tags = [
                    word_preds.get(word_index, "O")
                    for word_index in range(len(sample["raw_tags"]))
                ]
                pred_tags = _repair_bio_tags(pred_tags)
                allow_clinical = bool(sentence_pred[batch_index][0])
                allow_social = bool(sentence_pred[batch_index][1])
                pred_tags = _mask_typed_tags(pred_tags, allow_clinical=allow_clinical, allow_social=allow_social)

                all_gold_tags.append(sample["raw_tags"])
                all_pred_tags.append(pred_tags)
                all_sentence_gold.append(sentence_gold[batch_index])
                all_sentence_pred.append(sentence_pred[batch_index])
                sample_ptr += 1

    token_metrics = evaluate_ner(all_gold_tags, all_pred_tags, print_report=True)
    token_metrics["dev_loss"] = total_loss / num_batches if num_batches else float("nan")
    sentence_metrics = _multilabel_metrics(np.asarray(all_sentence_gold), np.asarray(all_sentence_pred))
    return token_metrics, sentence_metrics


def train_binary_impact_extractor(
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: str = "cuda:0",
    experiment_name: str = "impact_span_extractor",
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    top_k_checkpoints: int = 5,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
) -> tuple[Path, dict[str, float], list[dict[str, Any]]]:
    """Train the binary impact boundary detector for the two-step pipeline."""

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {experiment_name} (Binary Impact Extractor)")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    early_stopping = _init_early_stopping(early_stopping_patience, early_stopping_min_delta)
    print(f"  Early stopping: {_early_stopping_status(early_stopping)}")
    print(f"{'=' * 60}\n")

    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    train_df = load_dataframe(Path(data_dir).resolve() / "new_train_data.csv")
    dev_df = load_dataframe(Path(data_dir).resolve() / "new_dev_data.csv")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = BinaryImpactNERDataset(train_df, tokenizer)
    dev_dataset = BinaryImpactNERDataset(dev_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    model = DeBERTaNER(model_name=model_name, num_labels=len(BINARY_LABEL2ID)).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * max(1, int(epochs)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    checkpoint_manager = TopKCheckpointManager(
        experiment_name=experiment_name,
        output_dir=output_dir_path,
        top_k=top_k_checkpoints,
        metadata={"model_type": "binary_impact_extractor", "model_name": model_name},
    )

    best_span_f1 = 0.0
    best_checkpoint_path = output_dir_path / f"{experiment_name}_best.pt"
    results_log: list[dict[str, Any]] = []

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Extractor Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        dev_metrics = _evaluate_binary_extractor(model, dev_dataset, dev_loader, device)
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches if num_batches else float("nan"),
            **dev_metrics,
        }
        results_log.append(epoch_result)

        checkpoint_manager.maybe_save_state_dict(
            model.state_dict(),
            score=dev_metrics["span_f1"],
            epoch=epoch + 1,
            metrics=epoch_result,
        )
        if dev_metrics["span_f1"] > best_span_f1 or not best_checkpoint_path.exists():
            best_span_f1 = dev_metrics["span_f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> New best extractor span-F1: {best_span_f1:.4f} (saved)")

        if _update_early_stopping(early_stopping, dev_metrics["dev_loss"]):
            print(f"  -> Early stopping triggered at epoch {epoch + 1}")
            break

    with (output_dir_path / f"{experiment_name}_log.json").open("w", encoding="utf-8") as handle:
        json.dump(results_log, handle, indent=2)

    return best_checkpoint_path, {"span_f1": best_span_f1}, results_log


def train_impact_span_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: str = "cuda:0",
    experiment_name: str = "impact_span_classifier",
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    top_k_checkpoints: int = 5,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
) -> tuple[Path, dict[str, float], list[dict[str, Any]]]:
    """Train the span type classifier for the two-step pipeline."""

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {experiment_name} (Impact Span Classifier)")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    early_stopping = _init_early_stopping(early_stopping_patience, early_stopping_min_delta)
    print(f"  Early stopping: {_early_stopping_status(early_stopping)}")
    print(f"{'=' * 60}\n")

    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    train_df = load_dataframe(Path(data_dir).resolve() / "new_train_data.csv")
    dev_df = load_dataframe(Path(data_dir).resolve() / "new_dev_data.csv")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = SpanTypeDataset(train_df, tokenizer)
    dev_dataset = SpanTypeDataset(dev_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    model = ImpactSpanClassifier(model_name=model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * max(1, int(epochs)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    checkpoint_manager = TopKCheckpointManager(
        experiment_name=experiment_name,
        output_dir=output_dir_path,
        top_k=top_k_checkpoints,
        metadata={"model_type": "impact_span_classifier", "model_name": model_name},
    )

    best_macro_f1 = 0.0
    best_checkpoint_path = output_dir_path / f"{experiment_name}_best.pt"
    results_log: list[dict[str, Any]] = []

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Classifier Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        dev_metrics = _evaluate_span_classifier(model, dev_loader, device)
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches if num_batches else float("nan"),
            **dev_metrics,
        }
        results_log.append(epoch_result)

        checkpoint_manager.maybe_save_state_dict(
            model.state_dict(),
            score=dev_metrics["macro_f1"],
            epoch=epoch + 1,
            metrics=epoch_result,
        )
        if dev_metrics["macro_f1"] > best_macro_f1 or not best_checkpoint_path.exists():
            best_macro_f1 = dev_metrics["macro_f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> New best classifier macro-F1: {best_macro_f1:.4f} (saved)")

        if _update_early_stopping(early_stopping, dev_metrics["dev_loss"]):
            print(f"  -> Early stopping triggered at epoch {epoch + 1}")
            break

    with (output_dir_path / f"{experiment_name}_log.json").open("w", encoding="utf-8") as handle:
        json.dump(results_log, handle, indent=2)

    return best_checkpoint_path, {"macro_f1": best_macro_f1}, results_log


def run_two_step_impact_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: str = "cuda:0",
    experiment_name: str = "two_step_impact_pipeline",
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    top_k_checkpoints: int = 5,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
) -> dict[str, Any]:
    """Train a binary extractor and a span classifier, then evaluate the two-step pipeline."""

    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    extractor_checkpoint, extractor_metrics, _ = train_binary_impact_extractor(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        experiment_name=f"{experiment_name}_extractor",
        data_dir=data_dir,
        output_dir=output_dir,
        top_k_checkpoints=top_k_checkpoints,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    classifier_checkpoint, classifier_metrics, _ = train_impact_span_classifier(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        experiment_name=f"{experiment_name}_classifier",
        data_dir=data_dir,
        output_dir=output_dir,
        top_k_checkpoints=top_k_checkpoints,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    pipeline_metrics = _evaluate_two_step_pipeline(
        extractor_checkpoint=extractor_checkpoint,
        classifier_checkpoint=classifier_checkpoint,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        data_dir=data_dir,
    )

    results = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "extractor_checkpoint": str(extractor_checkpoint),
        "classifier_checkpoint": str(classifier_checkpoint),
        "extractor_metrics": extractor_metrics,
        "classifier_metrics": classifier_metrics,
        **pipeline_metrics,
    }
    with (output_dir_path / f"{experiment_name}_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    _clear_torch_memory()
    return results


def train_sentence_token_hierarchy(
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    threshold: float = 0.5,
    device: str = "cuda:0",
    experiment_name: str = "sentence_token_hierarchy",
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    top_k_checkpoints: int = 5,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
    aux_weight: float = 0.3,
) -> tuple[float, list[dict[str, Any]]]:
    """Train the joint sentence-level + token-level hierarchy model."""

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {experiment_name} (Sentence + Token Hierarchy)")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Threshold: {threshold}")
    early_stopping = _init_early_stopping(early_stopping_patience, early_stopping_min_delta)
    print(f"  Early stopping: {_early_stopping_status(early_stopping)}")
    print(f"{'=' * 60}\n")

    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    train_df = load_dataframe(Path(data_dir).resolve() / "new_train_data.csv")
    dev_df = load_dataframe(Path(data_dir).resolve() / "new_dev_data.csv")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = NERDataset(train_df, tokenizer)
    dev_dataset = NERDataset(dev_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    model = SentenceTokenHierarchyNER(
        model_name=model_name,
        num_token_labels=len(LABEL2ID),
        aux_weight=aux_weight,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * max(1, int(epochs)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    checkpoint_manager = TopKCheckpointManager(
        experiment_name=experiment_name,
        output_dir=output_dir_path,
        top_k=top_k_checkpoints,
        metadata={"model_type": "sentence_token_hierarchy", "model_name": model_name},
    )

    best_dev_f1 = float("-inf")
    best_dev_strict_f1 = float("-inf")
    best_checkpoint_path = output_dir_path / f"{experiment_name}_best.pt"
    results_log: list[dict[str, Any]] = []

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Hierarchy Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            sentence_labels = _sentence_targets_from_labels(batch_device["labels"])
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
                sentence_labels=sentence_labels,
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        token_metrics, sentence_metrics = _evaluate_sentence_token_hierarchy(
            model,
            dev_dataset,
            dev_loader,
            device=device,
            sentence_threshold=threshold,
        )
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches if num_batches else float("nan"),
            **token_metrics,
            **sentence_metrics,
        }
        results_log.append(epoch_result)

        checkpoint_manager.maybe_save_state_dict(
            model.state_dict(),
            score=token_metrics["relaxed_f1"],
            epoch=epoch + 1,
            metrics=epoch_result,
        )

        if token_metrics["relaxed_f1"] > best_dev_f1 or not best_checkpoint_path.exists():
            best_dev_f1 = token_metrics["relaxed_f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> New best dev Relaxed F1: {best_dev_f1:.4f} (saved)")

        best_dev_strict_f1 = _maybe_save_strict_best_state_dict(
            model.state_dict(),
            token_metrics,
            experiment_name,
            best_dev_strict_f1,
        )

        if _update_early_stopping(early_stopping, token_metrics["dev_loss"]):
            print(f"  -> Early stopping triggered at epoch {epoch + 1}")
            break

    with (output_dir_path / f"{experiment_name}_log.json").open("w", encoding="utf-8") as handle:
        json.dump(results_log, handle, indent=2)

    print(f"\nBest dev Relaxed F1: {best_dev_f1:.4f}")
    print(f"Best dev Strict F1:  {best_dev_strict_f1:.4f}")

    _clear_torch_memory()
    return best_dev_f1, results_log


def run_span_nested_gliner(
    base_model: str = "urchade/gliner_large-v2.1",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-5,
    threshold: float = 0.4,
    device: str = "cuda:0",
    experiment_name: str = "span_nested_gliner",
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    model_output_dir: str | None = None,
    top_k_checkpoints: int = 5,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
):
    """Alias around GLiNER fine-tuning for span-based / overlap-aware experiments."""

    return finetune_gliner(
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        threshold=threshold,
        device=device,
        experiment_name=experiment_name,
        data_dir=data_dir,
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        top_k_checkpoints=top_k_checkpoints,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )
