"""Search checkpoint ensembles across model combinations."""

from __future__ import annotations

import itertools
import json
import hashlib
import math
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .bilstm_crf import BiLSTMCRF
from .data import (
    BiLSTMDataset,
    ID2LABEL,
    NERDataset,
    NUM_LABELS,
    build_vocab,
    load_dataframe,
    set_runtime_preprocessing,
)
from .deberta_crf import DeBERTaCRF, DeBERTaCRFMultiTask
from .deberta_ner import DeBERTaNER, DeBERTaNERMultiTask
from .ensemble_v2 import apply_bio_repair
from .evaluation import bootstrap_ci, evaluate_ner
from .gliner_finetune import ENTITY_LABELS, _entities_to_bio
from .hierarchical import (
    SentenceImpactDataset,
    _load_ner_model,
    _load_sentence_classifier,
    _mask_predicted_tags,
    _predict_ner_subset,
    _predict_sentence_labels,
)
from .impact_pipelines import (
    BINARY_ID2LABEL,
    BinaryImpactNERDataset,
    ImpactSpanClassifier,
    SentenceTokenHierarchyNER,
    _classify_spans_for_sentence,
    _mask_typed_tags,
    _render_typed_spans,
    _repair_bio_tags,
    _typed_spans_from_tags,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2" / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-large"

SUPPORTED_MODEL_TYPES = {
    "deberta",
    "deberta_multitask",
    "deberta_crf",
    "deberta_crf_multitask",
    "bilstm_crf",
    "gliner",
    "hierarchical_deberta",
    "sentence_token_hierarchy",
    "two_step_impact_pipeline",
}

_PARALLEL_GOLD_TAGS: list[list[str]] | None = None
_PARALLEL_CANDIDATE_PROBS: dict[str, list[np.ndarray]] | None = None
_PARALLEL_CANDIDATE_PREDICTIONS: dict[str, list[list[str]]] | None = None
_PARALLEL_VOTE_METHOD: str = "probability_average"


@dataclass(frozen=True)
class CandidateCheckpoint:
    """A single ensemble candidate resolved from an experiment or checkpoint path."""

    name: str
    path: Path
    model_type: str
    model_name: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateSource:
    """Resolved source experiment location plus selection preferences."""

    experiment_name: str
    output_dir: Path
    selection_metric: str = "relaxed_f1"
    display_label: str | None = None
    enable_preprocessing: bool | None = None


@dataclass(frozen=True)
class CandidateOutputRoot:
    """An output root to scan for top ensemble candidates."""

    path: Path
    display_label: str
    enable_preprocessing: bool | None = None


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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _relative_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _parse_candidate_output_root(spec: str | os.PathLike[str]) -> CandidateOutputRoot:
    raw = str(spec)
    enable_preprocessing: bool | None = None
    path_text = raw

    if "::" in raw:
        path_text, flag = raw.rsplit("::", maxsplit=1)
        flag_normalized = flag.strip().lower()
        flag_map = {
            "enabled": True,
            "enable": True,
            "on": True,
            "true": True,
            "1": True,
            "disabled": False,
            "disable": False,
            "off": False,
            "false": False,
            "0": False,
        }
        if flag_normalized not in flag_map:
            raise ValueError(
                "Invalid candidate-output-dir preprocessing suffix. "
                "Use e.g. /path/to/dir::enabled or /path/to/dir::disabled."
            )
        enable_preprocessing = flag_map[flag_normalized]

    path = Path(path_text).resolve()
    return CandidateOutputRoot(
        path=path,
        display_label=_relative_label(path),
        enable_preprocessing=enable_preprocessing,
    )


def _summarize_log_artifact(path: Path) -> dict[str, Any] | None:
    entries = _read_json(path)
    if not isinstance(entries, list) or not entries:
        return None

    experiment_name = path.stem[:-4] if path.stem.endswith("_log") else path.stem
    relaxed_scores = [_safe_float(entry.get("relaxed_f1"), float("-inf")) for entry in entries]
    strict_scores = [_safe_float(entry.get("strict_f1"), float("-inf")) for entry in entries]
    return {
        "experiment_name": experiment_name,
        "relaxed_f1": max(relaxed_scores) if relaxed_scores else float("-inf"),
        "strict_f1": max(strict_scores) if strict_scores else float("-inf"),
        "output_dir": path.parent.resolve(),
    }


def _summarize_result_artifact(path: Path) -> dict[str, Any] | None:
    if path.name.endswith("_soup_results.json"):
        return None

    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    if "best_overall" in data and "best_by_size" in data:
        return None
    if "relaxed_f1" not in data and "strict_f1" not in data:
        return None

    experiment_name = data.get("experiment_name", path.stem.replace("_results", ""))
    return {
        "experiment_name": experiment_name,
        "relaxed_f1": _safe_float(data.get("relaxed_f1"), float("-inf")),
        "strict_f1": _safe_float(data.get("strict_f1"), float("-inf")),
        "output_dir": path.parent.resolve(),
    }


def _resolve_source_candidates(
    source: CandidateSource,
    checkpoint_limit: int,
) -> list[CandidateCheckpoint]:
    output_dir = source.output_dir.resolve()
    source_experiment = source.experiment_name
    selection_metric = source.selection_metric
    display_label = source.display_label
    enable_preprocessing = source.enable_preprocessing

    name_prefix = f"{display_label}::" if display_label else ""

    summary_path = output_dir / "checkpoints" / source_experiment / "topk_summary.json"
    if summary_path.exists():
        summary = _read_json(summary_path)
        metadata = dict(summary.get("metadata", {}))
        model_type = metadata.get("model_type")
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Source experiment '{source_experiment}' has unsupported model_type '{model_type}'. "
                f"Supported values: {sorted(SUPPORTED_MODEL_TYPES)}"
            )

        if enable_preprocessing is not None:
            metadata["enable_preprocessing"] = bool(enable_preprocessing)

        strict_best_path = output_dir / f"{source_experiment}_strict_best.pt"
        if model_type == "gliner":
            strict_best_dir = output_dir / f"{source_experiment}_strict_best"
            if selection_metric == "strict_f1" and strict_best_dir.exists():
                return [
                    CandidateCheckpoint(
                        name=f"{name_prefix}{source_experiment}@strict",
                        path=strict_best_dir.resolve(),
                        model_type=model_type,
                        model_name=metadata.get("model_name"),
                        metadata=metadata,
                    )
                ]

        if selection_metric == "strict_f1" and strict_best_path.exists():
            return [
                CandidateCheckpoint(
                    name=f"{name_prefix}{source_experiment}@strict",
                    path=strict_best_path.resolve(),
                    model_type=model_type,
                    model_name=metadata.get("model_name"),
                    metadata=metadata,
                )
            ]

        selected = summary.get("checkpoints", [])[:checkpoint_limit]
        if not selected:
            raise ValueError(f"No checkpoints found in top-k summary for '{source_experiment}'.")

        candidates: list[CandidateCheckpoint] = []
        for rank, record in enumerate(selected, start=1):
            checkpoint_path = Path(record["path"]).resolve()
            base_name = f"{name_prefix}{source_experiment}"
            candidate_name = base_name if len(selected) == 1 else f"{base_name}#top{rank}"
            candidates.append(
                CandidateCheckpoint(
                    name=candidate_name,
                    path=checkpoint_path,
                    model_type=model_type,
                    model_name=metadata.get("model_name"),
                    metadata=metadata,
                )
            )
        return candidates

    results_path = output_dir / f"{source_experiment}_results.json"
    if results_path.exists():
        results = _read_json(results_path)
        metadata = dict(results.get("metadata", {}))
        if enable_preprocessing is not None:
            metadata["enable_preprocessing"] = bool(enable_preprocessing)

        model_type = results.get("model_type", metadata.get("model_type"))
        saved_model_path = results.get("saved_model_path")

        if saved_model_path and model_type in SUPPORTED_MODEL_TYPES:
            strict_saved_path = output_dir / f"{source_experiment}_strict_best"
            chosen_path = strict_saved_path if selection_metric == "strict_f1" and strict_saved_path.exists() else Path(saved_model_path)
            return [
                CandidateCheckpoint(
                    name=f"{name_prefix}{results.get('experiment_name', source_experiment)}",
                    path=Path(chosen_path).resolve(),
                    model_type=model_type,
                    model_name=results.get("model_name", metadata.get("model_name")),
                    metadata=metadata,
                )
            ]

        if "classifier_checkpoint" in results and "ner_checkpoint" in results:
            ner_checkpoint = Path(str(results["ner_checkpoint"])).resolve()
            if selection_metric == "strict_f1":
                strict_ner_checkpoint = output_dir / f"{source_experiment}_ner_strict_best.pt"
                if strict_ner_checkpoint.exists():
                    ner_checkpoint = strict_ner_checkpoint.resolve()
            return [
                CandidateCheckpoint(
                    name=f"{name_prefix}{results.get('experiment_name', source_experiment)}",
                    path=results_path.resolve(),
                    model_type="hierarchical_deberta",
                    model_name=results.get("model_name", metadata.get("model_name")),
                    metadata={
                        **metadata,
                        "threshold": results.get("threshold", metadata.get("threshold", 0.5)),
                        "classifier_checkpoint": results["classifier_checkpoint"],
                        "ner_checkpoint": str(ner_checkpoint),
                    },
                )
            ]

        if "extractor_checkpoint" in results and "classifier_checkpoint" in results:
            return [
                CandidateCheckpoint(
                    name=f"{name_prefix}{results.get('experiment_name', source_experiment)}",
                    path=results_path.resolve(),
                    model_type="two_step_impact_pipeline",
                    model_name=results.get("model_name", metadata.get("model_name")),
                    metadata={
                        **metadata,
                        "extractor_checkpoint": results["extractor_checkpoint"],
                        "classifier_checkpoint": results["classifier_checkpoint"],
                    },
                )
            ]

    raise FileNotFoundError(
        f"Could not resolve source experiment '{source_experiment}'. "
        f"Expected either {summary_path} or {results_path}."
    )


def _collect_ranked_source_rows(root: CandidateOutputRoot) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[Path, str], dict[str, Any]] = {}

    for path in sorted(root.path.rglob("*_log.json")):
        summary = _summarize_log_artifact(path)
        if summary is None:
            continue
        key = (summary["output_dir"], summary["experiment_name"])
        existing = rows_by_key.get(key, {})
        rows_by_key[key] = {
            **existing,
            **summary,
            "output_root": root.path,
            "root_label": root.display_label,
            "enable_preprocessing": root.enable_preprocessing,
        }

    for path in sorted(root.path.rglob("*_results.json")):
        summary = _summarize_result_artifact(path)
        if summary is None:
            continue
        key = (summary["output_dir"], summary["experiment_name"])
        existing = rows_by_key.get(key, {})
        rows_by_key[key] = {
            **existing,
            **summary,
            "output_root": root.path,
            "root_label": root.display_label,
            "enable_preprocessing": root.enable_preprocessing,
        }

    return list(rows_by_key.values())


def _select_top_source_specs(
    candidate_output_dirs: list[str] | None,
    top_relaxed_per_output: int,
    top_strict_per_output: int,
) -> tuple[list[CandidateSource], list[dict[str, Any]]]:
    if not candidate_output_dirs:
        return [], []

    selected_sources: list[CandidateSource] = []
    selection_summary: list[dict[str, Any]] = []

    for spec in candidate_output_dirs:
        root = _parse_candidate_output_root(spec)
        if not root.path.exists():
            raise FileNotFoundError(f"Candidate output directory not found: {root.path}")

        rows = _collect_ranked_source_rows(root)
        rankings = [
            ("relaxed_f1", int(top_relaxed_per_output)),
            ("strict_f1", int(top_strict_per_output)),
        ]
        selected_keys: set[tuple[Path, str, str]] = set()

        for metric_name, limit in rankings:
            if limit <= 0:
                continue

            ranked = sorted(
                [row for row in rows if math.isfinite(_safe_float(row.get(metric_name), float("nan")))],
                key=lambda row: _safe_float(row.get(metric_name), float("-inf")),
                reverse=True,
            )

            chosen = 0
            for row in ranked:
                key = (row["output_dir"], row["experiment_name"], metric_name)
                if key in selected_keys:
                    continue

                candidate_source = CandidateSource(
                    experiment_name=row["experiment_name"],
                    output_dir=row["output_dir"],
                    selection_metric=metric_name,
                    display_label=_relative_label(row["output_dir"]),
                    enable_preprocessing=row.get("enable_preprocessing"),
                )
                try:
                    _resolve_source_candidates(candidate_source, checkpoint_limit=1)
                except (FileNotFoundError, ValueError):
                    continue

                selected_keys.add(key)
                selected_sources.append(candidate_source)
                selection_summary.append(
                    {
                        "selection_metric": metric_name,
                        "output_root": root.display_label,
                        "source_output_dir": _relative_label(row["output_dir"]),
                        "experiment_name": row["experiment_name"],
                        "relaxed_f1": row.get("relaxed_f1"),
                        "strict_f1": row.get("strict_f1"),
                        "enable_preprocessing": row.get("enable_preprocessing"),
                    }
                )
                chosen += 1
                if chosen >= limit:
                    break

    return selected_sources, selection_summary
def _resolve_direct_candidates(
    checkpoint_names: list[str],
    model_type: str | None,
    model_name: str,
) -> list[CandidateCheckpoint]:
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "When using --checkpoint directly, you must provide --model-type with one of: "
            f"{sorted(SUPPORTED_MODEL_TYPES)}"
        )

    candidates: list[CandidateCheckpoint] = []
    for checkpoint_name in checkpoint_names:
        checkpoint_path = Path(checkpoint_name).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        candidates.append(
            CandidateCheckpoint(
                name=checkpoint_path.stem,
                path=checkpoint_path,
                model_type=model_type,
                model_name=None if model_type == "bilstm_crf" else model_name,
                metadata={},
            )
        )
    return candidates


def _collect_transformer_probs(
    candidate: CandidateCheckpoint,
    dataset: NERDataset,
    batch_size: int,
    device: str,
) -> tuple[list[np.ndarray], list[list[str]]]:
    model_name = candidate.model_name or DEFAULT_MODEL_NAME
    model_type = candidate.model_type

    if model_type == "deberta":
        model = DeBERTaNER(model_name=model_name, num_labels=NUM_LABELS)
    elif model_type == "deberta_multitask":
        model = DeBERTaNERMultiTask(model_name=model_name, num_labels=NUM_LABELS)
    elif model_type == "deberta_crf":
        model = DeBERTaCRF(
            model_name=model_name,
            num_labels=NUM_LABELS,
            use_lstm=bool(candidate.metadata.get("use_lstm", False)),
        )
    elif model_type == "deberta_crf_multitask":
        model = DeBERTaCRFMultiTask(model_name=model_name, num_labels=NUM_LABELS)
    else:
        raise AssertionError(f"Unexpected transformer model_type: {model_type}")

    model.load_state_dict(torch.load(candidate.path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size)
    all_probs: list[np.ndarray] = []
    all_gold: list[list[str]] = []
    sample_index = 0

    with torch.no_grad():
        for batch in loader:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            token_scores = outputs["logits"].detach().cpu().float().numpy()

            for batch_index in range(token_scores.shape[0]):
                sample = dataset.get_full_sample(sample_index)
                gold_tags = sample["raw_tags"]
                word_scores = _first_subword_scores(
                    token_scores[batch_index],
                    sample["word_ids"],
                    len(gold_tags),
                )
                all_probs.append(_softmax(word_scores))
                all_gold.append(gold_tags)
                sample_index += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_probs, all_gold


def _collect_bilstm_probs(
    candidate: CandidateCheckpoint,
    dataset: BiLSTMDataset,
    batch_size: int,
    device: str,
    vocab_size: int,
) -> tuple[list[np.ndarray], list[list[str]]]:
    model = BiLSTMCRF(vocab_size=vocab_size, num_tags=NUM_LABELS)
    model.load_state_dict(torch.load(candidate.path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size)
    all_probs: list[np.ndarray] = []
    all_gold: list[list[str]] = []
    sample_index = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids=input_ids)
            emissions = outputs["emissions"].detach().cpu().float().numpy()
            lengths = batch["length"].cpu().numpy()

            for batch_index in range(emissions.shape[0]):
                sample = dataset.samples[sample_index]
                length = min(int(lengths[batch_index]), len(sample["raw_tags"]))
                all_probs.append(_softmax(emissions[batch_index][:length]))
                all_gold.append(sample["raw_tags"][:length])
                sample_index += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_probs, all_gold


def _decode_single_candidate_probs(probabilities: list[np.ndarray]) -> list[list[str]]:
    predictions: list[list[str]] = []
    for sample_probs in probabilities:
        pred_ids = sample_probs.argmax(axis=-1)
        pred_tags = [ID2LABEL[int(pred_id)] for pred_id in pred_ids]
        predictions.append(apply_bio_repair(pred_tags))
    return predictions


def _collect_hierarchical_predictions(
    candidate: CandidateCheckpoint,
    batch_size: int,
    device: str,
    data_dir: Path,
) -> tuple[list[list[str]], list[list[str]]]:
    model_name = candidate.model_name or DEFAULT_MODEL_NAME
    threshold = float(candidate.metadata.get("threshold", 0.5))
    classifier_checkpoint = Path(str(candidate.metadata["classifier_checkpoint"])).resolve()
    ner_checkpoint = Path(str(candidate.metadata["ner_checkpoint"])).resolve()
    enable_preprocessing = bool(candidate.metadata.get("enable_preprocessing", False))

    dev_df = load_dataframe(data_dir / "new_dev_data.csv", apply_preprocessing=enable_preprocessing)

    sentence_tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence_dataset = SentenceImpactDataset(dev_df, sentence_tokenizer)
    classifier_model = _load_sentence_classifier(classifier_checkpoint, model_name, device)
    _, sentence_preds = _predict_sentence_labels(
        classifier_model,
        sentence_dataset,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
    )

    ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_dataset = NERDataset(dev_df, ner_tokenizer, apply_preprocessing=False)
    gold_tags = [sample["raw_tags"] for sample in ner_dataset.samples]
    all_predictions = [["O"] * len(gold) for gold in gold_tags]

    positive_indices = [index for index, pred in enumerate(sentence_preds) if int(pred.sum()) > 0]
    if positive_indices:
        ner_model = _load_ner_model(ner_checkpoint, model_name, device, use_multitask=True)
        ner_predictions = _predict_ner_subset(ner_model, ner_dataset, positive_indices, batch_size, device)
        for sample_index in positive_indices:
            sample_pred = ner_predictions.get(sample_index, all_predictions[sample_index])
            allow_clinical = bool(sentence_preds[sample_index][0])
            allow_social = bool(sentence_preds[sample_index][1])
            all_predictions[sample_index] = _mask_predicted_tags(sample_pred, allow_clinical, allow_social)
        del ner_model
    del classifier_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_predictions, gold_tags


def _collect_sentence_token_hierarchy_predictions(
    candidate: CandidateCheckpoint,
    batch_size: int,
    device: str,
    data_dir: Path,
) -> tuple[list[list[str]], list[list[str]]]:
    model_name = candidate.model_name or DEFAULT_MODEL_NAME
    threshold = float(candidate.metadata.get("threshold", 0.5))
    enable_preprocessing = bool(candidate.metadata.get("enable_preprocessing", False))

    dev_df = load_dataframe(data_dir / "new_dev_data.csv", apply_preprocessing=enable_preprocessing)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = NERDataset(dev_df, tokenizer, apply_preprocessing=False)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = SentenceTokenHierarchyNER(
        model_name=model_name,
        num_token_labels=NUM_LABELS,
    ).to(device)
    model.load_state_dict(torch.load(candidate.path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    all_gold: list[list[str]] = []
    all_predictions: list[list[str]] = []
    sample_ptr = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            token_preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            sentence_probs = torch.sigmoid(outputs["sentence_logits"]).cpu().numpy()
            sentence_pred = (sentence_probs >= threshold).astype(np.int64)

            for batch_index in range(token_preds.shape[0]):
                sample = dataset.get_full_sample(sample_ptr)
                word_preds = {}
                for token_index, word_id in enumerate(sample["word_ids"]):
                    if word_id is not None and word_id not in word_preds:
                        word_preds[word_id] = ID2LABEL[int(token_preds[batch_index][token_index])]

                pred_tags = [
                    word_preds.get(word_index, "O")
                    for word_index in range(len(sample["raw_tags"]))
                ]
                pred_tags = _repair_bio_tags(pred_tags)
                pred_tags = _mask_typed_tags(
                    pred_tags,
                    allow_clinical=bool(sentence_pred[batch_index][0]),
                    allow_social=bool(sentence_pred[batch_index][1]),
                )

                all_gold.append(sample["raw_tags"])
                all_predictions.append(pred_tags)
                sample_ptr += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_predictions, all_gold


def _collect_gliner_predictions(
    candidate: CandidateCheckpoint,
    data_dir: Path,
) -> tuple[list[list[str]], list[list[str]]]:
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise ImportError(
            "GLiNER ensemble candidates require the 'gliner' package to be installed."
        ) from exc

    enable_preprocessing = bool(candidate.metadata.get("enable_preprocessing", False))
    threshold = float(candidate.metadata.get("threshold", 0.4))
    dev_df = load_dataframe(data_dir / "new_dev_data.csv", apply_preprocessing=enable_preprocessing)
    model = GLiNER.from_pretrained(str(candidate.path))

    all_gold: list[list[str]] = []
    all_predictions: list[list[str]] = []
    for _, row in dev_df.iterrows():
        tokens = list(row["tokens"])
        gold_tags = list(row["ner_tags"])
        text = " ".join(tokens)
        try:
            entities = model.predict_entities(text, ENTITY_LABELS, threshold=threshold)
        except Exception:
            entities = []
        all_gold.append(gold_tags)
        all_predictions.append(_entities_to_bio(tokens, entities))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_predictions, all_gold


def _collect_two_step_predictions(
    candidate: CandidateCheckpoint,
    batch_size: int,
    device: str,
    data_dir: Path,
) -> tuple[list[list[str]], list[list[str]]]:
    model_name = candidate.model_name or DEFAULT_MODEL_NAME
    enable_preprocessing = bool(candidate.metadata.get("enable_preprocessing", False))
    extractor_checkpoint = Path(str(candidate.metadata["extractor_checkpoint"])).resolve()
    classifier_checkpoint = Path(str(candidate.metadata["classifier_checkpoint"])).resolve()

    dev_df = load_dataframe(data_dir / "new_dev_data.csv", apply_preprocessing=enable_preprocessing)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    extractor_dataset = BinaryImpactNERDataset(dev_df, tokenizer)
    extractor_loader = DataLoader(extractor_dataset, batch_size=batch_size)

    extractor_model = DeBERTaNER(model_name=model_name, num_labels=len(BINARY_ID2LABEL)).to(device)
    extractor_model.load_state_dict(torch.load(extractor_checkpoint, map_location="cpu"))
    extractor_model = extractor_model.to(device)
    extractor_model.eval()

    classifier_model = ImpactSpanClassifier(model_name=model_name).to(device)
    classifier_model.load_state_dict(torch.load(classifier_checkpoint, map_location="cpu"))
    classifier_model = classifier_model.to(device)
    classifier_model.eval()

    all_gold: list[list[str]] = []
    all_predictions: list[list[str]] = []
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
                        word_preds[word_id] = BINARY_ID2LABEL[int(preds[batch_index][token_index])]
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
                all_predictions.append(_render_typed_spans(len(sample["raw_typed_tags"]), typed_spans))
                sample_ptr += 1

    del extractor_model, classifier_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_predictions, all_gold


def _decode_average_probs(probabilities: list[list[np.ndarray]]) -> list[list[str]]:
    combined_predictions: list[list[str]] = []
    num_samples = len(probabilities[0])

    for sample_index in range(num_samples):
        avg_probs = np.mean([candidate_probs[sample_index] for candidate_probs in probabilities], axis=0)
        pred_ids = avg_probs.argmax(axis=-1)
        pred_tags = [ID2LABEL[int(pred_id)] for pred_id in pred_ids]
        combined_predictions.append(apply_bio_repair(pred_tags))

    return combined_predictions


def _attach_bootstrap_from_predictions(
    record: dict[str, Any],
    predictions: list[list[str]],
    gold_tags: list[list[str]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        return record

    ci = bootstrap_ci(gold_tags, predictions, n_bootstrap=bootstrap_samples)
    enriched = dict(record)
    enriched["ci_lower"] = ci["ci_lower"]
    enriched["ci_upper"] = ci["ci_upper"]
    return enriched


def _attach_bootstrap_from_probabilities(
    record: dict[str, Any],
    probabilities: list[list[np.ndarray]],
    gold_tags: list[list[str]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        return record

    predictions = _decode_average_probs(probabilities)
    return _attach_bootstrap_from_predictions(record, predictions, gold_tags, bootstrap_samples)


def _slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "model"


def _combination_filename(index: int, size: int, models: list[str]) -> str:
    joined = "__".join(models)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    lead = "__".join(_slugify_name(model) for model in models[:2])
    lead = lead[:80].rstrip("-_")
    if lead:
        return f"{index:04d}_size{size}_{lead}_{digest}.json"
    return f"{index:04d}_size{size}_{digest}.json"


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


def _set_parallel_combination_state(
    gold_tags: list[list[str]],
    candidate_probs: dict[str, list[np.ndarray]],
    candidate_predictions: dict[str, list[list[str]]],
    vote_method: str,
) -> None:
    global _PARALLEL_GOLD_TAGS
    global _PARALLEL_CANDIDATE_PROBS
    global _PARALLEL_CANDIDATE_PREDICTIONS
    global _PARALLEL_VOTE_METHOD

    _PARALLEL_GOLD_TAGS = gold_tags
    _PARALLEL_CANDIDATE_PROBS = candidate_probs
    _PARALLEL_CANDIDATE_PREDICTIONS = candidate_predictions
    _PARALLEL_VOTE_METHOD = vote_method


def _score_combination_task(task: tuple[int, int, tuple[str, ...]]) -> dict[str, Any]:
    combination_index, size, combination_names = task

    if _PARALLEL_GOLD_TAGS is None:
        raise RuntimeError("Parallel ensemble scoring state was not initialized.")

    if _PARALLEL_VOTE_METHOD == "probability_average":
        assert _PARALLEL_CANDIDATE_PROBS is not None
        probabilities = [_PARALLEL_CANDIDATE_PROBS[name] for name in combination_names]
        predictions = _decode_average_probs(probabilities)
    else:
        assert _PARALLEL_CANDIDATE_PREDICTIONS is not None
        predictions = _majority_vote_predictions(
            [_PARALLEL_CANDIDATE_PREDICTIONS[name] for name in combination_names]
        )

    metrics = evaluate_ner(_PARALLEL_GOLD_TAGS, predictions, print_report=False)
    return {
        "combination_index": combination_index,
        "num_models": size,
        "models": list(combination_names),
        "vote_method": _PARALLEL_VOTE_METHOD,
        **metrics,
    }


def _iter_combination_tasks(
    candidates: list[CandidateCheckpoint],
    size: int,
    start_index: int,
) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    return tuple(
        (
            start_index + offset,
            size,
            tuple(candidate.name for candidate in combination),
        )
        for offset, combination in enumerate(itertools.combinations(candidates, size))
    )


def _iter_scored_combination_records(
    tasks: tuple[tuple[int, int, tuple[str, ...]], ...],
    parallel_workers: int,
) -> Any:
    if not tasks:
        return

    if parallel_workers <= 1:
        for task in tasks:
            yield _score_combination_task(task)
        return

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("Parallel ensemble scoring requires fork multiprocessing on this platform. Falling back to serial scoring.")
        for task in tasks:
            yield _score_combination_task(task)
        return

    chunksize = max(1, len(tasks) // max(1, parallel_workers * 4))
    with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=ctx) as executor:
        for record in executor.map(_score_combination_task, tasks, chunksize=chunksize):
            yield record


def _materialize_record_predictions(
    record: dict[str, Any],
    vote_method: str,
    candidate_probs: dict[str, list[np.ndarray]],
    candidate_predictions: dict[str, list[list[str]]],
) -> tuple[list[list[str]], list[list[np.ndarray]] | None]:
    if vote_method == "probability_average":
        probabilities = [candidate_probs[name] for name in record["models"]]
        return _decode_average_probs(probabilities), probabilities

    predictions = _majority_vote_predictions([candidate_predictions[name] for name in record["models"]])
    return predictions, None


def _enrich_record_with_bootstrap(
    record: dict[str, Any],
    vote_method: str,
    candidate_probs: dict[str, list[np.ndarray]],
    candidate_predictions: dict[str, list[list[str]]],
    gold_tags: list[list[str]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    predictions, probabilities = _materialize_record_predictions(
        record,
        vote_method,
        candidate_probs,
        candidate_predictions,
    )
    return (
        _attach_bootstrap_from_probabilities(
            record,
            probabilities,
            gold_tags,
            bootstrap_samples,
        )
        if vote_method == "probability_average" and probabilities is not None
        else _attach_bootstrap_from_predictions(
            record,
            predictions,
            gold_tags,
            bootstrap_samples,
        )
    )


def run_ensemble_search(
    source_experiments: list[str] | None = None,
    checkpoint_names: list[str] | None = None,
    candidate_output_dirs: list[str] | None = None,
    top_relaxed_per_output: int = 0,
    top_strict_per_output: int = 0,
    checkpoint_limit: int = 1,
    model_type: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 8,
    device: str = "cuda:0",
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    experiment_name: str = "ensemble_search",
    min_models: int = 2,
    max_models: int = 5,
    ensemble_sizes: list[int] | None = None,
    bootstrap_samples: int = 2000,
    vote_method: str = "probability_average",
    save_combination_files: bool = False,
    parallel_workers: int = 1,
) -> dict[str, Any]:
    """Evaluate every ensemble combination in the requested size range."""

    output_dir_path = Path(output_dir).resolve()
    data_dir_path = Path(data_dir).resolve()
    set_runtime_preprocessing(False)

    if not source_experiments and not checkpoint_names and not candidate_output_dirs:
        raise ValueError(
            "Provide at least one --source-experiment, --checkpoint, or --candidate-output-dir for ensemble_search."
        )
    if checkpoint_limit < 1:
        raise ValueError("--checkpoint-limit must be at least 1.")
    if int(parallel_workers) < 1:
        raise ValueError("--parallel-workers must be at least 1.")
    if vote_method not in {"probability_average", "majority_vote"}:
        raise ValueError("vote_method must be either 'probability_average' or 'majority_vote'.")

    candidates: list[CandidateCheckpoint] = []
    auto_sources, auto_selection_summary = _select_top_source_specs(
        candidate_output_dirs,
        top_relaxed_per_output=top_relaxed_per_output,
        top_strict_per_output=top_strict_per_output,
    )
    if source_experiments:
        for source_experiment in source_experiments:
            candidates.extend(
                _resolve_source_candidates(
                    CandidateSource(
                        experiment_name=source_experiment,
                        output_dir=output_dir_path,
                    ),
                    checkpoint_limit,
                )
            )
    for source in auto_sources:
        candidates.extend(_resolve_source_candidates(source, checkpoint_limit=1))
    if checkpoint_names:
        candidates.extend(_resolve_direct_candidates(checkpoint_names, model_type, model_name))

    unique_candidates: list[CandidateCheckpoint] = []
    seen_names: set[str] = set()
    for candidate in candidates:
        if candidate.name in seen_names:
            deduped_name = f"{candidate.name}_{len(seen_names) + 1}"
            candidate = CandidateCheckpoint(
                name=deduped_name,
                path=candidate.path,
                model_type=candidate.model_type,
                model_name=candidate.model_name,
                metadata=candidate.metadata,
            )
        seen_names.add(candidate.name)
        unique_candidates.append(candidate)
    candidates = unique_candidates

    if len(candidates) < 2:
        raise ValueError("Need at least two ensemble candidates after resolution.")
    if vote_method == "probability_average":
        unsupported_for_average = [
            candidate.name
            for candidate in candidates
            if candidate.model_type in {"gliner", "hierarchical_deberta", "two_step_impact_pipeline", "sentence_token_hierarchy"}
        ]
        if unsupported_for_average:
            formatted = ", ".join(unsupported_for_average)
            raise ValueError(
                "These pipeline candidates require --vote-method majority_vote because they do not expose "
                f"a directly comparable token-level probability tensor for averaging: {formatted}"
            )

    if ensemble_sizes:
        sizes = sorted({int(size) for size in ensemble_sizes})
        sizes = [size for size in sizes if size >= 2]
        sizes = [size for size in sizes if size <= len(candidates)]
        if not sizes:
            raise ValueError(
                f"No valid --ensemble-size values remain after filtering for candidate_count={len(candidates)}."
            )
    else:
        min_models = max(2, int(min_models))
        max_models = min(int(max_models), len(candidates))
        if min_models > max_models:
            raise ValueError(
                f"Invalid combination range: min_models={min_models}, max_models={max_models}, "
                f"candidates={len(candidates)}."
            )
        sizes = list(range(min_models, max_models + 1))

    print("\n" + "=" * 60)
    print(f"  Experiment: {experiment_name} (Ensemble Search)")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Combination sizes: {', '.join(str(size) for size in sizes)}")
    print(f"  Vote method: {vote_method}")
    print(f"  Parallel workers: {parallel_workers}")
    if auto_selection_summary:
        print(f"  Auto-selected candidates: {len(auto_selection_summary)}")
    print("=" * 60 + "\n")
    for candidate in candidates:
        print(f"  - {candidate.name}: {candidate.model_type} [{candidate.path.name}]")

    transformer_dataset_cache: dict[tuple[str, bool, bool], NERDataset] = {}
    bilstm_bundle_cache: dict[bool, tuple[int, BiLSTMDataset]] = {}
    gold_tags: list[list[str]] | None = None
    candidate_probs: dict[str, list[np.ndarray]] = {}
    candidate_predictions: dict[str, list[list[str]]] = {}

    for candidate in candidates:
        if candidate.model_type == "hierarchical_deberta":
            predictions, candidate_gold = _collect_hierarchical_predictions(
                candidate,
                batch_size=batch_size,
                device=device,
                data_dir=data_dir_path,
            )
            candidate_predictions[candidate.name] = predictions
        elif candidate.model_type == "gliner":
            predictions, candidate_gold = _collect_gliner_predictions(
                candidate,
                data_dir=data_dir_path,
            )
            candidate_predictions[candidate.name] = predictions
        elif candidate.model_type == "two_step_impact_pipeline":
            predictions, candidate_gold = _collect_two_step_predictions(
                candidate,
                batch_size=batch_size,
                device=device,
                data_dir=data_dir_path,
            )
            candidate_predictions[candidate.name] = predictions
        elif candidate.model_type == "sentence_token_hierarchy":
            predictions, candidate_gold = _collect_sentence_token_hierarchy_predictions(
                candidate,
                batch_size=batch_size,
                device=device,
                data_dir=data_dir_path,
            )
            candidate_predictions[candidate.name] = predictions
        elif candidate.model_type == "bilstm_crf":
            enable_preprocessing = bool(candidate.metadata.get("enable_preprocessing", False))
            if enable_preprocessing not in bilstm_bundle_cache:
                train_df = load_dataframe(
                    data_dir_path / "new_train_data.csv",
                    apply_preprocessing=enable_preprocessing,
                )
                dev_df = load_dataframe(
                    data_dir_path / "new_dev_data.csv",
                    apply_preprocessing=enable_preprocessing,
                )
                word2idx = build_vocab(train_df)
                bilstm_bundle_cache[enable_preprocessing] = (
                    len(word2idx),
                    BiLSTMDataset(dev_df, word2idx, apply_preprocessing=False),
                )
            vocab_size, bilstm_dataset = bilstm_bundle_cache[enable_preprocessing]
            probs, candidate_gold = _collect_bilstm_probs(
                candidate,
                bilstm_dataset,
                batch_size,
                device,
                vocab_size,
            )
            if vote_method == "probability_average":
                candidate_probs[candidate.name] = probs
            else:
                candidate_predictions[candidate.name] = _decode_single_candidate_probs(probs)
        else:
            definition_prompting = bool(candidate.metadata.get("definition_prompting", False))
            enable_preprocessing = bool(candidate.metadata.get("enable_preprocessing", False))
            dataset_key = (candidate.model_name or model_name, definition_prompting, enable_preprocessing)
            if dataset_key not in transformer_dataset_cache:
                dev_df = load_dataframe(
                    data_dir_path / "new_dev_data.csv",
                    apply_preprocessing=enable_preprocessing,
                )
                tokenizer = AutoTokenizer.from_pretrained(candidate.model_name or model_name)
                transformer_dataset_cache[dataset_key] = NERDataset(
                    dev_df,
                    tokenizer,
                    definition_prompting=definition_prompting,
                    apply_preprocessing=False,
                )
            probs, candidate_gold = _collect_transformer_probs(
                candidate,
                transformer_dataset_cache[dataset_key],
                batch_size,
                device,
            )
            if vote_method == "probability_average":
                candidate_probs[candidate.name] = probs
            else:
                candidate_predictions[candidate.name] = _decode_single_candidate_probs(probs)

        if gold_tags is None:
            gold_tags = candidate_gold
        elif len(gold_tags) != len(candidate_gold):
            raise ValueError(
                f"Candidate '{candidate.name}' produced {len(candidate_gold)} samples, "
                f"but earlier candidates produced {len(gold_tags)}."
            )

    assert gold_tags is not None
    _set_parallel_combination_state(gold_tags, candidate_probs, candidate_predictions, vote_method)

    total_combinations = sum(math.comb(len(candidates), size) for size in sizes)
    print(f"\nEvaluating {total_combinations} ensemble combinations...\n")

    all_results: list[dict[str, Any]] = []
    best_by_size_relaxed: dict[int, dict[str, Any]] = {}
    best_by_size_strict: dict[int, dict[str, Any]] = {}
    combination_index = 0
    next_progress_pct = 10
    combination_output_dir: Path | None = None
    if save_combination_files:
        combination_output_dir = output_dir_path / f"{experiment_name}_combinations"
        combination_output_dir.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        combinations_for_size = math.comb(len(candidates), size)
        print(f"[size={size}] {combinations_for_size} combinations")
        tasks = _iter_combination_tasks(candidates, size, combination_index + 1)
        size_best_relaxed: dict[str, Any] | None = None
        size_best_strict: dict[str, Any] | None = None

        for record in _iter_scored_combination_records(tasks, parallel_workers=parallel_workers):
            combination_index = record["combination_index"]
            all_results.append(record)

            if combination_output_dir is not None:
                combination_file = combination_output_dir / _combination_filename(
                    record["combination_index"],
                    size,
                    record["models"],
                )
                with combination_file.open("w", encoding="utf-8") as handle:
                    json.dump(record, handle, indent=2)

            if size_best_relaxed is None or record["relaxed_f1"] > size_best_relaxed["relaxed_f1"]:
                size_best_relaxed = record
            if size_best_strict is None or record["strict_f1"] > size_best_strict["strict_f1"]:
                size_best_strict = record

            while total_combinations > 0 and next_progress_pct <= 100 and combination_index * 100 >= total_combinations * next_progress_pct:
                print(
                    f"  Progress: {next_progress_pct}% "
                    f"({combination_index}/{total_combinations} combinations)"
                )
                next_progress_pct += 10

        assert size_best_relaxed is not None
        assert size_best_strict is not None
        best_by_size_relaxed[size] = size_best_relaxed
        best_by_size_strict[size] = size_best_strict
        print(
            f"  best size-{size} by relaxed: F1={size_best_relaxed['relaxed_f1']:.4f} | "
            f"{', '.join(size_best_relaxed['models'])}"
        )
        print(
            f"  best size-{size} by strict: F1={size_best_strict['strict_f1']:.4f} | "
            f"{', '.join(size_best_strict['models'])}"
        )

    ranked_results_relaxed = sorted(all_results, key=lambda item: item["relaxed_f1"], reverse=True)
    ranked_results_strict = sorted(all_results, key=lambda item: item["strict_f1"], reverse=True)
    best_overall_relaxed = ranked_results_relaxed[0]
    best_overall_strict = ranked_results_strict[0]

    best_by_size_relaxed_json: dict[str, Any] = {}
    for size, record in sorted(best_by_size_relaxed.items()):
        best_by_size_relaxed_json[str(size)] = _enrich_record_with_bootstrap(
            record,
            vote_method,
            candidate_probs,
            candidate_predictions,
            gold_tags,
            bootstrap_samples,
        )

    best_by_size_strict_json: dict[str, Any] = {}
    for size, record in sorted(best_by_size_strict.items()):
        best_by_size_strict_json[str(size)] = _enrich_record_with_bootstrap(
            record,
            vote_method,
            candidate_probs,
            candidate_predictions,
            gold_tags,
            bootstrap_samples,
        )

    best_overall_relaxed_json = _enrich_record_with_bootstrap(
        best_overall_relaxed,
        vote_method,
        candidate_probs,
        candidate_predictions,
        gold_tags,
        bootstrap_samples,
    )
    best_overall_strict_json = _enrich_record_with_bootstrap(
        best_overall_strict,
        vote_method,
        candidate_probs,
        candidate_predictions,
        gold_tags,
        bootstrap_samples,
    )

    results = {
        "experiment_name": experiment_name,
        "vote_method": vote_method,
        "candidate_count": len(candidates),
        "candidates": [
            {
                "name": candidate.name,
                "path": str(candidate.path),
                "model_type": candidate.model_type,
                "model_name": candidate.model_name,
                "metadata": candidate.metadata,
            }
            for candidate in candidates
        ],
        "search_space": {
            "min_models": min(sizes),
            "max_models": max(sizes),
            "ensemble_sizes": sizes,
            "checkpoint_limit": checkpoint_limit,
            "total_combinations": total_combinations,
        },
        "best_overall": best_overall_relaxed_json,
        "best_overall_by_relaxed": best_overall_relaxed_json,
        "best_overall_by_strict": best_overall_strict_json,
        "best_by_size": best_by_size_relaxed_json,
        "best_by_size_by_relaxed": best_by_size_relaxed_json,
        "best_by_size_by_strict": best_by_size_strict_json,
        "top_results": ranked_results_relaxed[:20],
        "top_results_by_relaxed": ranked_results_relaxed[:20],
        "top_results_by_strict": ranked_results_strict[:20],
        "all_results": ranked_results_relaxed,
        "all_results_by_relaxed": ranked_results_relaxed,
        "all_results_by_strict": ranked_results_strict,
    }
    if auto_selection_summary:
        results["auto_selected_sources"] = auto_selection_summary
    if combination_output_dir is not None:
        results["combination_results_dir"] = str(combination_output_dir)

    output_path = output_dir_path / f"{experiment_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nBest overall ensemble:")
    print(f"  Best by relaxed F1: {results['best_overall_by_relaxed']['relaxed_f1']:.4f}")
    print(f"  Models: {', '.join(results['best_overall_by_relaxed']['models'])}")
    print(f"  Best by strict F1:  {results['best_overall_by_strict']['strict_f1']:.4f}")
    print(f"  Models: {', '.join(results['best_overall_by_strict']['models'])}")
    print(f"\nSaved ensemble search results to {output_path}")
    return results
