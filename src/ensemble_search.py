"""Search checkpoint ensembles across model combinations."""

from __future__ import annotations

import itertools
import json
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .bilstm_crf import BiLSTMCRF
from .data import BiLSTMDataset, ID2LABEL, NERDataset, NUM_LABELS, build_vocab, load_dataframe
from .deberta_crf import DeBERTaCRF, DeBERTaCRFMultiTask
from .deberta_ner import DeBERTaNER, DeBERTaNERMultiTask
from .ensemble_v2 import apply_bio_repair
from .evaluation import bootstrap_ci, evaluate_ner
from .hierarchical import (
    SentenceImpactDataset,
    _load_ner_model,
    _load_sentence_classifier,
    _mask_predicted_tags,
    _predict_ner_subset,
    _predict_sentence_labels,
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
    "hierarchical_deberta",
}


@dataclass(frozen=True)
class CandidateCheckpoint:
    """A single ensemble candidate resolved from an experiment or checkpoint path."""

    name: str
    path: Path
    model_type: str
    model_name: str | None
    metadata: dict[str, Any]


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


def _resolve_source_candidates(
    source_experiment: str,
    output_dir: Path,
    checkpoint_limit: int,
) -> list[CandidateCheckpoint]:
    summary_path = output_dir / "checkpoints" / source_experiment / "topk_summary.json"
    if summary_path.exists():
        summary = _read_json(summary_path)
        metadata = summary.get("metadata", {})
        model_type = metadata.get("model_type")
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Source experiment '{source_experiment}' has unsupported model_type '{model_type}'. "
                f"Supported values: {sorted(SUPPORTED_MODEL_TYPES)}"
            )

        selected = summary.get("checkpoints", [])[:checkpoint_limit]
        if not selected:
            raise ValueError(f"No checkpoints found in top-k summary for '{source_experiment}'.")

        candidates: list[CandidateCheckpoint] = []
        for rank, record in enumerate(selected, start=1):
            checkpoint_path = Path(record["path"]).resolve()
            candidate_name = source_experiment if len(selected) == 1 else f"{source_experiment}#top{rank}"
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
        metadata = results.get("metadata", {})
        model_type = results.get("model_type", metadata.get("model_type"))
        saved_model_path = results.get("saved_model_path")

        if saved_model_path and model_type in SUPPORTED_MODEL_TYPES:
            return [
                CandidateCheckpoint(
                    name=results.get("experiment_name", source_experiment),
                    path=Path(saved_model_path).resolve(),
                    model_type=model_type,
                    model_name=results.get("model_name", metadata.get("model_name")),
                    metadata=metadata,
                )
            ]

        if "classifier_checkpoint" in results and "ner_checkpoint" in results:
            return [
                CandidateCheckpoint(
                    name=results.get("experiment_name", source_experiment),
                    path=results_path.resolve(),
                    model_type="hierarchical_deberta",
                    model_name=results.get("model_name", metadata.get("model_name")),
                    metadata={
                        **metadata,
                        "threshold": results.get("threshold", metadata.get("threshold", 0.5)),
                        "classifier_checkpoint": results["classifier_checkpoint"],
                        "ner_checkpoint": results["ner_checkpoint"],
                    },
                )
            ]

    raise FileNotFoundError(
        f"Could not resolve source experiment '{source_experiment}'. "
        f"Expected either {summary_path} or {results_path}."
    )


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

    dev_df = load_dataframe(data_dir / "new_dev_data.csv")

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
    ner_dataset = NERDataset(dev_df, ner_tokenizer)
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


def run_ensemble_search(
    source_experiments: list[str] | None = None,
    checkpoint_names: list[str] | None = None,
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
    bootstrap_samples: int = 2000,
    vote_method: str = "probability_average",
    save_combination_files: bool = False,
) -> dict[str, Any]:
    """Evaluate every ensemble combination in the requested size range."""

    output_dir_path = Path(output_dir).resolve()
    data_dir_path = Path(data_dir).resolve()

    if not source_experiments and not checkpoint_names:
        raise ValueError("Provide at least one --source-experiment or --checkpoint for ensemble_search.")
    if checkpoint_limit < 1:
        raise ValueError("--checkpoint-limit must be at least 1.")
    if vote_method not in {"probability_average", "majority_vote"}:
        raise ValueError("vote_method must be either 'probability_average' or 'majority_vote'.")

    candidates: list[CandidateCheckpoint] = []
    if source_experiments:
        for source_experiment in source_experiments:
            candidates.extend(_resolve_source_candidates(source_experiment, output_dir_path, checkpoint_limit))
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
        hierarchical_names = [candidate.name for candidate in candidates if candidate.model_type == "hierarchical_deberta"]
        if hierarchical_names:
            formatted = ", ".join(hierarchical_names)
            raise ValueError(
                "Hierarchical pipeline candidates require --vote-method majority_vote because "
                f"they do not expose token-level probability tensors for averaging: {formatted}"
            )

    min_models = max(2, int(min_models))
    max_models = min(int(max_models), len(candidates))
    if min_models > max_models:
        raise ValueError(
            f"Invalid combination range: min_models={min_models}, max_models={max_models}, "
            f"candidates={len(candidates)}."
        )

    print("\n" + "=" * 60)
    print(f"  Experiment: {experiment_name} (Ensemble Search)")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Combination sizes: {min_models}..{max_models}")
    print(f"  Vote method: {vote_method}")
    print("=" * 60 + "\n")
    for candidate in candidates:
        print(f"  - {candidate.name}: {candidate.model_type} [{candidate.path.name}]")

    transformer_dataset_cache: dict[tuple[str, bool], NERDataset] = {}
    bilstm_bundle: tuple[int, BiLSTMDataset] | None = None
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
        elif candidate.model_type == "bilstm_crf":
            if bilstm_bundle is None:
                train_df = load_dataframe(data_dir_path / "new_train_data.csv")
                dev_df = load_dataframe(data_dir_path / "new_dev_data.csv")
                word2idx = build_vocab(train_df)
                bilstm_bundle = (len(word2idx), BiLSTMDataset(dev_df, word2idx))
            vocab_size, bilstm_dataset = bilstm_bundle
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
            dataset_key = (candidate.model_name or model_name, definition_prompting)
            if dataset_key not in transformer_dataset_cache:
                dev_df = load_dataframe(data_dir_path / "new_dev_data.csv")
                tokenizer = AutoTokenizer.from_pretrained(candidate.model_name or model_name)
                transformer_dataset_cache[dataset_key] = NERDataset(
                    dev_df,
                    tokenizer,
                    definition_prompting=definition_prompting,
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

    total_combinations = sum(math.comb(len(candidates), size) for size in range(min_models, max_models + 1))
    print(f"\nEvaluating {total_combinations} ensemble combinations...\n")

    all_results: list[dict[str, Any]] = []
    best_by_size: dict[int, dict[str, Any]] = {}
    best_probabilities: dict[str, list[list[np.ndarray]]] = {}
    best_predictions: dict[str, list[list[str]]] = {}
    combination_index = 0
    combination_output_dir: Path | None = None
    if save_combination_files:
        combination_output_dir = output_dir_path / f"{experiment_name}_combinations"
        combination_output_dir.mkdir(parents=True, exist_ok=True)

    for size in range(min_models, max_models + 1):
        combinations_for_size = math.comb(len(candidates), size)
        print(f"[size={size}] {combinations_for_size} combinations")
        for combination in itertools.combinations(candidates, size):
            combination_index += 1
            combination_names = [candidate.name for candidate in combination]
            if vote_method == "probability_average":
                probabilities = [candidate_probs[name] for name in combination_names]
                predictions = _decode_average_probs(probabilities)
            else:
                probabilities = None
                predictions = _majority_vote_predictions([candidate_predictions[name] for name in combination_names])
            metrics = evaluate_ner(gold_tags, predictions, print_report=False)
            record = {
                "combination_index": combination_index,
                "num_models": size,
                "models": combination_names,
                "vote_method": vote_method,
                **metrics,
            }
            all_results.append(record)

            if combination_output_dir is not None:
                combination_file = combination_output_dir / _combination_filename(combination_index, size, combination_names)
                with combination_file.open("w", encoding="utf-8") as handle:
                    json.dump(record, handle, indent=2)

            current_best = best_by_size.get(size)
            if current_best is None or record["relaxed_f1"] > current_best["relaxed_f1"]:
                best_by_size[size] = record
                if vote_method == "probability_average" and probabilities is not None:
                    best_probabilities[str(size)] = probabilities
                else:
                    best_predictions[str(size)] = predictions
                print(
                    f"  [{combination_index}/{total_combinations}] "
                    f"new best size-{size}: F1={record['relaxed_f1']:.4f} | {', '.join(combination_names)}"
                )

    ranked_results = sorted(all_results, key=lambda item: item["relaxed_f1"], reverse=True)
    best_overall = ranked_results[0]

    best_by_size_json: dict[str, Any] = {}
    for size, record in sorted(best_by_size.items()):
        if vote_method == "probability_average":
            best_by_size_json[str(size)] = _attach_bootstrap_from_probabilities(
                record,
                best_probabilities[str(size)],
                gold_tags,
                bootstrap_samples,
            )
        else:
            best_by_size_json[str(size)] = _attach_bootstrap_from_predictions(
                record,
                best_predictions[str(size)],
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
            "min_models": min_models,
            "max_models": max_models,
            "checkpoint_limit": checkpoint_limit,
            "total_combinations": total_combinations,
        },
        "best_overall": (
            _attach_bootstrap_from_probabilities(
                best_overall,
                [candidate_probs[name] for name in best_overall["models"]],
                gold_tags,
                bootstrap_samples,
            )
            if vote_method == "probability_average"
            else _attach_bootstrap_from_predictions(
                best_overall,
                _majority_vote_predictions([candidate_predictions[name] for name in best_overall["models"]]),
                gold_tags,
                bootstrap_samples,
            )
        ),
        "best_by_size": best_by_size_json,
        "top_results": ranked_results[:20],
        "all_results": ranked_results,
    }
    if combination_output_dir is not None:
        results["combination_results_dir"] = str(combination_output_dir)

    output_path = output_dir_path / f"{experiment_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nBest overall ensemble:")
    print(f"  Relaxed F1: {results['best_overall']['relaxed_f1']:.4f}")
    print(f"  Models: {', '.join(results['best_overall']['models'])}")
    print(f"\nSaved ensemble search results to {output_path}")
    return results
