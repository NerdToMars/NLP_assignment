"""Utilities for weight-space model soups built from saved checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .bilstm_crf import BiLSTMCRF
from .data import BiLSTMDataset, NERDataset, NUM_LABELS, build_vocab, load_dataframe
from .deberta_crf import DeBERTaCRF, DeBERTaCRFMultiTask
from .deberta_ner import DeBERTaNER, DeBERTaNERMultiTask
from .train import evaluate_bilstm, evaluate_model_crf, evaluate_model_deberta


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2" / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

SUPPORTED_MODEL_TYPES = {
    "deberta",
    "deberta_multitask",
    "deberta_crf",
    "deberta_crf_multitask",
    "bilstm_crf",
}


def _average_state_dicts(checkpoint_paths: list[Path]) -> dict[str, torch.Tensor]:
    if len(checkpoint_paths) < 2:
        raise ValueError("Model soup needs at least two checkpoints.")

    avg_state: dict[str, torch.Tensor] | None = None
    expected_keys: set[str] | None = None
    loaded = 0

    for checkpoint_path in checkpoint_paths:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_keys = set(state.keys())
        if expected_keys is None:
            expected_keys = state_keys
            avg_state = {key: value.float().clone() for key, value in state.items()}
        else:
            if state_keys != expected_keys:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} is incompatible with the current soup. "
                    "Make sure all checkpoints come from the same architecture."
                )
            for key in avg_state:
                avg_state[key] += state[key].float()
        loaded += 1

    assert avg_state is not None
    for key in avg_state:
        avg_state[key] /= loaded
    return avg_state


def _load_source_experiments(
    source_experiments: list[str],
    output_dir: Path,
    checkpoint_limit: int,
) -> tuple[list[Path], dict[str, Any], list[dict[str, Any]]]:
    if checkpoint_limit < 1:
        raise ValueError("--checkpoint-limit must be at least 1.")

    checkpoint_paths: list[Path] = []
    source_details: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None

    for source_experiment in source_experiments:
        summary_path = output_dir / "checkpoints" / source_experiment / "topk_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Top-k summary not found for source experiment '{source_experiment}': {summary_path}"
            )

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        source_metadata = summary.get("metadata", {})
        source_model_type = source_metadata.get("model_type")
        if source_model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Model soup currently supports {sorted(SUPPORTED_MODEL_TYPES)}, "
                f"but '{source_experiment}' has model_type='{source_model_type}'."
            )

        if metadata is None:
            metadata = source_metadata
        else:
            if metadata.get("model_type") != source_model_type:
                raise ValueError(
                    "All source experiments in a soup must share the same model_type. "
                    f"Got '{metadata.get('model_type')}' and '{source_model_type}'."
                )
            if metadata.get("model_name") != source_metadata.get("model_name"):
                raise ValueError(
                    "All source experiments in a soup must share the same model_name. "
                    f"Got '{metadata.get('model_name')}' and '{source_metadata.get('model_name')}'."
                )

        selected = summary.get("checkpoints", [])[:checkpoint_limit]
        if not selected:
            raise ValueError(f"No saved checkpoints found for source experiment '{source_experiment}'.")

        source_paths = [Path(item["path"]).resolve() for item in selected]
        checkpoint_paths.extend(source_paths)
        source_details.append(
            {
                "source_experiment": source_experiment,
                "checkpoint_paths": [str(path) for path in source_paths],
            }
        )

    assert metadata is not None
    return checkpoint_paths, metadata, source_details


def _build_transformer_eval_bundle(
    data_dir: Path,
    model_name: str,
    batch_size: int,
) -> tuple[Any, Any, Any]:
    dev_df = load_dataframe(data_dir / "new_dev_data.csv")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dev_dataset = NERDataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    return tokenizer, dev_dataset, dev_loader


def _build_bilstm_eval_bundle(data_dir: Path, batch_size: int) -> tuple[Any, Any, Any]:
    train_df = load_dataframe(data_dir / "new_train_data.csv")
    dev_df = load_dataframe(data_dir / "new_dev_data.csv")
    word2idx = build_vocab(train_df)
    dev_dataset = BiLSTMDataset(dev_df, word2idx)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    return word2idx, dev_dataset, dev_loader


def run_model_soup(
    source_experiments: list[str] | None = None,
    checkpoint_names: list[str] | None = None,
    checkpoint_limit: int = 5,
    model_type: str | None = None,
    model_name: str = "microsoft/deberta-v3-large",
    batch_size: int = 8,
    device: str = "cuda:0",
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    experiment_name: str = "model_soup",
) -> dict[str, Any]:
    """Average compatible checkpoints in weight space and evaluate the soup."""

    output_dir_path = Path(output_dir).resolve()
    data_dir_path = Path(data_dir).resolve()
    checkpoint_paths: list[Path] = []
    source_details: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    if source_experiments:
        source_paths, source_metadata, source_details = _load_source_experiments(
            source_experiments,
            output_dir_path,
            checkpoint_limit,
        )
        checkpoint_paths.extend(source_paths)
        metadata.update(source_metadata)

    if checkpoint_names:
        if model_type is None and not metadata.get("model_type"):
            raise ValueError("When using --checkpoint directly, you must also provide --model-type.")
        checkpoint_paths.extend(Path(path).resolve() for path in checkpoint_names)

    if len(checkpoint_paths) < 2:
        raise ValueError("Model soup needs at least two checkpoints after resolution.")

    model_type = model_type or metadata.get("model_type")
    model_name = metadata.get("model_name", model_name)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported or missing model_type '{model_type}'. "
            f"Supported values: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    print("\n" + "=" * 60)
    print(f"  Experiment: {experiment_name} (Model Soup)")
    print(f"  Model type: {model_type}")
    print(f"  Checkpoints: {len(checkpoint_paths)}")
    print("=" * 60 + "\n")

    avg_state = _average_state_dicts(checkpoint_paths)

    if model_type == "deberta":
        _, dev_dataset, dev_loader = _build_transformer_eval_bundle(data_dir_path, model_name, batch_size)
        model = DeBERTaNER(model_name=model_name, num_labels=NUM_LABELS)
        model.load_state_dict(avg_state)
        model = model.to(device)
        metrics = evaluate_model_deberta(model, dev_dataset, dev_loader, device)
    elif model_type == "deberta_multitask":
        _, dev_dataset, dev_loader = _build_transformer_eval_bundle(data_dir_path, model_name, batch_size)
        model = DeBERTaNERMultiTask(model_name=model_name, num_labels=NUM_LABELS)
        model.load_state_dict(avg_state)
        model = model.to(device)
        metrics = evaluate_model_deberta(model, dev_dataset, dev_loader, device)
    elif model_type == "deberta_crf":
        _, dev_dataset, dev_loader = _build_transformer_eval_bundle(data_dir_path, model_name, batch_size)
        model = DeBERTaCRF(
            model_name=model_name,
            num_labels=NUM_LABELS,
            use_lstm=bool(metadata.get("use_lstm", False)),
        )
        model.load_state_dict(avg_state)
        model = model.to(device)
        metrics = evaluate_model_crf(model, dev_dataset, dev_loader, device)
    elif model_type == "deberta_crf_multitask":
        _, dev_dataset, dev_loader = _build_transformer_eval_bundle(data_dir_path, model_name, batch_size)
        model = DeBERTaCRFMultiTask(model_name=model_name, num_labels=NUM_LABELS)
        model.load_state_dict(avg_state)
        model = model.to(device)
        metrics = evaluate_model_crf(model, dev_dataset, dev_loader, device)
    elif model_type == "bilstm_crf":
        _, dev_dataset, dev_loader = _build_bilstm_eval_bundle(data_dir_path, batch_size)
        train_df = load_dataframe(data_dir_path / "new_train_data.csv")
        word2idx = build_vocab(train_df)
        model = BiLSTMCRF(vocab_size=len(word2idx), num_tags=NUM_LABELS)
        model.load_state_dict(avg_state)
        model = model.to(device)
        metrics = evaluate_bilstm(model, dev_dataset, dev_loader, device)
    else:
        raise AssertionError(f"Unexpected model_type: {model_type}")

    save_path = output_dir_path / f"{experiment_name}_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    results = {
        **metrics,
        "experiment_name": experiment_name,
        "model_type": model_type,
        "model_name": model_name,
        "metadata": metadata,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "source_experiments": source_details,
        "saved_model_path": str(save_path),
    }
    with (output_dir_path / f"{experiment_name}_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Soup model saved to {save_path}")
    return results
