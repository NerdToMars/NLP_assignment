"""Hierarchical sentence-classification + NER pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .checkpoints import TopKCheckpointManager
from .data import ID2LABEL, LABEL2ID, NERDataset, load_dataframe, preprocess_tokens
from .deberta_ner import DeBERTaNER, DeBERTaNERMultiTask
from .ensemble_v2 import apply_bio_repair
from .evaluation import evaluate_ner
from .preprocessing import restore_forced_o_predictions
from .train import _clear_torch_memory, _early_stopping_status, _init_early_stopping, _update_early_stopping, train_deberta


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2" / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-large"
ENTITY_TYPES = ("ClinicalImpacts", "SocialImpacts")


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _derive_sentence_labels(tags: list[str]) -> list[float]:
    """Convert BIO tags into multi-label sentence targets."""
    has_clinical = any("ClinicalImpacts" in tag for tag in tags)
    has_social = any("SocialImpacts" in tag for tag in tags)
    return [float(has_clinical), float(has_social)]


def _row_has_any_impact(tags: list[str]) -> bool:
    return any(tag != "O" for tag in tags)


def _downsample_no_impact_rows(df, keep_ratio: float, seed: int | None) -> tuple[Any, dict[str, Any]]:
    """Keep all impact-positive rows and only a fraction of all-O rows."""
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"ner_no_impact_keep_ratio must be in [0, 1], got {keep_ratio}")

    positive_mask = df["ner_tags"].apply(_row_has_any_impact)
    positive_df = df[positive_mask].copy()
    no_impact_df = df[~positive_mask].copy()

    if keep_ratio >= 1.0 or no_impact_df.empty:
        filtered_df = df.copy().reset_index(drop=True)
        kept_no_impact_df = no_impact_df
    elif keep_ratio <= 0.0:
        kept_no_impact_df = no_impact_df.iloc[0:0].copy()
        filtered_df = positive_df.reset_index(drop=True)
    else:
        keep_count = max(1, int(round(len(no_impact_df) * keep_ratio)))
        keep_count = min(keep_count, len(no_impact_df))
        kept_no_impact_df = no_impact_df.sample(n=keep_count, random_state=seed).sort_index()
        filtered_df = pd.concat([positive_df, kept_no_impact_df], ignore_index=False).sort_index().reset_index(drop=True)

    stats = {
        "original_rows": int(len(df)),
        "positive_rows": int(len(positive_df)),
        "no_impact_rows_original": int(len(no_impact_df)),
        "no_impact_keep_ratio": float(keep_ratio),
        "no_impact_rows_kept": int(len(kept_no_impact_df)),
        "no_impact_rows_dropped": int(len(no_impact_df) - len(kept_no_impact_df)),
        "filtered_rows": int(len(filtered_df)),
    }
    return filtered_df, stats


def _safe_divide(num: float, den: float) -> float:
    return num / den if den else 0.0


def _compute_multilabel_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute compact multi-label metrics for the sentence classifier."""
    metrics: dict[str, float] = {}
    micro_tp = micro_fp = micro_fn = 0

    for label_index, label_name in enumerate(ENTITY_TYPES):
        gold_col = gold[:, label_index].astype(bool)
        pred_col = pred[:, label_index].astype(bool)
        tp = int(np.logical_and(gold_col, pred_col).sum())
        fp = int(np.logical_and(~gold_col, pred_col).sum())
        fn = int(np.logical_and(gold_col, ~pred_col).sum())

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        prefix = label_name.lower().replace("impacts", "")
        metrics[f"{prefix}_precision"] = precision
        metrics[f"{prefix}_recall"] = recall
        metrics[f"{prefix}_f1"] = f1

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro_precision = _safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_divide(micro_tp, micro_tp + micro_fn)
    metrics["micro_precision"] = micro_precision
    metrics["micro_recall"] = micro_recall
    metrics["micro_f1"] = _safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    metrics["subset_accuracy"] = float(np.mean(np.all(gold == pred, axis=1)))
    metrics["any_impact_accuracy"] = float(
        np.mean(np.any(gold == 1, axis=1) == np.any(pred == 1, axis=1))
    )
    return metrics


def _mask_predicted_tags(pred_tags: list[str], allow_clinical: bool, allow_social: bool) -> list[str]:
    """Suppress entity types rejected by the sentence classifier."""
    allowed = []
    for tag in pred_tags:
        if "ClinicalImpacts" in tag and not allow_clinical:
            allowed.append("O")
        elif "SocialImpacts" in tag and not allow_social:
            allowed.append("O")
        else:
            allowed.append(tag)
    return apply_bio_repair(allowed)


class SentenceImpactDataset(Dataset):
    """Sentence-level multi-label dataset derived from token-level annotations."""

    def __init__(self, df, tokenizer, max_length: int = 512):
        self.samples: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            tokens = preprocess_tokens(row["tokens"])
            labels = _derive_sentence_labels(row["ner_tags"])
            encoding = tokenizer(
                tokens,
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
                    "labels": torch.tensor(labels, dtype=torch.float32),
                    "tokens": tokens,
                    "raw_tags": row["ner_tags"],
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


class DeBERTaSentenceClassifier(nn.Module):
    """CLS-based multi-label sentence classifier for impact detection."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0].float())
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return {"loss": loss, "logits": logits}


def evaluate_sentence_classifier(
    model: DeBERTaSentenceClassifier,
    dataset: SentenceImpactDataset,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.5,
    print_report: bool = True,
) -> dict[str, float]:
    """Evaluate the sentence classifier."""
    model.eval()
    all_gold = []
    all_pred = []
    total_loss = 0.0
    num_batches = 0

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

            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            preds = (probs >= threshold).astype(np.int64)
            gold = batch["labels"].cpu().numpy().astype(np.int64)

            all_pred.append(preds)
            all_gold.append(gold)

    gold_array = np.concatenate(all_gold, axis=0)
    pred_array = np.concatenate(all_pred, axis=0)
    metrics = _compute_multilabel_metrics(gold_array, pred_array)
    metrics["dev_loss"] = total_loss / num_batches if num_batches > 0 else float("nan")
    metrics["positive_predictions"] = int(np.any(pred_array == 1, axis=1).sum())

    if print_report:
        print("=" * 60)
        print(
            f"  Sentence classifier micro-F1: {metrics['micro_f1']:.4f} "
            f"(P={metrics['micro_precision']:.4f}, R={metrics['micro_recall']:.4f})"
        )
        print(
            f"  Clinical F1: {metrics['clinical_f1']:.4f} | "
            f"Social F1: {metrics['social_f1']:.4f}"
        )
        print(
            f"  Subset accuracy: {metrics['subset_accuracy']:.4f} | "
            f"Any-impact accuracy: {metrics['any_impact_accuracy']:.4f}"
        )
        print("=" * 60)

    return metrics


def train_sentence_impact_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    threshold: float = 0.5,
    device: str = "cuda:0",
    experiment_name: str = "hierarchical_deberta_classifier",
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    top_k_checkpoints: int = 5,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
    seed: int | None = 42,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Train the first-stage sentence classifier."""
    _set_seed(seed)

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {experiment_name} (Sentence Classifier)")
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
    train_dataset = SentenceImpactDataset(train_df, tokenizer)
    dev_dataset = SentenceImpactDataset(dev_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    model = DeBERTaSentenceClassifier(model_name=model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    checkpoint_manager = TopKCheckpointManager(
        experiment_name=experiment_name,
        output_dir=output_dir_path,
        top_k=top_k_checkpoints,
        metadata={
            "model_type": "sentence_classifier",
            "model_name": model_name,
            "threshold": threshold,
        },
    )

    best_dev_micro_f1 = 0.0
    best_checkpoint_path = output_dir_path / f"{experiment_name}_best.pt"
    results_log: list[dict[str, Any]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Classifier Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        dev_results = evaluate_sentence_classifier(model, dev_dataset, dev_loader, device, threshold=threshold)
        print(
            f"Epoch {epoch + 1} - Train Loss: {total_loss / num_batches:.4f} | "
            f"Dev Loss: {dev_results.get('dev_loss', float('nan')):.4f}"
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            **dev_results,
        }
        results_log.append(epoch_result)

        checkpoint_manager.maybe_save_state_dict(
            model.state_dict(),
            score=dev_results["micro_f1"],
            epoch=epoch + 1,
            metrics=epoch_result,
        )

        if dev_results["micro_f1"] > best_dev_micro_f1 or not best_checkpoint_path.exists():
            best_dev_micro_f1 = dev_results["micro_f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> New best classifier micro-F1: {best_dev_micro_f1:.4f} (saved)")

        if _update_early_stopping(early_stopping, dev_results.get("dev_loss", float("nan"))):
            print(
                "  -> Early stopping triggered "
                f"(no validation-loss decrease > {early_stopping['min_delta']:.6f} for "
                f"{early_stopping['patience']} epoch(s))"
            )
            break
        if early_stopping is not None and early_stopping["epochs_without_improvement"] > 0:
            print(
                "  -> Early stopping counter: "
                f"{early_stopping['epochs_without_improvement']}/{early_stopping['patience']}"
            )

    with (output_dir_path / f"{experiment_name}_log.json").open("w", encoding="utf-8") as handle:
        json.dump(results_log, handle, indent=2)

    if best_checkpoint_path.exists():
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    final_metrics = evaluate_sentence_classifier(model, dev_dataset, dev_loader, device, threshold=threshold, print_report=False)
    print(f"\nBest classifier micro-F1: {best_dev_micro_f1:.4f}")

    optimizer.zero_grad(set_to_none=True)
    del model, optimizer, scheduler, train_loader, dev_loader
    del train_dataset, dev_dataset, tokenizer, train_df, dev_df
    _clear_torch_memory()
    return best_checkpoint_path, final_metrics, results_log


def _load_sentence_classifier(checkpoint_path: Path, model_name: str, device: str) -> DeBERTaSentenceClassifier:
    model = DeBERTaSentenceClassifier(model_name=model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def _load_ner_model(checkpoint_path: Path, model_name: str, device: str, use_multitask: bool = True):
    if use_multitask:
        model = DeBERTaNERMultiTask(model_name=model_name, num_labels=len(LABEL2ID))
    else:
        model = DeBERTaNER(model_name=model_name, num_labels=len(LABEL2ID))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def _predict_sentence_labels(
    model: DeBERTaSentenceClassifier,
    dataset: SentenceImpactDataset,
    batch_size: int,
    device: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size)
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)
            all_preds.append((probs >= threshold).astype(np.int64))

    return np.concatenate(all_probs, axis=0), np.concatenate(all_preds, axis=0)


def _predict_ner_subset(
    model,
    dataset: NERDataset,
    indices: list[int],
    batch_size: int,
    device: str,
) -> dict[int, list[str]]:
    """Run NER only on the subset selected by the sentence classifier."""
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size)
    decoded_predictions: dict[int, list[str]] = {}
    sample_ptr = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()

            for batch_index in range(preds.shape[0]):
                original_index = indices[sample_ptr]
                sample = dataset.get_full_sample(original_index)
                word_preds = {}
                for token_index, word_id in enumerate(sample["word_ids"]):
                    if word_id is not None and word_id not in word_preds:
                        word_preds[word_id] = ID2LABEL[preds[batch_index][token_index]]

                if "kept_indices" in sample and "raw_tokens" in sample:
                    cleaned_pred_tags = [
                        word_preds.get(word_index, "O")
                        for word_index in range(len(sample["kept_indices"]))
                    ]
                    decoded_predictions[original_index] = restore_forced_o_predictions(
                        sample["raw_tokens"],
                        sample["kept_indices"],
                        cleaned_pred_tags,
                    )
                else:
                    decoded_predictions[original_index] = [
                        word_preds.get(word_index, "O")
                        for word_index in range(len(sample["raw_tags"]))
                    ]
                sample_ptr += 1

    return decoded_predictions


def evaluate_hierarchical_pipeline(
    classifier_checkpoint: str | Path,
    ner_checkpoint: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    threshold: float = 0.5,
    device: str = "cuda:0",
    data_dir: str = str(DEFAULT_DATA_DIR),
) -> dict[str, Any]:
    """Evaluate the full hierarchical pipeline on the dev set."""
    data_dir_path = Path(data_dir).resolve()
    dev_df = load_dataframe(data_dir_path / "new_dev_data.csv")

    sentence_tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence_dataset = SentenceImpactDataset(dev_df, sentence_tokenizer)
    sentence_loader = DataLoader(sentence_dataset, batch_size=batch_size)

    classifier_model = _load_sentence_classifier(Path(classifier_checkpoint).resolve(), model_name, device)
    classifier_metrics = evaluate_sentence_classifier(
        classifier_model,
        sentence_dataset,
        sentence_loader,
        device,
        threshold=threshold,
        print_report=True,
    )
    _, sentence_preds = _predict_sentence_labels(
        classifier_model,
        sentence_dataset,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
    )

    ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_dataset = NERDataset(dev_df, ner_tokenizer)
    all_gold = [sample["raw_tags"] for sample in ner_dataset.samples]
    all_pred = [["O"] * len(gold_tags) for gold_tags in all_gold]

    positive_indices = [index for index, pred in enumerate(sentence_preds) if int(pred.sum()) > 0]
    if positive_indices:
        ner_model = _load_ner_model(Path(ner_checkpoint).resolve(), model_name, device, use_multitask=True)
        ner_predictions = _predict_ner_subset(ner_model, ner_dataset, positive_indices, batch_size, device)

        for sample_index in positive_indices:
            sample_pred = ner_predictions.get(sample_index, all_pred[sample_index])
            allow_clinical = bool(sentence_preds[sample_index][0])
            allow_social = bool(sentence_preds[sample_index][1])
            all_pred[sample_index] = _mask_predicted_tags(sample_pred, allow_clinical, allow_social)

        del ner_model
        _clear_torch_memory()

    hierarchical_metrics = evaluate_ner(all_gold, all_pred, print_report=True)
    hierarchical_metrics["ner_invocations"] = len(positive_indices)
    hierarchical_metrics["skip_rate"] = 1.0 - (len(positive_indices) / len(all_gold) if all_gold else 0.0)
    hierarchical_metrics["classifier_positive_predictions"] = int(np.any(sentence_preds == 1, axis=1).sum())

    del classifier_model, sentence_loader, sentence_dataset, sentence_tokenizer, ner_dataset, ner_tokenizer, dev_df
    _clear_torch_memory()
    return {
        "classifier_metrics": classifier_metrics,
        "hierarchical_metrics": hierarchical_metrics,
        "threshold": threshold,
    }


def run_hierarchical_deberta(
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    threshold: float = 0.5,
    device: str = "cuda:0",
    experiment_name: str = "hierarchical_deberta",
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    top_k_checkpoints: int = 5,
    seed: int | None = 42,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
    ner_no_impact_keep_ratio: float = 0.1,
) -> dict[str, Any]:
    """Train a sentence classifier, then run NER only on predicted-positive samples."""
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    classifier_experiment = f"{experiment_name}_classifier"
    ner_experiment = f"{experiment_name}_ner"

    classifier_checkpoint, classifier_dev_metrics, _ = train_sentence_impact_classifier(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        threshold=threshold,
        device=device,
        experiment_name=classifier_experiment,
        data_dir=data_dir,
        output_dir=output_dir,
        top_k_checkpoints=top_k_checkpoints,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        seed=seed,
    )

    ner_train_df = load_dataframe(Path(data_dir).resolve() / "new_train_data.csv")
    ner_train_df, ner_sampling = _downsample_no_impact_rows(
        ner_train_df,
        keep_ratio=ner_no_impact_keep_ratio,
        seed=seed,
    )
    print("\n" + "=" * 60)
    print("  Hierarchical NER training data filtering")
    print(
        "  Keeping all impact rows and "
        f"{ner_sampling['no_impact_rows_kept']}/{ner_sampling['no_impact_rows_original']} "
        f"no-impact rows ({ner_sampling['no_impact_keep_ratio']:.1%})"
    )
    print(
        f"  Filtered NER train rows: {ner_sampling['filtered_rows']} "
        f"(from {ner_sampling['original_rows']})"
    )
    print("=" * 60)

    train_deberta(
        model_name=model_name,
        use_multitask=True,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        experiment_name=ner_experiment,
        top_k_checkpoints=top_k_checkpoints,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        train_df_override=ner_train_df,
    )
    ner_checkpoint = output_dir_path / f"{ner_experiment}_best.pt"
    if not ner_checkpoint.exists():
        raise FileNotFoundError(f"Expected NER checkpoint not found: {ner_checkpoint}")

    evaluation = evaluate_hierarchical_pipeline(
        classifier_checkpoint=classifier_checkpoint,
        ner_checkpoint=ner_checkpoint,
        model_name=model_name,
        batch_size=batch_size,
        threshold=threshold,
        device=device,
        data_dir=data_dir,
    )
    hierarchical_metrics = evaluation["hierarchical_metrics"]

    results = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "threshold": threshold,
        "classifier_checkpoint": str(classifier_checkpoint),
        "ner_checkpoint": str(ner_checkpoint),
        "classifier_metrics": evaluation["classifier_metrics"],
        "classifier_dev_metrics": classifier_dev_metrics,
        "hierarchical_metrics": hierarchical_metrics,
        "ner_training_sampling": ner_sampling,
        "relaxed_f1": hierarchical_metrics["relaxed_f1"],
        "strict_f1": hierarchical_metrics["strict_f1"],
        "relaxed_precision": hierarchical_metrics["relaxed_precision"],
        "relaxed_recall": hierarchical_metrics["relaxed_recall"],
    }

    results_path = output_dir_path / f"{experiment_name}_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nHierarchical pipeline results saved to {results_path}")
    return results
