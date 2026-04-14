"""GLiNER fine-tuning for SMM4H-HeaRD NER task."""

import math
import json
import shutil
import numpy as np
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data import load_dataframe, ID2LABEL, LABEL2ID
from .evaluation import evaluate_ner
from .checkpoints import TopKCheckpointManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2" / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

GLINER_BASE_MODEL = "urchade/gliner_large-v2.1"

ENTITY_LABELS = ["ClinicalImpacts", "SocialImpacts"]


def _build_gliner_collate_fn(model, entity_types):
    """Create a stable collate function for GLiNER span batches."""
    collator = model.data_collator_class(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
    )

    def collate_fn(batch_examples):
        per_example_entity_types = [list(entity_types) for _ in batch_examples]
        return collator(batch_examples, entity_types=per_example_entity_types)

    return collate_fn


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _compute_grad_norm(parameters):
    total_norm_sq = 0.0
    has_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total_norm_sq += float(grad.norm(2).item() ** 2)
        has_grad = True
    if not has_grad:
        return 0.0
    return total_norm_sq ** 0.5


def _evaluate_gliner_epoch_loss(model, dataloader, device):
    """Evaluate the GLiNER training loss on a dataloader."""
    model.eval()
    losses = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if str(device).startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            with autocast_ctx:
                outputs = model(**batch, reduction="mean")
                loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                continue
            losses.append(float(loss.detach().item()))

    if not losses:
        return float("inf")
    return float(sum(losses) / len(losses))


def _save_gliner_model(model, save_path):
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))


def _init_early_stopping(early_stopping_patience=5, early_stopping_min_delta=0.0):
    """Create mutable early-stopping state or disable it."""
    if early_stopping_patience is None:
        return None

    patience = int(early_stopping_patience)
    if patience <= 0:
        return None

    return {
        "patience": patience,
        "min_delta": float(early_stopping_min_delta),
        "monitor": "eval_loss",
        "mode": "min",
        "best_value": float("inf"),
        "epochs_without_improvement": 0,
    }


def _early_stopping_status(early_stopping):
    """Human-readable description for logs."""
    if early_stopping is None:
        return "disabled"
    return (
        f"monitor={early_stopping['monitor']}, "
        f"mode={early_stopping['mode']}, "
        f"patience={early_stopping['patience']}, "
        f"min_delta={early_stopping['min_delta']}"
    )


def _update_early_stopping(early_stopping, score):
    """Update early-stopping state and return whether training should stop."""
    if early_stopping is None:
        return False

    if score is not None and math.isfinite(score):
        improved = score < early_stopping["best_value"] - early_stopping["min_delta"]
    else:
        improved = False

    if improved:
        early_stopping["best_value"] = float(score)
        early_stopping["epochs_without_improvement"] = 0
        return False

    early_stopping["epochs_without_improvement"] += 1
    return early_stopping["epochs_without_improvement"] >= early_stopping["patience"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def convert_bio_to_gliner_format(df, skip_empty=False):
    """
    Convert a BIO-tagged DataFrame to the GLiNER training format.

    GLiNER expects:
        [{"tokenized_text": [...], "ner": [[start, end, "Label"], ...]}, ...]

    where start/end are inclusive token indices.
    """
    data = []
    skipped_empty = 0
    for _, row in df.iterrows():
        tokens = row["tokens"] if isinstance(row["tokens"], list) else eval(row["tokens"])
        tags   = row["ner_tags"] if isinstance(row["ner_tags"], list) else eval(row["ner_tags"])

        ner_spans = []
        start, etype = None, None
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if etype is not None:
                    ner_spans.append([start, i - 1, etype])
                etype, start = tag[2:], i
            elif tag == "O":
                if etype is not None:
                    ner_spans.append([start, i - 1, etype])
                    etype, start = None, None
        if etype is not None:
            ner_spans.append([start, len(tags) - 1, etype])

        if skip_empty and not ner_spans:
            skipped_empty += 1
            continue

        data.append({"tokenized_text": tokens, "ner": ner_spans})
    return data, skipped_empty


def _entities_to_bio(tokens, entities):
    """
    Convert GLiNER entity predictions (character-level spans) to BIO token tags.
    """
    from .ensemble_v2 import apply_bio_repair

    char_to_token = {}
    char_pos = 0
    for tok_idx, tok in enumerate(tokens):
        for _ in tok:
            char_to_token[char_pos] = tok_idx
            char_pos += 1
        char_to_token[char_pos] = tok_idx   # space
        char_pos += 1

    bio = ["O"] * len(tokens)
    for ent in entities:
        start_char = ent.get("start", -1)
        end_char   = ent.get("end",   -1)
        label      = ent.get("label", "")
        if label not in ENTITY_LABELS:
            continue

        start_tok = char_to_token.get(start_char)
        end_tok   = char_to_token.get(end_char - 1)
        if start_tok is None or end_tok is None:
            continue

        bio[start_tok] = f"B-{label}"
        for t in range(start_tok + 1, end_tok + 1):
            bio[t] = f"I-{label}"

    return apply_bio_repair(bio)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_gliner(model, dev_df, device, threshold=0.4, print_report=True):
    """Evaluate a GLiNER model on the dev set and return metrics."""
    model.to(device)
    all_gold, all_pred = [], []

    for _, row in dev_df.iterrows():
        tokens = row["tokens"] if isinstance(row["tokens"], list) else eval(row["tokens"])
        gold   = row["ner_tags"] if isinstance(row["ner_tags"], list) else eval(row["ner_tags"])
        all_gold.append(gold)

        try:
            entities = model.predict_entities(" ".join(tokens), ENTITY_LABELS, threshold=threshold)
        except Exception:
            entities = []

        all_pred.append(_entities_to_bio(tokens, entities))

    return evaluate_ner(all_gold, all_pred, print_report=print_report)


# ── Training ──────────────────────────────────────────────────────────────────

def finetune_gliner(
    base_model=GLINER_BASE_MODEL,
    epochs=5,
    batch_size=8,
    lr=1e-5,
    threshold=0.4,
    device="cuda:0",
    experiment_name="gliner_finetune",
    data_dir=str(DEFAULT_DATA_DIR),
    output_dir=str(DEFAULT_OUTPUT_DIR),
    model_output_dir=None,
    top_k_checkpoints=5,
    early_stopping_patience=5,
    early_stopping_min_delta=0.0,
):
    """Fine-tune GLiNER with a local PyTorch loop and evaluate on dev."""
    try:
        from gliner import GLiNER
    except ImportError:
        print("gliner is not installed. Run: pip install gliner")
        return None, []

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name} (GLiNER fine-tuning)")
    print(f"  Base model : {base_model}")
    print(f"  Epochs: {epochs} | LR: {lr} | Batch: {batch_size} | Threshold: {threshold}")
    early_stopping = _init_early_stopping(early_stopping_patience, early_stopping_min_delta)
    print(f"  Early stopping: {_early_stopping_status(early_stopping)}")
    print(f"{'='*60}\n")

    data_dir_path = Path(data_dir).resolve()
    output_dir_path = Path(output_dir).resolve()
    model_output_dir_path = (
        Path(model_output_dir).resolve()
        if model_output_dir is not None
        else output_dir_path / experiment_name
    )
    strict_model_output_dir_path = output_dir_path / f"{experiment_name}_strict_best"

    output_dir_path.mkdir(parents=True, exist_ok=True)
    model_output_dir_path.mkdir(parents=True, exist_ok=True)

    checkpoint_manager = TopKCheckpointManager(
        experiment_name=experiment_name,
        output_dir=output_dir_path,
        top_k=top_k_checkpoints,
        metadata={
            "model_type": "gliner",
            "model_name": base_model,
            "threshold": threshold,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        },
    )

    train_df = load_dataframe(data_dir_path / "new_train_data.csv")
    dev_df = load_dataframe(data_dir_path / "new_dev_data.csv")

    train_data, skipped_train_empty = convert_bio_to_gliner_format(train_df, skip_empty=False)
    dev_data, skipped_dev_empty = convert_bio_to_gliner_format(dev_df, skip_empty=False)
    print(
        f"Train: {len(train_data)} GLiNER samples "
        f"(skipped {skipped_train_empty} empty-span rows)  |  "
        f"Dev: {len(dev_data)} GLiNER samples "
        f"(skipped {skipped_dev_empty} empty-span rows)\n"
    )
    if not train_data:
        raise RuntimeError("GLiNER training data is empty.")
    if not dev_data:
        raise RuntimeError("GLiNER validation data is empty.")

    model = GLiNER.from_pretrained(base_model)
    model = model.to(device)

    # Use the full fixed label set so all-O examples remain valid instead of
    # producing zero-width label tensors inside GLiNER's collator.
    train_collate_fn = _build_gliner_collate_fn(model, ENTITY_LABELS)
    dev_collate_fn = _build_gliner_collate_fn(model, ENTITY_LABELS)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_collate_fn,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dev_collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    total_steps = max(1, len(train_loader) * max(1, int(epochs)))
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    use_autocast = str(device).startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_autocast)

    best_dev_f1 = 0.0
    best_dev_strict_f1 = 0.0
    best_dev_loss = float("inf")
    results_log = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses = []
        grad_norms = []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = _move_batch_to_device(batch, device)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_autocast
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(**batch, reduction="mean")
                loss = outputs.loss

            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(
                    "GLiNER custom training loop produced an invalid loss. "
                    "Stopping instead of silently skipping the batch."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm_value = (
                float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(float(loss.detach().item()))
            grad_norms.append(grad_norm_value)

        train_loss = float(sum(train_losses) / len(train_losses)) if train_losses else float("nan")
        grad_norm = float(sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0
        dev_loss = _evaluate_gliner_epoch_loss(model, dev_loader, device)
        dev_results = evaluate_gliner(model, dev_df, device, threshold=threshold, print_report=False)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "grad_norm": grad_norm,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **dev_results,
        }
        results_log.append(log_entry)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_loss={dev_loss:.4f} | "
            f"grad_norm={grad_norm:.4f} | "
            f"relaxed_f1={dev_results['relaxed_f1']:.4f} | "
            f"strict_f1={dev_results['strict_f1']:.4f}"
        )

        checkpoint_manager.maybe_save_directory(
            lambda path: _save_gliner_model(model, path),
            score=dev_results["relaxed_f1"],
            epoch=epoch,
            metrics=log_entry,
        )

        if dev_results["relaxed_f1"] >= best_dev_f1:
            best_dev_f1 = float(dev_results["relaxed_f1"])

        if dev_results["strict_f1"] >= best_dev_strict_f1:
            best_dev_strict_f1 = float(dev_results["strict_f1"])
            _save_gliner_model(model, strict_model_output_dir_path)
            print(f"  -> Saved best strict-F1 checkpoint to {strict_model_output_dir_path}")

        if dev_loss < best_dev_loss:
            best_dev_loss = float(dev_loss)
            _save_gliner_model(model, model_output_dir_path)
            print(f"  -> Saved best validation-loss checkpoint to {model_output_dir_path}")

        should_stop = _update_early_stopping(early_stopping, dev_loss)
        if early_stopping is not None:
            print(
                f"  -> Early stopping best dev_loss={early_stopping['best_value']:.4f}; "
                f"epochs_without_improvement={early_stopping['epochs_without_improvement']}/"
                f"{early_stopping['patience']}"
            )
        if should_stop:
            print(f"  -> Early stopping triggered at epoch {epoch}")
            break

    model = GLiNER.from_pretrained(str(model_output_dir_path))
    model = model.to(device)
    final_results = evaluate_gliner(model, dev_df, device, threshold=threshold)
    best_dev_f1 = float(final_results["relaxed_f1"])

    with (output_dir_path / f"{experiment_name}_log.json").open("w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)

    print(f"\nBest dev Relaxed F1: {best_dev_f1:.4f}")
    print(f"Best dev Strict F1:  {best_dev_strict_f1:.4f}")
    return best_dev_f1, results_log


# ── Inference ─────────────────────────────────────────────────────────────────

def run_gliner_inference(
    model_dir=None,
    threshold=0.4,
    device="cuda:0",
    experiment_name="gliner_inference",
    data_dir=str(DEFAULT_DATA_DIR),
    output_dir=str(DEFAULT_OUTPUT_DIR),
):
    """Load the fine-tuned GLiNER checkpoint and evaluate on dev."""
    try:
        from gliner import GLiNER
    except ImportError:
        print("gliner is not installed.")
        return None, []

    output_dir_path = Path(output_dir).resolve()
    if model_dir is None:
        model_dir = output_dir_path / "gliner_finetune"
    model_dir = Path(model_dir).resolve()

    if not model_dir.exists():
        print(f"Fine-tuned model not found at {model_dir}.")
        print("Run finetune_gliner() first.")
        return None, []

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name} (GLiNER inference)")
    print(f"  Model dir  : {model_dir}")
    print(f"  Threshold  : {threshold}")
    print(f"{'='*60}\n")

    output_dir_path.mkdir(parents=True, exist_ok=True)

    dev_df = load_dataframe(Path(data_dir).resolve() / "new_dev_data.csv")

    model = GLiNER.from_pretrained(str(model_dir))
    model = model.to(device)

    dev_results = evaluate_gliner(model, dev_df, device, threshold=threshold, print_report=True)

    print(f"\nFine-tuned GLiNER dev Relaxed F1: {dev_results['relaxed_f1']:.4f}")

    results_log = [{"epoch": 0, "train_loss": float("nan"), **dev_results}]

    with (output_dir_path / f"{experiment_name}_log.json").open("w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)

    return dev_results["relaxed_f1"], results_log
