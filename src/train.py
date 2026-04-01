"""Training and evaluation script for all models."""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm

from .data import (
    load_dataframe, NERDataset, BiLSTMDataset, build_vocab, load_glove_embeddings,
    LABEL2ID, ID2LABEL, NUM_LABELS,
)
from .evaluation import evaluate_ner, decode_predictions, bootstrap_ci
from .bilstm_crf import BiLSTMCRF
from .deberta_ner import DeBERTaNER, DeBERTaNERMultiTask, SpanNER, FocalLoss
from .deberta_crf import DeBERTaCRF, DeBERTaCRFMultiTask
from .synthetic_data import generate_synthetic_data, get_curriculum_order

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def get_entity_weights(df, smoothing=0.3):
    """Compute per-class weights to address distribution mismatch."""
    counts = {label: 0 for label in LABEL2ID}
    for _, row in df.iterrows():
        for tag in row["ner_tags"]:
            counts[tag] += 1

    total = sum(counts.values())
    weights = []
    for label in ["O", "B-ClinicalImpacts", "I-ClinicalImpacts", "B-SocialImpacts", "I-SocialImpacts"]:
        freq = counts[label] / total
        w = 1.0 / (freq + 1e-8)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    # Apply smoothing
    weights = weights ** smoothing
    weights = weights / weights.sum() * len(weights)

    # Boost Social Impacts to account for test set distribution (70% Social)
    weights[3] *= 1.5  # B-SocialImpacts
    weights[4] *= 1.5  # I-SocialImpacts
    return weights


def derive_entity_presence_labels(labels_batch):
    """Derive binary entity presence labels from BIO tags."""
    presence = []
    for labels in labels_batch:
        has_entity = any(l.item() > 0 for l in labels if l.item() != -100)
        presence.append(1 if has_entity else 0)
    return torch.tensor(presence, dtype=torch.long)


def train_deberta(
    model_name="microsoft/deberta-v3-large",
    use_focal_loss=False,
    definition_prompting=False,
    use_multitask=False,
    use_synthetic=False,
    use_curriculum=False,
    epochs=10,
    batch_size=8,
    lr=2e-5,
    device="cuda:0",
    experiment_name="deberta_baseline",
):
    """Train DeBERTa-based NER model with optional innovations."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Model: {model_name}")
    print(f"  Focal Loss: {use_focal_loss} | Definitions: {definition_prompting}")
    print(f"  Multi-task: {use_multitask} | Synthetic: {use_synthetic} | Curriculum: {use_curriculum}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    train_df = load_dataframe(os.path.join(DATA_DIR, "new_train_data.csv"))
    dev_df = load_dataframe(os.path.join(DATA_DIR, "new_dev_data.csv"))

    # Add synthetic data if requested
    if use_synthetic:
        synthetic = generate_synthetic_data(n_per_category=50)
        synth_df = pd.DataFrame(synthetic)
        synth_df["labels"] = synth_df["labels"].apply(str)
        train_df = pd.concat([train_df, synth_df[["tokens", "ner_tags", "ID"]]], ignore_index=True)
        # Re-parse since concat might mess up types
        if not isinstance(train_df["tokens"].iloc[0], list):
            from .data import parse_list_col
            train_df["tokens"] = train_df["tokens"].apply(
                lambda x: x if isinstance(x, list) else parse_list_col(x)
            )
            train_df["ner_tags"] = train_df["ner_tags"].apply(
                lambda x: x if isinstance(x, list) else parse_list_col(x)
            )
        print(f"Training with {len(train_df)} samples (including synthetic)")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Datasets
    train_dataset = NERDataset(train_df, tokenizer, definition_prompting=definition_prompting)
    dev_dataset = NERDataset(dev_df, tokenizer, definition_prompting=definition_prompting)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=not use_curriculum)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # Model
    focal_alpha = None
    if use_focal_loss:
        focal_alpha = get_entity_weights(train_df).tolist()

    if use_multitask:
        model = DeBERTaNERMultiTask(
            model_name=model_name,
            num_labels=NUM_LABELS,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
        )
    else:
        model = DeBERTaNER(
            model_name=model_name,
            num_labels=NUM_LABELS,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
        )

    model = model.to(device)
    # Move focal loss alpha tensor to device
    loss_fn = getattr(model, 'loss_fn', None) or getattr(model, 'ner_loss_fn', None)
    if loss_fn is not None and hasattr(loss_fn, 'alpha') and loss_fn.alpha is not None:
        loss_fn.alpha = loss_fn.alpha.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_dev_f1 = 0.0
    results_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # Curriculum learning: adjust data order
        if use_curriculum and epoch > 0:
            indices = get_curriculum_order(train_dataset.samples, epoch, epochs)
            subset = Subset(train_dataset, indices)
            train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_multitask:
                entity_presence = derive_entity_presence_labels(batch["labels"]).to(device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    entity_presence_labels=entity_presence,
                )
            else:
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
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        # Evaluate on dev set
        dev_results = evaluate_model_deberta(model, dev_dataset, dev_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/num_batches:.4f}")

        results_log.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            **dev_results,
        })

        if dev_results["relaxed_f1"] > best_dev_f1:
            best_dev_f1 = dev_results["relaxed_f1"]
            save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best dev Relaxed F1: {best_dev_f1:.4f} (saved)")

    # Save results log
    with open(os.path.join(OUTPUT_DIR, f"{experiment_name}_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\nBest dev Relaxed F1: {best_dev_f1:.4f}")
    return best_dev_f1, results_log


def evaluate_model_deberta(model, dataset, dataloader, device):
    """Evaluate DeBERTa model and return metrics."""
    model.eval()
    all_gold = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].numpy()

            for i in range(preds.shape[0]):
                sample_idx = len(all_gold)
                if sample_idx >= len(dataset.samples):
                    break
                sample = dataset.get_full_sample(sample_idx)
                word_ids = sample["word_ids"]
                raw_tags = sample["raw_tags"]

                # Decode back to word-level
                word_preds = {}
                for idx, wid in enumerate(word_ids):
                    if wid is not None and wid not in word_preds:
                        word_preds[wid] = ID2LABEL[preds[i][idx]]

                pred_tags = [word_preds.get(j, "O") for j in range(len(raw_tags))]
                all_gold.append(raw_tags)
                all_pred.append(pred_tags)

    return evaluate_ner(all_gold, all_pred, print_report=True)


def train_bilstm_crf(
    glove_path=None,
    epochs=30,
    batch_size=32,
    lr=1e-3,
    device="cuda:0",
    experiment_name="bilstm_crf",
):
    """Train BiLSTM-CRF baseline."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name} (BiLSTM-CRF)")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = load_dataframe(os.path.join(DATA_DIR, "new_train_data.csv"))
    dev_df = load_dataframe(os.path.join(DATA_DIR, "new_dev_data.csv"))

    # Build vocab
    word2idx = build_vocab(train_df)
    print(f"Vocabulary size: {len(word2idx)}")

    # Load embeddings
    pretrained_emb = None
    if glove_path and os.path.exists(glove_path):
        pretrained_emb = load_glove_embeddings(glove_path, word2idx)

    # Datasets
    train_dataset = BiLSTMDataset(train_df, word2idx)
    dev_dataset = BiLSTMDataset(dev_df, word2idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    model = BiLSTMCRF(
        vocab_size=len(word2idx),
        num_tags=NUM_LABELS,
        pretrained_embeddings=pretrained_emb,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_f1 = 0.0
    results_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch["input_ids"], labels=batch["labels"], length=batch["length"])
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        # Evaluate
        dev_results = evaluate_bilstm(model, dev_dataset, dev_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/num_batches:.4f}")

        results_log.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            **dev_results,
        })

        if dev_results["relaxed_f1"] > best_dev_f1:
            best_dev_f1 = dev_results["relaxed_f1"]
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{experiment_name}_best.pt"))
            print(f"  -> New best dev Relaxed F1: {best_dev_f1:.4f}")

    with open(os.path.join(OUTPUT_DIR, f"{experiment_name}_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    return best_dev_f1, results_log


def evaluate_bilstm(model, dataset, dataloader, device):
    """Evaluate BiLSTM-CRF model."""
    model.eval()
    all_gold = []
    all_pred = []

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch_device["input_ids"])
            pred_ids = outputs["predictions"].cpu().numpy()

            for i in range(pred_ids.shape[0]):
                if sample_idx >= len(dataset.samples):
                    break
                s = dataset.samples[sample_idx]
                length = s["length"]
                raw_tags = s["raw_tags"]

                pred_tags = [ID2LABEL[pred_ids[i][j]] for j in range(min(length, len(raw_tags)))]
                # Pad if needed
                while len(pred_tags) < len(raw_tags):
                    pred_tags.append("O")

                all_gold.append(raw_tags)
                all_pred.append(pred_tags[:len(raw_tags)])
                sample_idx += 1

    return evaluate_ner(all_gold, all_pred, print_report=True)


def train_deberta_crf(
    model_name="microsoft/deberta-v3-large",
    use_multitask=False,
    use_lstm=False,
    epochs=15,
    batch_size=8,
    lr=2e-5,
    encoder_lr=None,
    warmup_ratio=0.1,
    gradient_accumulation_steps=1,
    device="cuda:0",
    experiment_name="deberta_crf",
):
    """Train DeBERTa + CRF model."""
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  CRF Model | Multi-task: {use_multitask} | LSTM: {use_lstm}")
    print(f"  LR: {lr} | Encoder LR: {encoder_lr} | Epochs: {epochs}")
    print(f"  Batch size: {batch_size} | Grad accum: {gradient_accumulation_steps}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = load_dataframe(os.path.join(DATA_DIR, "new_train_data.csv"))
    dev_df = load_dataframe(os.path.join(DATA_DIR, "new_dev_data.csv"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = NERDataset(train_df, tokenizer)
    dev_dataset = NERDataset(dev_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    if use_multitask:
        model = DeBERTaCRFMultiTask(model_name=model_name, num_labels=NUM_LABELS)
    else:
        model = DeBERTaCRF(model_name=model_name, num_labels=NUM_LABELS, use_lstm=use_lstm)
    model = model.to(device)

    # Differential learning rates: lower LR for encoder, higher for CRF/classifier
    if encoder_lr is not None:
        encoder_params = list(model.encoder.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        optimizer = AdamW([
            {"params": encoder_params, "lr": encoder_lr},
            {"params": other_params, "lr": lr},
        ], weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps
    )

    best_dev_f1 = 0.0
    results_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_multitask:
                entity_presence = derive_entity_presence_labels(batch["labels"]).to(device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    entity_presence_labels=entity_presence,
                )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

            loss = outputs["loss"] / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        dev_results = evaluate_model_crf(model, dev_dataset, dev_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/num_batches:.4f}")

        results_log.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            **dev_results,
        })

        if dev_results["relaxed_f1"] > best_dev_f1:
            best_dev_f1 = dev_results["relaxed_f1"]
            save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best dev Relaxed F1: {best_dev_f1:.4f} (saved)")

    with open(os.path.join(OUTPUT_DIR, f"{experiment_name}_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\nBest dev Relaxed F1: {best_dev_f1:.4f}")
    return best_dev_f1, results_log


def evaluate_model_crf(model, dataset, dataloader, device):
    """Evaluate DeBERTa+CRF model using Viterbi decoding."""
    model.eval()
    all_gold = []
    all_pred = []

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )

            # CRF model returns Viterbi decoded predictions
            if "predictions" in outputs:
                preds = outputs["predictions"].cpu().numpy()
            else:
                preds = outputs["logits"].argmax(dim=-1).cpu().numpy()

            for i in range(preds.shape[0]):
                if sample_idx >= len(dataset.samples):
                    break
                sample = dataset.get_full_sample(sample_idx)
                word_ids = sample["word_ids"]
                raw_tags = sample["raw_tags"]

                word_preds = {}
                for idx, wid in enumerate(word_ids):
                    if wid is not None and wid not in word_preds:
                        word_preds[wid] = ID2LABEL[preds[i][idx]]

                pred_tags = [word_preds.get(j, "O") for j in range(len(raw_tags))]
                all_gold.append(raw_tags)
                all_pred.append(pred_tags)
                sample_idx += 1

    return evaluate_ner(all_gold, all_pred, print_report=True)


def train_deberta_recall_boost(
    model_name="microsoft/deberta-v3-large",
    epochs=15,
    batch_size=4,
    lr=2e-5,
    gradient_accumulation_steps=4,
    o_weight=0.3,
    device="cuda:0",
    experiment_name="deberta_recall_boost",
    use_multitask=True,
    seed=42,
):
    """Train DeBERTa with aggressive O-class downweighting to boost recall."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  O-weight: {o_weight} | LR: {lr} | Epochs: {epochs}")
    print(f"  Batch: {batch_size} x {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = load_dataframe(os.path.join(DATA_DIR, "new_train_data.csv"))
    dev_df = load_dataframe(os.path.join(DATA_DIR, "new_dev_data.csv"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = NERDataset(train_df, tokenizer)
    dev_dataset = NERDataset(dev_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # Class weights: aggressively downweight O to boost entity recall
    # O=0.3, B-Clinical=1.0, I-Clinical=1.0, B-Social=1.5, I-Social=1.5
    class_weights = torch.tensor(
        [o_weight, 1.0, 1.0, 1.5, 1.5], dtype=torch.float32
    ).to(device)

    if use_multitask:
        model = DeBERTaNERMultiTask(model_name=model_name, num_labels=NUM_LABELS)
    else:
        model = DeBERTaNER(model_name=model_name, num_labels=NUM_LABELS)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_dev_f1 = 0.0
    results_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass through encoder once
            encoder_out = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            sequence_output = model.dropout(encoder_out.last_hidden_state.float())

            if use_multitask:
                logits = model.ner_classifier(sequence_output)
            else:
                logits = model.classifier(sequence_output)

            # Custom weighted CE loss
            loss = F.cross_entropy(
                logits.view(-1, NUM_LABELS),
                batch["labels"].view(-1),
                weight=class_weights,
                ignore_index=-100,
            )

            if use_multitask:
                entity_presence = derive_entity_presence_labels(batch["labels"]).to(device)
                cls_output = sequence_output[:, 0]
                presence_logits = model.entity_presence_head(cls_output)
                presence_loss = F.cross_entropy(presence_logits, entity_presence)
                loss = loss + 0.3 * presence_loss

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        dev_results = evaluate_model_deberta(model, dev_dataset, dev_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/num_batches:.4f}")

        results_log.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            **dev_results,
        })

        if dev_results["relaxed_f1"] > best_dev_f1:
            best_dev_f1 = dev_results["relaxed_f1"]
            save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best dev Relaxed F1: {best_dev_f1:.4f} (saved)")

    with open(os.path.join(OUTPUT_DIR, f"{experiment_name}_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\nBest dev Relaxed F1: {best_dev_f1:.4f}")
    return best_dev_f1, results_log


def train_deberta_rdrop(
    model_name="microsoft/deberta-v3-large",
    epochs=15,
    batch_size=4,
    lr=1.5e-5,
    gradient_accumulation_steps=4,
    o_weight=0.2,
    rdrop_alpha=1.0,
    device="cuda:0",
    experiment_name="deberta_rdrop",
    seed=42,
):
    """Train DeBERTa with R-Drop consistency regularization + class reweighting."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  R-Drop alpha: {rdrop_alpha} | O-weight: {o_weight}")
    print(f"  LR: {lr} | Epochs: {epochs} | Seed: {seed}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = load_dataframe(os.path.join(DATA_DIR, "new_train_data.csv"))
    dev_df = load_dataframe(os.path.join(DATA_DIR, "new_dev_data.csv"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = NERDataset(train_df, tokenizer)
    dev_dataset = NERDataset(dev_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    class_weights = torch.tensor(
        [o_weight, 1.0, 1.0, 1.5, 1.5], dtype=torch.float32
    ).to(device)

    model = DeBERTaNERMultiTask(model_name=model_name, num_labels=NUM_LABELS)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_dev_f1 = 0.0
    results_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Two forward passes with different dropout masks
            encoder_out1 = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            seq1 = model.dropout(encoder_out1.last_hidden_state.float())
            logits1 = model.ner_classifier(seq1)

            encoder_out2 = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            seq2 = model.dropout(encoder_out2.last_hidden_state.float())
            logits2 = model.ner_classifier(seq2)

            # CE loss (average of both passes)
            ce1 = F.cross_entropy(
                logits1.view(-1, NUM_LABELS), batch["labels"].view(-1),
                weight=class_weights, ignore_index=-100,
            )
            ce2 = F.cross_entropy(
                logits2.view(-1, NUM_LABELS), batch["labels"].view(-1),
                weight=class_weights, ignore_index=-100,
            )

            # KL divergence between the two passes (R-Drop)
            mask = (batch["labels"].view(-1) != -100)
            p1 = F.log_softmax(logits1.view(-1, NUM_LABELS)[mask], dim=-1)
            p2 = F.log_softmax(logits2.view(-1, NUM_LABELS)[mask], dim=-1)
            q1 = F.softmax(logits1.view(-1, NUM_LABELS)[mask], dim=-1)
            q2 = F.softmax(logits2.view(-1, NUM_LABELS)[mask], dim=-1)
            kl_loss = 0.5 * (F.kl_div(p1, q2, reduction='batchmean') +
                             F.kl_div(p2, q1, reduction='batchmean'))

            # Aux loss
            entity_presence = derive_entity_presence_labels(batch["labels"]).to(device)
            cls_output = seq1[:, 0]
            presence_logits = model.entity_presence_head(cls_output)
            aux_loss = F.cross_entropy(presence_logits, entity_presence)

            loss = 0.5 * (ce1 + ce2) + rdrop_alpha * kl_loss + 0.3 * aux_loss

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        dev_results = evaluate_model_deberta(model, dev_dataset, dev_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/num_batches:.4f}")

        results_log.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            **dev_results,
        })

        if dev_results["relaxed_f1"] > best_dev_f1:
            best_dev_f1 = dev_results["relaxed_f1"]
            save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best dev Relaxed F1: {best_dev_f1:.4f} (saved)")

    with open(os.path.join(OUTPUT_DIR, f"{experiment_name}_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\nBest dev Relaxed F1: {best_dev_f1:.4f}")
    return best_dev_f1, results_log


def train_deberta_fgm_swa(
    model_name="microsoft/deberta-v3-large",
    epochs=15,
    batch_size=4,
    lr=1.5e-5,
    gradient_accumulation_steps=4,
    o_weight=0.2,
    fgm_epsilon=0.5,
    swa_start_epoch=10,
    device="cuda:0",
    experiment_name="deberta_fgm_swa",
    seed=42,
):
    """Train with FGM adversarial perturbation + SWA weight averaging."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  FGM epsilon: {fgm_epsilon} | SWA start: {swa_start_epoch}")
    print(f"  O-weight: {o_weight} | LR: {lr} | Epochs: {epochs} | Seed: {seed}")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = load_dataframe(os.path.join(DATA_DIR, "new_train_data.csv"))
    dev_df = load_dataframe(os.path.join(DATA_DIR, "new_dev_data.csv"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = NERDataset(train_df, tokenizer)
    dev_dataset = NERDataset(dev_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    class_weights = torch.tensor(
        [o_weight, 1.0, 1.0, 1.5, 1.5], dtype=torch.float32
    ).to(device)

    model = DeBERTaNERMultiTask(model_name=model_name, num_labels=NUM_LABELS)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    # SWA state
    swa_state_dict = None
    swa_count = 0

    best_dev_f1 = 0.0
    results_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            encoder_out = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            seq = model.dropout(encoder_out.last_hidden_state.float())
            logits = model.ner_classifier(seq)

            loss = F.cross_entropy(
                logits.view(-1, NUM_LABELS), batch["labels"].view(-1),
                weight=class_weights, ignore_index=-100,
            )

            # Aux loss
            entity_presence = derive_entity_presence_labels(batch["labels"]).to(device)
            presence_logits = model.entity_presence_head(seq[:, 0])
            aux_loss = F.cross_entropy(presence_logits, entity_presence)
            loss = loss + 0.3 * aux_loss

            loss_scaled = loss / gradient_accumulation_steps
            loss_scaled.backward()

            # FGM adversarial perturbation
            if fgm_epsilon > 0:
                # Save perturbations for proper restoration
                saved_perturbations = {}
                for name_p, param in model.encoder.named_parameters():
                    if 'word_embeddings' in name_p and param.grad is not None:
                        norm = torch.norm(param.grad)
                        if norm != 0:
                            r_at = fgm_epsilon * param.grad / norm
                            saved_perturbations[name_p] = r_at.clone()
                            param.data.add_(r_at)

                # Adversarial forward pass
                encoder_out_adv = model.encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                seq_adv = model.dropout(encoder_out_adv.last_hidden_state.float())
                logits_adv = model.ner_classifier(seq_adv)
                loss_adv = F.cross_entropy(
                    logits_adv.view(-1, NUM_LABELS), batch["labels"].view(-1),
                    weight=class_weights, ignore_index=-100,
                )
                loss_adv_scaled = loss_adv / gradient_accumulation_steps
                loss_adv_scaled.backward()

                # Restore embeddings using saved perturbations
                for name_p, param in model.encoder.named_parameters():
                    if name_p in saved_perturbations:
                        param.data.sub_(saved_perturbations[name_p])

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        # SWA: accumulate model weights after swa_start_epoch
        if epoch >= swa_start_epoch:
            if swa_state_dict is None:
                swa_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state_dict:
                    swa_state_dict[k] += model.state_dict()[k]
                swa_count += 1

        dev_results = evaluate_model_deberta(model, dev_dataset, dev_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/num_batches:.4f}")

        results_log.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_batches,
            **dev_results,
        })

        if dev_results["relaxed_f1"] > best_dev_f1:
            best_dev_f1 = dev_results["relaxed_f1"]
            save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best dev Relaxed F1: {best_dev_f1:.4f} (saved)")

    # Evaluate SWA model
    if swa_state_dict is not None and swa_count > 0:
        print(f"\nEvaluating SWA model (averaged over {swa_count} epochs)...")
        avg_state = {k: v / swa_count for k, v in swa_state_dict.items()}
        model.load_state_dict(avg_state)
        swa_results = evaluate_model_deberta(model, dev_dataset, dev_loader, device)
        swa_f1 = swa_results["relaxed_f1"]
        print(f"SWA Relaxed F1: {swa_f1:.4f}")

        if swa_f1 > best_dev_f1:
            best_dev_f1 = swa_f1
            save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_swa_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> SWA is better! Saved as {save_path}")
        else:
            save_path = os.path.join(OUTPUT_DIR, f"{experiment_name}_swa.pt")
            torch.save(model.state_dict(), save_path)

    with open(os.path.join(OUTPUT_DIR, f"{experiment_name}_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\nBest dev Relaxed F1: {best_dev_f1:.4f}")
    return best_dev_f1, results_log


def ensemble_evaluate(model_paths, model_class, model_kwargs, dev_dataset, dev_loader, device):
    """Ensemble evaluation: average logits from multiple models."""
    all_logits = []

    for path in model_paths:
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        batch_logits = []
        with torch.no_grad():
            for batch in dev_loader:
                batch_device = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch_device["input_ids"],
                    attention_mask=batch_device["attention_mask"],
                )
                batch_logits.append(outputs["logits"].cpu())

        all_logits.append(torch.cat(batch_logits, dim=0))
        del model
        torch.cuda.empty_cache()

    # Average logits
    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = avg_logits.argmax(dim=-1).numpy()

    # Decode to word-level tags
    all_gold = []
    all_pred = []
    for i in range(preds.shape[0]):
        if i >= len(dev_dataset.samples):
            break
        sample = dev_dataset.get_full_sample(i)
        word_ids = sample["word_ids"]
        raw_tags = sample["raw_tags"]

        word_preds = {}
        for idx, wid in enumerate(word_ids):
            if wid is not None and wid not in word_preds:
                word_preds[wid] = ID2LABEL[preds[i][idx]]

        pred_tags = [word_preds.get(j, "O") for j in range(len(raw_tags))]
        all_gold.append(raw_tags)
        all_pred.append(pred_tags)

    return evaluate_ner(all_gold, all_pred, print_report=True)


def fix_bio_tags(tags):
    """Post-process BIO tags to fix invalid transitions."""
    fixed = list(tags)
    for i, tag in enumerate(fixed):
        if tag.startswith("I-"):
            entity = tag[2:]
            if i == 0 or (not fixed[i-1].endswith(entity)):
                fixed[i] = f"B-{entity}"
    return fixed


def evaluate_model_deberta_with_postprocess(model, dataset, dataloader, device):
    """Evaluate DeBERTa model with BIO post-processing."""
    model.eval()
    all_gold = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1).cpu().numpy()

            for i in range(preds.shape[0]):
                sample_idx = len(all_gold)
                if sample_idx >= len(dataset.samples):
                    break
                sample = dataset.get_full_sample(sample_idx)
                word_ids = sample["word_ids"]
                raw_tags = sample["raw_tags"]

                word_preds = {}
                for idx, wid in enumerate(word_ids):
                    if wid is not None and wid not in word_preds:
                        word_preds[wid] = ID2LABEL[preds[i][idx]]

                pred_tags = [word_preds.get(j, "O") for j in range(len(raw_tags))]
                pred_tags = fix_bio_tags(pred_tags)
                all_gold.append(raw_tags)
                all_pred.append(pred_tags)

    return evaluate_ner(all_gold, all_pred, print_report=True)


def run_gliner_experiment(device="cuda:0", experiment_name="gliner"):
    """Run GLiNER zero-shot and fine-tuned experiments."""
    from gliner import GLiNER

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name} (GLiNER)")
    print(f"{'='*60}\n")

    dev_df = load_dataframe(os.path.join(DATA_DIR, "new_dev_data.csv"))

    # Load GLiNER model
    model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
    model = model.to(device)

    # Use specific sub-labels that GLiNER can detect, then map to our types
    clinical_labels = ["addiction", "overdose", "withdrawal", "depression", "anxiety",
                       "hospitalization", "medical treatment", "mental illness", "drug abuse",
                       "relapse", "substance abuse", "psychosis", "detox", "rehab"]
    social_labels = ["arrest", "criminal charge", "job loss", "divorce", "eviction",
                     "homelessness", "incarceration", "probation", "financial hardship",
                     "family breakdown", "relationship breakdown", "legal trouble",
                     "dropping out", "unemployment"]

    clinical_set = set(clinical_labels)
    social_set = set(social_labels)
    all_labels = clinical_labels + social_labels

    all_gold = []
    all_pred = []

    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="GLiNER inference"):
        tokens = row["tokens"]
        raw_tags = row["ner_tags"]
        text = " ".join(tokens)

        try:
            entities = model.predict_entities(text, all_labels, threshold=0.3)
        except Exception:
            entities = []

        # Convert entity spans to BIO tags
        pred_tags = ["O"] * len(tokens)
        # Map character offsets to token indices
        char_to_token = {}
        offset = 0
        for i, t in enumerate(tokens):
            for c in range(offset, offset + len(t)):
                char_to_token[c] = i
            offset += len(t) + 1  # +1 for space

        for ent in entities:
            start_char = ent.get("start", 0)
            end_char = ent.get("end", 0)
            raw_label = ent.get("label", "")

            # Map to our entity types
            if raw_label in clinical_set:
                label = "ClinicalImpacts"
            elif raw_label in social_set:
                label = "SocialImpacts"
            else:
                continue

            token_indices = sorted(set(
                char_to_token.get(c, -1) for c in range(start_char, end_char)
            ))
            token_indices = [t for t in token_indices if 0 <= t < len(tokens)]

            if token_indices:
                # Don't overwrite existing predictions
                if pred_tags[token_indices[0]] == "O":
                    pred_tags[token_indices[0]] = f"B-{label}"
                    for ti in token_indices[1:]:
                        if pred_tags[ti] == "O":
                            pred_tags[ti] = f"I-{label}"

        all_gold.append(raw_tags)
        all_pred.append(pred_tags)

    results = evaluate_ner(all_gold, all_pred, print_report=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{experiment_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_all_experiments(device="cuda:0"):
    """Run the complete ablation study."""
    all_results = {}

    # 1. BiLSTM-CRF
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: BiLSTM-CRF Baseline")
    print("=" * 80)
    f1, log = train_bilstm_crf(device=device, epochs=30)
    all_results["bilstm_crf"] = {"best_dev_f1": f1, "log": log}

    # 2. DeBERTa-large baseline
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: DeBERTa-large Baseline")
    print("=" * 80)
    f1, log = train_deberta(
        device=device, epochs=10, experiment_name="deberta_baseline",
    )
    all_results["deberta_baseline"] = {"best_dev_f1": f1, "log": log}

    # 3. DeBERTa + Focal Loss (Innovation 1)
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: DeBERTa + Distribution-Aware Training")
    print("=" * 80)
    f1, log = train_deberta(
        device=device, epochs=10, use_focal_loss=True,
        experiment_name="deberta_focal",
    )
    all_results["deberta_focal"] = {"best_dev_f1": f1, "log": log}

    # 4. DeBERTa + Definition Prompting (Innovation 2)
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: DeBERTa + Definition Prompting")
    print("=" * 80)
    f1, log = train_deberta(
        device=device, epochs=10, definition_prompting=True,
        experiment_name="deberta_definition",
    )
    all_results["deberta_definition"] = {"best_dev_f1": f1, "log": log}

    # 5. DeBERTa + Multi-task (Innovation 3)
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: DeBERTa + Auxiliary Tasks")
    print("=" * 80)
    f1, log = train_deberta(
        device=device, epochs=10, use_multitask=True,
        experiment_name="deberta_multitask",
    )
    all_results["deberta_multitask"] = {"best_dev_f1": f1, "log": log}

    # 6. DeBERTa + Synthetic Data + Curriculum (Innovation 4)
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: DeBERTa + Synthetic Data + Curriculum")
    print("=" * 80)
    f1, log = train_deberta(
        device=device, epochs=10, use_synthetic=True, use_curriculum=True,
        experiment_name="deberta_synthetic_curriculum",
    )
    all_results["deberta_synthetic_curriculum"] = {"best_dev_f1": f1, "log": log}

    # 7. DeBERTa Combined (all innovations)
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: DeBERTa Combined (All Innovations)")
    print("=" * 80)
    f1, log = train_deberta(
        device=device, epochs=10,
        use_focal_loss=True,
        definition_prompting=True,
        use_multitask=True,
        use_synthetic=True,
        use_curriculum=True,
        experiment_name="deberta_combined",
    )
    all_results["deberta_combined"] = {"best_dev_f1": f1, "log": log}

    # 8. GLiNER zero-shot
    print("\n" + "=" * 80)
    print("EXPERIMENT 8: GLiNER Zero-Shot")
    print("=" * 80)
    gliner_results = run_gliner_experiment(device=device)
    all_results["gliner_zeroshot"] = gliner_results

    # Save all results
    summary = {}
    for name, res in all_results.items():
        if "best_dev_f1" in res:
            summary[name] = res["best_dev_f1"]
        elif "relaxed_f1" in res:
            summary[name] = res["relaxed_f1"]

    with open(os.path.join(OUTPUT_DIR, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    for name, f1 in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:40s} Relaxed F1: {f1:.4f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "bilstm", "deberta_baseline", "deberta_focal",
                                 "deberta_definition", "deberta_multitask",
                                 "deberta_synthetic", "deberta_combined",
                                 "gliner", "span_ner",
                                 "deberta_crf", "deberta_crf_multitask",
                                 "deberta_crf_lstm", "hp_sweep", "ensemble"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.experiment == "all":
        run_all_experiments(device=args.device)
    elif args.experiment == "bilstm":
        train_bilstm_crf(device=args.device, epochs=args.epochs)
    elif args.experiment == "deberta_baseline":
        train_deberta(device=args.device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.experiment == "deberta_focal":
        train_deberta(device=args.device, epochs=args.epochs, use_focal_loss=True,
                      experiment_name="deberta_focal", batch_size=args.batch_size, lr=args.lr)
    elif args.experiment == "deberta_definition":
        train_deberta(device=args.device, epochs=args.epochs, definition_prompting=True,
                      experiment_name="deberta_definition", batch_size=args.batch_size, lr=args.lr)
    elif args.experiment == "deberta_multitask":
        train_deberta(device=args.device, epochs=args.epochs, use_multitask=True,
                      experiment_name="deberta_multitask", batch_size=args.batch_size, lr=args.lr)
    elif args.experiment == "deberta_synthetic":
        train_deberta(device=args.device, epochs=args.epochs, use_synthetic=True, use_curriculum=True,
                      experiment_name="deberta_synthetic_curriculum", batch_size=args.batch_size, lr=args.lr)
    elif args.experiment == "deberta_combined":
        train_deberta(device=args.device, epochs=args.epochs,
                      use_focal_loss=True, definition_prompting=True, use_multitask=True,
                      use_synthetic=True, use_curriculum=True,
                      experiment_name="deberta_combined", batch_size=args.batch_size, lr=args.lr)
    elif args.experiment == "gliner":
        run_gliner_experiment(device=args.device)
    elif args.experiment == "deberta_crf":
        train_deberta_crf(device=args.device, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, experiment_name="deberta_crf")
    elif args.experiment == "deberta_crf_multitask":
        train_deberta_crf(device=args.device, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, use_multitask=True, experiment_name="deberta_crf_multitask")
    elif args.experiment == "deberta_crf_lstm":
        train_deberta_crf(device=args.device, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, use_lstm=True, experiment_name="deberta_crf_lstm")
    elif args.experiment == "hp_sweep":
        # Hyperparameter sweep for best config
        best_f1 = 0
        best_config = {}
        configs = [
            {"lr": 1e-5, "epochs": 15, "name": "hp_lr1e5_e15"},
            {"lr": 2e-5, "epochs": 15, "name": "hp_lr2e5_e15"},
            {"lr": 3e-5, "epochs": 15, "name": "hp_lr3e5_e15"},
            {"lr": 2e-5, "epochs": 20, "name": "hp_lr2e5_e20"},
            {"lr": 1e-5, "epochs": 20, "name": "hp_lr1e5_e20"},
        ]
        for cfg in configs:
            f1, _ = train_deberta(
                device=args.device, epochs=cfg["epochs"],
                lr=cfg["lr"], use_multitask=True,
                experiment_name=cfg["name"], batch_size=args.batch_size,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_config = cfg
            print(f"Config {cfg['name']}: F1={f1:.4f}")
        print(f"\nBest config: {best_config} with F1={best_f1:.4f}")
    elif args.experiment == "ensemble":
        # Multi-seed ensemble
        seeds = [42, 123, 456]
        all_preds_list = []
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        dev_df = load_dataframe(os.path.join(DATA_DIR, "new_dev_data.csv"))
        dev_dataset = NERDataset(dev_df, tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            f1, _ = train_deberta(
                device=args.device, epochs=15, lr=2e-5,
                use_multitask=True,
                experiment_name=f"ensemble_seed{seed}",
                batch_size=args.batch_size,
            )
            print(f"Seed {seed}: F1={f1:.4f}")
