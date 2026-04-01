"""Deep error analysis to identify improvement opportunities."""

import os
import json
import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import load_dataframe, NERDataset, LABEL2ID, ID2LABEL, NUM_LABELS
from .deberta_ner import DeBERTaNER, DeBERTaNERMultiTask
from .evaluation import evaluate_ner, bootstrap_ci

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def run_inference(model, dataset, device):
    """Run model inference and return gold/pred pairs."""
    model.eval()
    loader = DataLoader(dataset, batch_size=8)
    all_gold, all_pred, all_tokens = [], [], []

    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1).cpu().numpy()
            # Also get probabilities for confidence analysis
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for i in range(preds.shape[0]):
                if sample_idx >= len(dataset.samples):
                    break
                sample = dataset.get_full_sample(sample_idx)
                word_ids = sample["word_ids"]
                raw_tags = sample["raw_tags"]
                raw_tokens = sample["raw_tokens"]

                word_preds = {}
                word_probs = {}
                for idx, wid in enumerate(word_ids):
                    if wid is not None and wid not in word_preds:
                        word_preds[wid] = ID2LABEL[preds[i][idx]]
                        word_probs[wid] = probs[i][idx]

                pred_tags = [word_preds.get(j, "O") for j in range(len(raw_tags))]
                pred_probs = [word_probs.get(j, np.zeros(NUM_LABELS)) for j in range(len(raw_tags))]

                all_gold.append(raw_tags)
                all_pred.append(pred_tags)
                all_tokens.append(raw_tokens)
                sample_idx += 1

    return all_gold, all_pred, all_tokens


def extract_spans(tags):
    """Extract entity spans from BIO tags."""
    spans = []
    start, etype = None, None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if etype is not None:
                spans.append((start, i - 1, etype))
            etype = tag[2:]
            start = i
        elif tag.startswith("I-"):
            if etype is None or tag[2:] != etype:
                if etype is not None:
                    spans.append((start, i - 1, etype))
                etype = tag[2:]
                start = i
        else:
            if etype is not None:
                spans.append((start, i - 1, etype))
                etype = None
    if etype is not None:
        spans.append((start, len(tags) - 1, etype))
    return spans


def detailed_error_analysis(all_gold, all_pred, all_tokens):
    """Comprehensive error analysis."""
    stats = {
        "total_samples": len(all_gold),
        "samples_with_entities": 0,
        "samples_without_entities": 0,
        "gold_spans": {"ClinicalImpacts": 0, "SocialImpacts": 0},
        "pred_spans": {"ClinicalImpacts": 0, "SocialImpacts": 0},
        "true_positives": {"ClinicalImpacts": 0, "SocialImpacts": 0},
        "false_negatives": {"ClinicalImpacts": 0, "SocialImpacts": 0},
        "false_positives": {"ClinicalImpacts": 0, "SocialImpacts": 0},
        "type_confusion": 0,
        "boundary_errors": 0,
        "missed_by_length": defaultdict(int),
        "missed_examples": [],
        "fp_examples": [],
        "boundary_error_examples": [],
    }

    for gold_tags, pred_tags, tokens in zip(all_gold, all_pred, all_tokens):
        gold_spans = extract_spans(gold_tags)
        pred_spans = extract_spans(pred_tags)

        has_entity = len(gold_spans) > 0
        if has_entity:
            stats["samples_with_entities"] += 1
        else:
            stats["samples_without_entities"] += 1

        for s, e, t in gold_spans:
            stats["gold_spans"][t] += 1
        for s, e, t in pred_spans:
            stats["pred_spans"][t] += 1

        # Match gold to pred
        matched_pred = set()
        for gs, ge, gt in gold_spans:
            best_match = None
            best_overlap = 0
            for pi, (ps, pe, pt) in enumerate(pred_spans):
                if pi in matched_pred:
                    continue
                overlap = max(0, min(ge, pe) - max(gs, ps) + 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (pi, ps, pe, pt)

            if best_match is not None and best_overlap > 0:
                pi, ps, pe, pt = best_match
                matched_pred.add(pi)
                if pt == gt:
                    stats["true_positives"][gt] += 1
                    if gs != ps or ge != pe:
                        stats["boundary_errors"] += 1
                        span_text = " ".join(tokens[gs:ge+1])
                        pred_text = " ".join(tokens[ps:pe+1])
                        if len(stats["boundary_error_examples"]) < 10:
                            stats["boundary_error_examples"].append({
                                "gold": f"[{gs}:{ge}] {span_text}",
                                "pred": f"[{ps}:{pe}] {pred_text}",
                                "type": gt,
                            })
                else:
                    stats["type_confusion"] += 1
            else:
                # False negative
                stats["false_negatives"][gt] += 1
                span_len = ge - gs + 1
                stats["missed_by_length"][span_len] += 1
                span_text = " ".join(tokens[gs:ge+1])
                context = " ".join(tokens[max(0,gs-3):min(len(tokens),ge+4)])
                if len(stats["missed_examples"]) < 20:
                    stats["missed_examples"].append({
                        "span": span_text,
                        "context": context,
                        "type": gt,
                        "length": span_len,
                    })

        # False positives
        for pi, (ps, pe, pt) in enumerate(pred_spans):
            if pi not in matched_pred:
                stats["false_positives"][pt] += 1
                span_text = " ".join(tokens[ps:pe+1])
                if len(stats["fp_examples"]) < 10:
                    stats["fp_examples"].append({
                        "span": span_text,
                        "type": pt,
                        "context": " ".join(tokens[max(0,ps-3):min(len(tokens),pe+4)]),
                    })

    return stats


def analyze_model(model_name, model_class, checkpoint_path, device="cuda:0", **model_kwargs):
    """Run full analysis on a trained model."""
    print(f"\n{'='*70}")
    print(f"  Analyzing: {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    dev_df = load_dataframe(os.path.join(DATA_DIR, "new_dev_data.csv"))
    dataset = NERDataset(dev_df, tokenizer)

    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    all_gold, all_pred, all_tokens = run_inference(model, dataset, device)

    # Metrics
    metrics = evaluate_ner(all_gold, all_pred, print_report=True)

    # Bootstrap CI
    ci = bootstrap_ci(all_gold, all_pred)
    print(f"\nBootstrap 95% CI for Relaxed F1: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # Detailed errors
    stats = detailed_error_analysis(all_gold, all_pred, all_tokens)

    print(f"\n--- Entity-Level Statistics ---")
    for etype in ["ClinicalImpacts", "SocialImpacts"]:
        g = stats["gold_spans"][etype]
        p = stats["pred_spans"][etype]
        tp = stats["true_positives"][etype]
        fn = stats["false_negatives"][etype]
        fp = stats["false_positives"][etype]
        prec = tp / max(p, 1)
        rec = tp / max(g, 1)
        print(f"  {etype}: gold={g}, pred={p}, TP={tp}, FN={fn}, FP={fp}, P={prec:.3f}, R={rec:.3f}")

    print(f"\n  Boundary errors: {stats['boundary_errors']}")
    print(f"  Type confusion: {stats['type_confusion']}")

    print(f"\n--- Missed Spans by Length ---")
    for length in sorted(stats["missed_by_length"].keys()):
        print(f"  Length {length}: {stats['missed_by_length'][length]} missed")

    print(f"\n--- Top Missed Entity Examples ---")
    for ex in stats["missed_examples"][:10]:
        print(f"  [{ex['type']}] \"{ex['span']}\" (len={ex['length']})")
        print(f"    Context: ...{ex['context']}...")

    print(f"\n--- False Positive Examples ---")
    for ex in stats["fp_examples"][:5]:
        print(f"  [{ex['type']}] \"{ex['span']}\"")
        print(f"    Context: ...{ex['context']}...")

    print(f"\n--- Boundary Error Examples ---")
    for ex in stats["boundary_error_examples"][:5]:
        print(f"  [{ex['type']}] Gold: {ex['gold']} -> Pred: {ex['pred']}")

    return metrics, stats


def analyze_sentence_length_effect(all_gold, all_pred, all_tokens):
    """Analyze how sentence length affects performance."""
    bins = [(0, 15), (15, 30), (30, 50), (50, 100), (100, 999)]
    for lo, hi in bins:
        g_sub = [g for g, t in zip(all_gold, all_tokens) if lo <= len(t) < hi]
        p_sub = [p for p, t in zip(all_pred, all_tokens) if lo <= len(t) < hi]
        if g_sub:
            metrics = evaluate_ner(g_sub, p_sub, print_report=False)
            print(f"  Length [{lo}-{hi}): {len(g_sub)} samples, Relaxed F1: {metrics['relaxed_f1']:.4f}")


if __name__ == "__main__":
    device = "cuda:0"

    # Analyze best model (multitask)
    metrics, stats = analyze_model(
        "DeBERTa Multi-task (best)",
        DeBERTaNERMultiTask,
        os.path.join(OUTPUT_DIR, "deberta_multitask_best.pt"),
        device=device,
        num_labels=NUM_LABELS,
    )

    # Also analyze baseline for comparison
    metrics_base, stats_base = analyze_model(
        "DeBERTa Baseline",
        DeBERTaNER,
        os.path.join(OUTPUT_DIR, "deberta_baseline_best.pt"),
        device=device,
        num_labels=NUM_LABELS,
    )
