"""Evaluation utilities wrapping the official evaluation script."""

import sys
import os
import pandas as pd
import numpy as np
from typing import Optional

# Add parent dir to import official script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2"))
from evaluation_script import (
    evaluate_test_strict_ner,
    calculate_f1_per_entity_covering_all,
    _to_list,
)

from .data import ID2LABEL


def decode_predictions(predictions, word_ids_list):
    """Convert model output logits back to word-level BIO tags."""
    all_preds = []
    for pred_ids, word_ids in zip(predictions, word_ids_list):
        word_preds = {}
        for idx, wid in enumerate(word_ids):
            if wid is not None and wid not in word_preds:
                word_preds[wid] = ID2LABEL[pred_ids[idx]]
        preds = [word_preds.get(i, "O") for i in range(max(word_preds.keys()) + 1)] if word_preds else ["O"]
        all_preds.append(preds)
    return all_preds


def evaluate_ner(gold_tags: list[list[str]], pred_tags: list[list[str]], print_report: bool = True) -> dict:
    """Run both strict and relaxed evaluation."""
    # Ensure lengths match by truncating/padding predictions
    aligned_gold = []
    aligned_pred = []
    for g, p in zip(gold_tags, pred_tags):
        if len(p) < len(g):
            p = p + ["O"] * (len(g) - len(p))
        elif len(p) > len(g):
            p = p[:len(g)]
        aligned_gold.append(g)
        aligned_pred.append(p)

    # Strict evaluation
    df = pd.DataFrame({"test": [str(g) for g in aligned_gold], "prediction": [str(p) for p in aligned_pred]})
    strict_metrics = evaluate_test_strict_ner(df, gold_col="test", pred_col="prediction", print_report=False)

    # Relaxed evaluation
    relaxed_results = calculate_f1_per_entity_covering_all(aligned_gold, aligned_pred)

    results = {
        "strict_f1": strict_metrics["f1_strict"],
        "strict_precision": strict_metrics["precision_strict"],
        "strict_recall": strict_metrics["recall_strict"],
        "relaxed_f1": relaxed_results.get("Overall", {}).get("F1-Score", 0.0),
        "relaxed_precision": relaxed_results.get("Overall", {}).get("Precision", 0.0),
        "relaxed_recall": relaxed_results.get("Overall", {}).get("Recall", 0.0),
    }

    # Per-entity metrics
    for entity_type in ["ClinicalImpacts", "SocialImpacts"]:
        if entity_type in relaxed_results:
            results[f"relaxed_f1_{entity_type}"] = relaxed_results[entity_type]["F1-Score"]
            results[f"relaxed_precision_{entity_type}"] = relaxed_results[entity_type]["Precision"]
            results[f"relaxed_recall_{entity_type}"] = relaxed_results[entity_type]["Recall"]

    if print_report:
        print("=" * 60)
        print(f"  Strict F1:  {results['strict_f1']:.4f}  (P={results['strict_precision']:.4f}, R={results['strict_recall']:.4f})")
        print(f"  Relaxed F1: {results['relaxed_f1']:.4f}  (P={results['relaxed_precision']:.4f}, R={results['relaxed_recall']:.4f})")
        for entity_type in ["ClinicalImpacts", "SocialImpacts"]:
            key = f"relaxed_f1_{entity_type}"
            if key in results:
                print(f"    {entity_type}: F1={results[key]:.4f}")
        print("=" * 60)

    return results


def bootstrap_ci(gold_tags, pred_tags, n_bootstrap=1000, seed=42):
    """Compute bootstrap 95% confidence intervals for relaxed F1."""
    rng = np.random.RandomState(seed)
    n = len(gold_tags)
    f1_scores = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        g = [gold_tags[i] for i in indices]
        p = [pred_tags[i] for i in indices]
        try:
            relaxed_results = calculate_f1_per_entity_covering_all(g, p)
            f1_scores.append(relaxed_results.get("Overall", {}).get("F1-Score", 0.0))
        except Exception:
            continue

    f1_scores = np.array(f1_scores)
    return {
        "mean": float(np.mean(f1_scores)),
        "std": float(np.std(f1_scores)),
        "ci_lower": float(np.percentile(f1_scores, 2.5)),
        "ci_upper": float(np.percentile(f1_scores, 97.5)),
    }
