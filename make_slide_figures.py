"""Render training-curve figures from outputs_new_final_clean for the slides.

Reads per-epoch JSON logs, picks the best LR per family, and produces a single
PNG with Dev Relaxed F1 and Dev Loss over epochs for the top ensemble members.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "outputs_new_final_clean"
OUT_PATH = ROOT / "slides" / "fig_training_curves.png"

FAMILIES = [
    ("DeBERTa baseline", "deberta_baseline", "#607D8B"),
    ("Recall-Boost (s42)", "recall_boost_ow02_s42", "#9C27B0"),
    ("R-Drop (s123)", "rdrop_a1_s123", "#FF9800"),
    ("FGM+SWA (s42)", "fgm05_swa_s42", "#2E7D32"),
]


def best_lr_log(prefix: str):
    best = None
    best_f1 = -1.0
    for p in LOG_DIR.glob(f"{prefix}_lr*_log.json"):
        with p.open() as f:
            entries = json.load(f)
        if not entries:
            continue
        rf1 = max(e.get("relaxed_f1", 0.0) or 0.0 for e in entries)
        if rf1 > best_f1:
            best_f1 = rf1
            best = (p, entries)
    if best is None:
        raise FileNotFoundError(prefix)
    return best, best_f1


def main():
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for label, prefix, color in FAMILIES:
        (path, entries), rf1 = best_lr_log(prefix)
        lr = path.stem.split(f"{prefix}_")[1].replace("_log", "").replace("lr", "").replace("em0", "e-")
        epochs = [e["epoch"] for e in entries]
        f1 = [e.get("relaxed_f1", 0.0) for e in entries]
        loss = [e.get("dev_loss", 0.0) for e in entries]
        disp = f"{label} @ {lr}  (best {rf1:.3f})"
        axes[0].plot(epochs, f1, marker="o", color=color, linewidth=2, label=disp)
        axes[1].plot(epochs, loss, marker="o", color=color, linewidth=2, label=label)
        print(f"{label:22s} {path.name}  best relaxed F1={rf1:.4f}")

    axes[0].set_title("Dev Relaxed F1 by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Relaxed F1")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].set_title("Validation Loss by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dev Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
