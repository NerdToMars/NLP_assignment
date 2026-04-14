"""Generate a competition submission CSV using the 5-model ensemble.

Mirrors the ensemble pattern in notebook.ipynb (probability averaging over
FGM+SWA, R-Drop×2 seeds, recall-boost×2 seeds) and writes one row per test
example with the predicted BIO tags, matching sample_submission.csv format.
"""
import argparse
import ast
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data import ID2LABEL, NUM_LABELS, NERDataset
from src.deberta_ner import DeBERTaNERMultiTask

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs_new_final_clean"
DATA_DIR = ROOT / "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2" / "dataset"
SUBMISSION_OUT = ROOT / "submission.csv"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# (display name, log-file prefix, checkpoint suffix) for each ensemble member.
# We read each model's *_lr*_log.json and pick the LR with the highest relaxed_f1.
ENSEMBLE_MEMBERS = [
    ("fgm05_swa_s42", "fgm05_swa_s42", "swa.pt"),
    ("rdrop_a1_s42", "rdrop_a1_s42", "best.pt"),
    ("rdrop_a1_s123", "rdrop_a1_s123", "best.pt"),
    ("recall_boost_ow02_s42", "recall_boost_ow02_s42", "best.pt"),
    ("recall_boost_ow02_s123", "recall_boost_ow02_s123", "best.pt"),
]


def pick_best_lr(prefix: str) -> str:
    """Return the LR suffix (e.g. 'lr2em05') with the highest relaxed F1."""
    best_f1 = -1.0
    best_suffix = None
    for log_path in OUTPUT_DIR.glob(f"{prefix}_lr*_log.json"):
        with log_path.open() as f:
            log = json.load(f)
        if not log:
            continue
        rf1 = max(entry.get("relaxed_f1", 0.0) for entry in log)
        if rf1 > best_f1:
            best_f1 = rf1
            # extract e.g. "lr2em05" from "fgm05_swa_s42_lr2em05_log.json"
            stem = log_path.stem  # "fgm05_swa_s42_lr2em05_log"
            assert stem.endswith("_log")
            stem = stem[: -len("_log")]
            best_suffix = stem[len(prefix) + 1 :]  # skip the underscore
    if best_suffix is None:
        raise FileNotFoundError(f"No logs found for {prefix}")
    print(f"  {prefix}: best LR={best_suffix} (relaxed F1 {best_f1:.4f})")
    return best_suffix


def resolve_checkpoints():
    ckpts = []
    for display, prefix, ckpt_suffix in ENSEMBLE_MEMBERS:
        lr_suffix = pick_best_lr(prefix)
        ckpt = OUTPUT_DIR / f"{prefix}_{lr_suffix}_{ckpt_suffix}"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        ckpts.append((display, ckpt))
    return ckpts


def run_model(ckpt_path: Path, dev_loader: DataLoader) -> torch.Tensor:
    model = DeBERTaNERMultiTask(num_labels=NUM_LABELS).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for batch in dev_loader:
            batch_device = {k: v.to(DEVICE) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            all_logits.append(out["logits"].cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(all_logits, dim=0)


def load_input_df(csv_path: Path) -> pd.DataFrame:
    """Read the competition CSV and ensure ner_tags exist (dummy 'O' for blind test sets)."""
    df = pd.read_csv(csv_path)
    df["tokens"] = df["tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    if "ner_tags" in df.columns:
        df["ner_tags"] = df["ner_tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    else:
        df["ner_tags"] = df["tokens"].apply(lambda toks: ["O"] * len(toks))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "new_test_data.csv"),
        help="Competition input CSV (columns: ID, tokens[, ner_tags]).",
    )
    parser.add_argument("--output", default=str(SUBMISSION_OUT), help="Submission CSV path.")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    input_df = load_input_df(input_path)
    print(f"Input: {input_path}  ({len(input_df)} rows)")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    dev_dataset = NERDataset(input_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

    print("\nResolving best LR per ensemble member:")
    ckpts = resolve_checkpoints()
    print(f"\n{len(ckpts)} models to ensemble.")

    all_logits = []
    for display, ckpt_path in ckpts:
        print(f"Running {display} ({ckpt_path.name}) ...")
        all_logits.append(run_model(ckpt_path, dev_loader))

    avg_logits = torch.stack(all_logits).mean(dim=0)
    subword_preds = avg_logits.argmax(dim=-1).numpy()

    # Reconstruct word-level BIO tags per sample using first-subword alignment.
    rows = []
    for idx in range(len(dev_dataset)):
        sample = dev_dataset.get_full_sample(idx)
        word_ids = sample["word_ids"]
        raw_tokens = sample["raw_tokens"]

        word_pred = {}
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid in word_pred:
                continue
            word_pred[wid] = ID2LABEL[int(subword_preds[idx][pos])]

        pred_tags = [word_pred.get(j, "O") for j in range(len(raw_tokens))]
        rows.append({
            "ID": sample["id"],
            "predicted_ner_tags": str(pred_tags),
        })

    sub_df = pd.DataFrame(rows, columns=["ID", "predicted_ner_tags"])
    sub_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(sub_df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
