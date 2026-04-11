"""Clean CLI for training and evaluating model experiments."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "SMM4H-HeaRD-2026-Task-7-Reddit-Impacts2" / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_GLOVE_PATH = PROJECT_ROOT / "glove.6B.300d.txt"
DEFAULT_DEVICE = "cuda:0"

CORE_EXPERIMENTS = (
    "bilstm",
    "deberta_baseline",
    "deberta_focal",
    "deberta_definition",
    "deberta_multitask",
    "deberta_synthetic_curriculum",
    "deberta_combined",
    "deberta_combined_no_synth",
    "gliner",
    "gliner_finetune",
)

ADVANCED_EXPERIMENTS = (
    "recall_boost_s42",
    "recall_boost_s123",
    "rdrop_s42",
    "rdrop_s123",
    "fgm_swa_s42",
)


@dataclass(frozen=True)
class ExperimentPreset:
    """A named experiment preset plus its default arguments."""

    runner: Callable[..., Any] | str
    description: str
    defaults: dict[str, Any]
    supports_lr: bool = False


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _filter_supported_kwargs(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in signature.parameters
    }


def _resolve_runner(runner: Callable[..., Any] | str) -> Callable[..., Any]:
    if callable(runner):
        return runner

    module_name, attr_name = runner.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _pin_single_cuda_device(run_configs: list[dict[str, Any]]) -> str | None:
    """Constrain the process to one visible CUDA device when requested."""
    requested_devices = {
        str(run_config.get("device"))
        for run_config in run_configs
        if run_config.get("device") is not None
    }
    if len(requested_devices) != 1:
        return None

    requested_device = requested_devices.pop()
    if not requested_device.startswith("cuda:"):
        return None

    index = requested_device.split(":", maxsplit=1)[1]
    if not index.isdigit():
        return None

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = index
    for run_config in run_configs:
        run_config["device"] = "cuda:0"
    return requested_device


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_runtime_paths(data_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> None:
    from src import train as train_module

    train_module.DATA_DIR = str(Path(data_dir).resolve())
    train_module.OUTPUT_DIR = str(Path(output_dir).resolve())


def _resolve_checkpoint_paths(
    checkpoint_names: list[str] | None,
    output_dir: str | os.PathLike[str],
) -> list[Path]:
    selected = checkpoint_names or [
        "recall_boost_ow02_s42_best.pt",
        "recall_boost_ow02_s123_best.pt",
        "rdrop_a1_s123_best.pt",
        "rdrop_a1_s42_best.pt",
        "fgm05_swa_s42_swa.pt",
    ]

    resolved: list[Path] = []
    missing: list[str] = []
    output_dir_path = Path(output_dir).resolve()

    for checkpoint in selected:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = output_dir_path / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if checkpoint_path.exists():
            resolved.append(checkpoint_path)
        else:
            missing.append(str(checkpoint_path))

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing ensemble checkpoint(s):\n"
            f"{formatted}\n"
            "Run the prerequisite experiments first or pass --checkpoint to override the list."
        )

    return resolved


def _format_lr_suffix(lr: float) -> str:
    """Create a filesystem-friendly suffix for a learning rate."""

    return format(lr, ".12g").replace(".", "p").replace("-", "m").replace("+", "")


def _summarize_result(result: Any) -> dict[str, Any]:
    """Extract a compact summary from a training or evaluation return value."""

    if isinstance(result, tuple):
        summary = {"best_dev_f1": result[0]}
        if len(result) > 1 and isinstance(result[1], list):
            summary["num_logged_epochs"] = len(result[1])
        return summary

    if isinstance(result, dict):
        summary: dict[str, Any] = {}
        for key in ("relaxed_f1", "strict_f1", "relaxed_precision", "relaxed_recall", "ci_lower", "ci_upper"):
            if key in result:
                summary[key] = result[key]
        return summary or {"result": result}

    return {"result": result}


def _write_sweep_summary(output_dir: str | os.PathLike[str], experiment_name: str, runs: list[dict[str, Any]]) -> Path:
    """Persist a summary JSON for a multi-LR sweep."""

    output_path = Path(output_dir).resolve() / f"{experiment_name}_lr_sweep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"experiment_name": experiment_name, "runs": runs}, handle, indent=2)
    return output_path


def run_ensemble(
    model_name: str = "microsoft/deberta-v3-large",
    batch_size: int = 30,
    device: str = DEFAULT_DEVICE,
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    checkpoint_names: list[str] | None = None,
    bootstrap_samples: int = 2000,
    experiment_name: str = "ensemble",
) -> dict[str, Any]:
    """Evaluate the 5-model ensemble and write a result JSON."""

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from src.data import ID2LABEL, NERDataset, NUM_LABELS, load_dataframe
    from src.deberta_ner import DeBERTaNERMultiTask
    from src.evaluation import bootstrap_ci
    from src.train import ensemble_evaluate

    checkpoint_paths = _resolve_checkpoint_paths(checkpoint_names, output_dir)
    print("\n" + "=" * 60)
    print(f"  Experiment: {experiment_name}")
    print(f"  Device: {device}")
    print(f"  Checkpoints: {len(checkpoint_paths)}")
    print("=" * 60 + "\n")

    dev_df = load_dataframe(Path(data_dir) / "new_dev_data.csv")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dev_dataset = NERDataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    results = ensemble_evaluate(
        [str(path) for path in checkpoint_paths],
        DeBERTaNERMultiTask,
        {"model_name": model_name, "num_labels": NUM_LABELS},
        dev_dataset,
        dev_loader,
        device,
    )

    all_logits = []
    for checkpoint_path in checkpoint_paths:
        model = DeBERTaNERMultiTask(model_name=model_name, num_labels=NUM_LABELS).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        batch_logits = []
        with torch.no_grad():
            for batch in dev_loader:
                batch_device = {key: value.to(device) for key, value in batch.items()}
                outputs = model(
                    input_ids=batch_device["input_ids"],
                    attention_mask=batch_device["attention_mask"],
                )
                batch_logits.append(outputs["logits"].cpu())

        all_logits.append(torch.cat(batch_logits, dim=0))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = avg_logits.argmax(dim=-1).numpy()

    all_gold = []
    all_pred = []
    for index in range(min(preds.shape[0], len(dev_dataset.samples))):
        sample = dev_dataset.get_full_sample(index)
        word_preds = {}
        for token_index, word_id in enumerate(sample["word_ids"]):
            if word_id is not None and word_id not in word_preds:
                word_preds[word_id] = ID2LABEL[preds[index][token_index]]

        pred_tags = [word_preds.get(word_index, "O") for word_index in range(len(sample["raw_tags"]))]
        all_gold.append(sample["raw_tags"])
        all_pred.append(pred_tags)

    ci = bootstrap_ci(all_gold, all_pred, n_bootstrap=bootstrap_samples)
    results["ci_lower"] = ci["ci_lower"]
    results["ci_upper"] = ci["ci_upper"]
    results["checkpoints"] = [str(path) for path in checkpoint_paths]

    output_path = Path(output_dir) / f"{experiment_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(
        f"\nEnsemble Relaxed F1: {results['relaxed_f1']:.4f} "
        f"(95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}])"
    )
    return results


def build_presets() -> "OrderedDict[str, ExperimentPreset]":
    """Return the clean experiment registry."""

    return OrderedDict(
        {
            "bilstm": ExperimentPreset(
                runner="src.train:train_bilstm_crf",
                description="BiLSTM-CRF baseline.",
                defaults={
                    "glove_path": str(DEFAULT_GLOVE_PATH),
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 1e-3,
                    "device": DEFAULT_DEVICE,
                    "experiment_name": "bilstm_crf",
                },
                supports_lr=True,
            ),
            "deberta_baseline": ExperimentPreset(
                runner="src.train:train_deberta",
                description="DeBERTa-large baseline.",
                defaults={
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 2e-5,
                    "device": DEFAULT_DEVICE,
                    "experiment_name": "deberta_baseline",
                },
                supports_lr=True,
            ),
            "deberta_focal": ExperimentPreset(
                runner="src.train:train_deberta",
                description="DeBERTa with focal loss.",
                defaults={
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 2e-5,
                    "device": DEFAULT_DEVICE,
                    "use_focal_loss": True,
                    "experiment_name": "deberta_focal",
                },
                supports_lr=True,
            ),
            "deberta_definition": ExperimentPreset(
                runner="src.train:train_deberta",
                description="DeBERTa with entity definition prompting.",
                defaults={
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 2e-5,
                    "device": DEFAULT_DEVICE,
                    "definition_prompting": True,
                    "experiment_name": "deberta_definition",
                },
                supports_lr=True,
            ),
            "deberta_multitask": ExperimentPreset(
                runner="src.train:train_deberta",
                description="DeBERTa with auxiliary entity-presence learning.",
                defaults={
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 2e-5,
                    "device": DEFAULT_DEVICE,
                    "use_multitask": True,
                    "experiment_name": "deberta_multitask",
                },
                supports_lr=True,
            ),
            "deberta_synthetic_curriculum": ExperimentPreset(
                runner="src.train:train_deberta",
                description="DeBERTa with synthetic data and curriculum learning.",
                defaults={
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 2e-5,
                    "device": DEFAULT_DEVICE,
                    "use_synthetic": True,
                    "use_curriculum": True,
                    "experiment_name": "deberta_synthetic_curriculum",
                },
                supports_lr=True,
            ),
            "deberta_combined": ExperimentPreset(
                runner="src.train:train_deberta",
                description="Combined DeBERTa configuration.",
                defaults={
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 2e-5,
                    "device": DEFAULT_DEVICE,
                    "use_focal_loss": True,
                    "definition_prompting": True,
                    "use_multitask": True,
                    "use_synthetic": True,
                    "use_curriculum": True,
                    "experiment_name": "deberta_combined",
                },
                supports_lr=True,
            ),
            "deberta_combined_no_synth": ExperimentPreset(
                runner="src.train:train_deberta",
                description="Combined DeBERTa ablation without synthetic data.",
                defaults={
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 2e-5,
                    "device": DEFAULT_DEVICE,
                    "use_focal_loss": True,
                    "definition_prompting": True,
                    "use_multitask": True,
                    "use_synthetic": False,
                    "use_curriculum": True,
                    "experiment_name": "deberta_combined_no_synth",
                },
                supports_lr=True,
            ),
            "gliner": ExperimentPreset(
                runner="src.train:run_gliner_experiment",
                description="GLiNER zero-shot evaluation.",
                defaults={
                    "device": DEFAULT_DEVICE,
                    "experiment_name": "gliner",
                },
            ),
            "gliner_finetune": ExperimentPreset(
                runner="src.gliner_finetune:finetune_gliner",
                description="GLiNER fine-tuning on the task data.",
                defaults={
                    "epochs": 5,
                    "batch_size": 8,
                    "lr": 1e-5,
                    "threshold": 0.4,
                    "device": DEFAULT_DEVICE,
                    "experiment_name": "gliner_finetune",
                },
                supports_lr=True,
            ),
            "gliner_inference": ExperimentPreset(
                runner="src.gliner_finetune:run_gliner_inference",
                description="Evaluate a fine-tuned GLiNER checkpoint.",
                defaults={
                    "threshold": 0.4,
                    "device": DEFAULT_DEVICE,
                    "experiment_name": "gliner_inference",
                },
            ),
            "hierarchical_deberta": ExperimentPreset(
                runner="src.hierarchical:run_hierarchical_deberta",
                description="Sentence classifier first, then NER only on predicted impact posts.",
                defaults={
                    "epochs": 10,
                    "batch_size": 16,
                    "lr": 2e-5,
                    "threshold": 0.5,
                    "ner_no_impact_keep_ratio": 0.1,
                    "device": DEFAULT_DEVICE,
                    "experiment_name": "hierarchical_deberta",
                },
                supports_lr=True,
            ),
            "model_soup": ExperimentPreset(
                runner="src.model_soup:run_model_soup",
                description="Average compatible checkpoints in weight space and evaluate the soup.",
                defaults={
                    "batch_size": 8,
                    "device": DEFAULT_DEVICE,
                    "data_dir": str(DEFAULT_DATA_DIR),
                    "output_dir": str(DEFAULT_OUTPUT_DIR),
                    "checkpoint_limit": 5,
                    "experiment_name": "model_soup",
                },
            ),
            "ensemble_search": ExperimentPreset(
                runner="src.ensemble_search:run_ensemble_search",
                description="Search 2..N probability-averaged ensemble combinations over saved checkpoints.",
                defaults={
                    "batch_size": 8,
                    "device": DEFAULT_DEVICE,
                    "data_dir": str(DEFAULT_DATA_DIR),
                    "output_dir": str(DEFAULT_OUTPUT_DIR),
                    "checkpoint_limit": 1,
                    "min_models": 2,
                    "max_models": 5,
                    "bootstrap_samples": 2000,
                    "experiment_name": "ensemble_search",
                },
            ),
            "recall_boost_s42": ExperimentPreset(
                runner="src.train:train_deberta_recall_boost",
                description="Recall-boost training with seed 42.",
                defaults={
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 1.5e-5,
                    "gradient_accumulation_steps": 1,
                    "o_weight": 0.2,
                    "device": DEFAULT_DEVICE,
                    "seed": 42,
                    "experiment_name": "recall_boost_ow02_s42",
                },
                supports_lr=True,
            ),
            "recall_boost_s123": ExperimentPreset(
                runner="src.train:train_deberta_recall_boost",
                description="Recall-boost training with seed 123.",
                defaults={
                    "epochs": 30,
                    "batch_size": 30,
                    "lr": 1.5e-5,
                    "gradient_accumulation_steps": 1,
                    "o_weight": 0.2,
                    "device": DEFAULT_DEVICE,
                    "seed": 123,
                    "experiment_name": "recall_boost_ow02_s123",
                },
                supports_lr=True,
            ),
            "rdrop_s42": ExperimentPreset(
                runner="src.train:train_deberta_rdrop",
                description="R-Drop training with seed 42.",
                defaults={
                    "epochs": 90,
                    "batch_size": 8,
                    "lr": 1.5e-5,
                    "gradient_accumulation_steps": 4,
                    "o_weight": 0.2,
                    "rdrop_alpha": 1.0,
                    "device": DEFAULT_DEVICE,
                    "seed": 42,
                    "experiment_name": "rdrop_a1_s42",
                },
                supports_lr=True,
            ),
            "rdrop_s123": ExperimentPreset(
                runner="src.train:train_deberta_rdrop",
                description="R-Drop training with seed 123.",
                defaults={
                    "epochs": 90,
                    "batch_size": 8,
                    "lr": 1.5e-5,
                    "gradient_accumulation_steps": 4,
                    "o_weight": 0.2,
                    "rdrop_alpha": 1.0,
                    "device": DEFAULT_DEVICE,
                    "seed": 123,
                    "experiment_name": "rdrop_a1_s123",
                },
                supports_lr=True,
            ),
            "fgm_swa_s42": ExperimentPreset(
                runner="src.train:train_deberta_fgm_swa",
                description="FGM + SWA training with seed 42.",
                defaults={
                    "epochs": 120,
                    "batch_size": 8,
                    "lr": 1.5e-5,
                    "gradient_accumulation_steps": 4,
                    "o_weight": 0.2,
                    "fgm_epsilon": 0.5,
                    "swa_start_epoch": 10,
                    "device": DEFAULT_DEVICE,
                    "seed": 42,
                    "experiment_name": "fgm05_swa_s42",
                },
                supports_lr=True,
            ),
            "ensemble": ExperimentPreset(
                runner=run_ensemble,
                description="5-model ensemble evaluation.",
                defaults={
                    "batch_size": 30,
                    "device": DEFAULT_DEVICE,
                    "data_dir": str(DEFAULT_DATA_DIR),
                    "output_dir": str(DEFAULT_OUTPUT_DIR),
                    "bootstrap_samples": 2000,
                    "experiment_name": "ensemble",
                },
            ),
        }
    )


def _maybe_adjust_seeded_name(
    preset: ExperimentPreset,
    config: dict[str, Any],
    explicit_experiment_name: str | None,
    explicit_seed: int | None,
) -> None:
    if explicit_experiment_name is not None or explicit_seed is None:
        return

    default_seed = preset.defaults.get("seed")
    experiment_name = config.get("experiment_name")
    if default_seed is None or experiment_name is None:
        return

    config["experiment_name"] = str(experiment_name).replace(f"s{default_seed}", f"s{explicit_seed}")


def list_command(_: argparse.Namespace) -> int:
    """Print available experiments and groups."""

    presets = build_presets()
    print("Available experiments:\n")
    for name, preset in presets.items():
        print(f"  {name:<28} {preset.description}")

    print("\nGroups:\n")
    print(f"  core      {' '.join(CORE_EXPERIMENTS)}")
    print(f"  advanced  {' '.join(ADVANCED_EXPERIMENTS)}")
    print(f"  all       {' '.join(CORE_EXPERIMENTS + ADVANCED_EXPERIMENTS)}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    """Resolve one preset, apply overrides, and execute it."""

    presets = build_presets()
    preset = presets[args.experiment]

    config = dict(preset.defaults)
    config["data_dir"] = str(Path(args.data_dir).resolve())
    config["output_dir"] = str(Path(args.output_dir).resolve())
    lr_values = args.lr or []
    if lr_values and not preset.supports_lr:
        raise SystemExit(f"Experiment '{args.experiment}' does not accept --lr overrides.")

    overrides = {
        "model_name": args.model_name,
        "glove_path": args.glove_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "experiment_name": args.experiment_name,
        "threshold": args.threshold,
        "model_dir": args.model_dir,
        "model_output_dir": args.model_output_dir,
        "model_type": args.model_type,
        "checkpoint_limit": args.checkpoint_limit,
        "top_k_checkpoints": args.top_k_checkpoints,
        "min_models": args.min_models,
        "max_models": args.max_models,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "o_weight": args.o_weight,
        "rdrop_alpha": args.rdrop_alpha,
        "fgm_epsilon": args.fgm_epsilon,
        "swa_start_epoch": args.swa_start_epoch,
        "bootstrap_samples": args.bootstrap_samples,
        "ner_no_impact_keep_ratio": args.ner_no_impact_keep_ratio,
        "vote_method": args.vote_method,
        "save_combination_files": args.save_combination_files,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if args.checkpoint:
        config["checkpoint_names"] = args.checkpoint
    if args.source_experiment:
        config["source_experiments"] = args.source_experiment

    _maybe_adjust_seeded_name(preset, config, args.experiment_name, args.seed)
    run_configs: list[dict[str, Any]]
    if lr_values:
        run_configs = []
        for lr in lr_values:
            run_config = dict(config)
            run_config["lr"] = lr
            if len(lr_values) > 1:
                run_config["experiment_name"] = f"{config['experiment_name']}_lr{_format_lr_suffix(lr)}"
            run_configs.append(run_config)
    else:
        run_configs = [config]

    pinned_device = _pin_single_cuda_device(run_configs)
    printable_runs = [{key: _json_ready(value) for key, value in run_config.items()} for run_config in run_configs]
    print(json.dumps({"experiment": args.experiment, "runs": printable_runs}, indent=2))
    if pinned_device is not None:
        print(
            f"CUDA visibility pinned to requested device {pinned_device} "
            "(exposed to the process as cuda:0)."
        )

    if args.dry_run:
        return 0

    runner = _resolve_runner(preset.runner)
    sweep_results: list[dict[str, Any]] = []
    for index, run_config in enumerate(run_configs, start=1):
        print("\n" + "-" * 60)
        print(f"Run {index}/{len(run_configs)}: {run_config['experiment_name']}")
        if "lr" in run_config:
            print(f"Learning rate: {run_config['lr']}")
        print("-" * 60)

        _apply_runtime_paths(run_config["data_dir"], run_config["output_dir"])
        if args.seed is not None:
            _set_seed(args.seed)
        elif "seed" in run_config:
            _set_seed(int(run_config["seed"]))

        call_kwargs = _filter_supported_kwargs(runner, run_config)
        result = runner(**call_kwargs)
        summary = _summarize_result(result)
        summary["experiment_name"] = run_config["experiment_name"]
        if "lr" in run_config:
            summary["lr"] = run_config["lr"]
        sweep_results.append(summary)

    if len(sweep_results) > 1:
        summary_path = _write_sweep_summary(config["output_dir"], config["experiment_name"], sweep_results)
        print(f"\nSaved LR sweep summary to {summary_path}")

        ranked = sorted(
            sweep_results,
            key=lambda item: item.get("best_dev_f1", item.get("relaxed_f1", float("-inf"))),
            reverse=True,
        )
        print("LR sweep results:")
        for item in ranked:
            score = item.get("best_dev_f1", item.get("relaxed_f1"))
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            print(f"  {item['experiment_name']}: lr={item['lr']} score={score_text}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the project experiments as standalone Python jobs. "
            "Each job can be launched in a separate process to keep GPU memory usage predictable."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="Show available experiments and groups.")
    list_parser.set_defaults(handler=list_command)

    run_parser = subparsers.add_parser("run", help="Run one experiment preset.")
    run_parser.add_argument("--experiment", required=True, choices=list(build_presets().keys()))
    run_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    run_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run_parser.add_argument("--model-name")
    run_parser.add_argument("--glove-path")
    run_parser.add_argument("--epochs", type=int)
    run_parser.add_argument("--batch-size", type=int)
    run_parser.add_argument(
        "--lr",
        type=float,
        action="append",
        help="Learning rate override. Repeat the flag to run a sweep across multiple learning rates.",
    )
    run_parser.add_argument("--device")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--experiment-name")
    run_parser.add_argument("--threshold", type=float)
    run_parser.add_argument("--model-dir")
    run_parser.add_argument("--model-output-dir")
    run_parser.add_argument("--model-type")
    run_parser.add_argument("--checkpoint-limit", type=int)
    run_parser.add_argument("--top-k-checkpoints", type=int)
    run_parser.add_argument("--min-models", type=int)
    run_parser.add_argument("--max-models", type=int)
    run_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Stop training after this many epochs without validation-loss improvement above --early-stopping-min-delta. Use 0 to disable.",
    )
    run_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        help="Minimum validation-loss decrease needed to reset early-stopping patience.",
    )
    run_parser.add_argument(
        "--source-experiment",
        action="append",
        help="Source experiment name for model_soup or ensemble_search. Repeat to combine multiple runs.",
    )
    run_parser.add_argument("--gradient-accumulation-steps", type=int)
    run_parser.add_argument("--o-weight", type=float)
    run_parser.add_argument("--rdrop-alpha", type=float)
    run_parser.add_argument("--fgm-epsilon", type=float)
    run_parser.add_argument("--swa-start-epoch", type=int)
    run_parser.add_argument("--bootstrap-samples", type=int)
    run_parser.add_argument(
        "--vote-method",
        choices=("probability_average", "majority_vote"),
        help="How to combine model outputs during ensemble_search.",
    )
    run_parser.add_argument(
        "--save-combination-files",
        action="store_true",
        help="For ensemble_search, write one JSON file per evaluated model combination.",
    )
    run_parser.add_argument(
        "--ner-no-impact-keep-ratio",
        type=float,
        help="For hierarchical_deberta, keep this fraction of all-O training rows when training the NER stage.",
    )
    run_parser.add_argument(
        "--checkpoint",
        action="append",
        help="Checkpoint path for ensemble, model_soup, or ensemble_search. Repeat to provide multiple files.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without launching training or evaluation.",
    )
    run_parser.set_defaults(handler=run_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
