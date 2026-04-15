"""Utilities for the results and ablation notebook."""

from __future__ import annotations

import json
import math
from pathlib import Path

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import pandas as pd
except ImportError:
    pd = None


PUBLISHED_SOTA = 0.61
GPT4O_THREE_SHOT = 0.44

KNOWN_BACKBONES = [
    "socbert",
    "stress_roberta",
    "mental_bert",
    "mental_roberta",
]

CORE_ABLATION_ORDER = [
    ("BiLSTM-CRF", "bilstm_crf"),
    ("DeBERTa Baseline", "deberta_baseline"),
    ("+ Focal Loss", "deberta_focal"),
    ("+ Definition Prompting", "deberta_definition"),
    ("+ Multi-task", "deberta_multitask"),
    ("+ Synthetic + Curriculum", "deberta_synthetic_curriculum"),
    ("Combined w/o Synthetic", "deberta_combined_no_synth"),
    ("Combined", "deberta_combined"),
    ("Hierarchical Pipeline", "hierarchical_deberta"),
    ("Hierarchical Pipeline (0.1 no-impact)", "hierarchical_deberta_0.1_no_impact"),
]

ADVANCED_ORDER = [
    ("Recall Boost (seed 42)", "recall_boost_ow02_s42"),
    ("Recall Boost (seed 123)", "recall_boost_ow02_s123"),
    ("R-Drop (seed 42)", "rdrop_a1_s42"),
    ("R-Drop (seed 123)", "rdrop_a1_s123"),
    ("FGM + SWA (seed 42)", "fgm05_swa_s42"),
]

EXTRA_PIPELINE_ORDER = [
    ("Hierarchical Pipeline", "hierarchical_deberta"),
    ("Hierarchical Pipeline (0.1 no-impact)", "hierarchical_deberta_0.1_no_impact"),
    ("Two-Step Pipeline", "two_step_impact_pipeline"),
    ("Sentence + Token Hierarchy", "sentence_token_hierarchy"),
    ("Span / Nested GLiNER", "span_nested_gliner"),
]

FAMILY_LABELS = {
    "bilstm_crf": "BiLSTM-CRF",
    "deberta_baseline": "DeBERTa Baseline",
    "deberta_focal": "DeBERTa + Focal Loss",
    "deberta_definition": "DeBERTa + Definition Prompting",
    "deberta_multitask": "DeBERTa + Multi-task",
    "deberta_synthetic_curriculum": "DeBERTa + Synthetic + Curriculum",
    "deberta_combined_no_synth": "Combined w/o Synthetic",
    "deberta_combined": "Combined",
    "hierarchical_deberta": "Hierarchical Pipeline",
    "hierarchical_deberta_0.1_no_impact": "Hierarchical Pipeline (0.1 no-impact)",
    "recall_boost_ow02_s42": "Recall Boost (seed 42)",
    "recall_boost_ow02_s123": "Recall Boost (seed 123)",
    "rdrop_a1_s42": "R-Drop (seed 42)",
    "rdrop_a1_s123": "R-Drop (seed 123)",
    "fgm05_swa_s42": "FGM + SWA (seed 42)",
    "two_step_impact_pipeline": "Two-Step Pipeline",
    "sentence_token_hierarchy": "Sentence + Token Hierarchy",
    "span_nested_gliner": "Span / Nested GLiNER",
}

COLOR_MAP = {
    "bilstm_crf": "#9E9E9E",
    "deberta_baseline": "#2196F3",
    "deberta_focal": "#4CAF50",
    "deberta_definition": "#66BB6A",
    "deberta_multitask": "#43A047",
    "deberta_synthetic_curriculum": "#26A69A",
    "deberta_combined_no_synth": "#FFB74D",
    "deberta_combined": "#FB8C00",
    "hierarchical_deberta": "#00897B",
    "hierarchical_deberta_0.1_no_impact": "#00695C",
    "recall_boost_ow02_s42": "#8E24AA",
    "recall_boost_ow02_s123": "#AB47BC",
    "rdrop_a1_s42": "#E53935",
    "rdrop_a1_s123": "#EF5350",
    "fgm05_swa_s42": "#5E35B1",
    "two_step_impact_pipeline": "#3949AB",
    "sentence_token_hierarchy": "#00838F",
    "span_nested_gliner": "#6D4C41",
    "ensemble_size_2": "#1565C0",
    "ensemble_size_3": "#2E7D32",
    "ensemble_size_4": "#EF6C00",
    "ensemble_size_5": "#6A1B9A",
}


def configure_plot_style():
    if plt is not None:
        plt.style.use("seaborn-v0_8-whitegrid")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_lr(value):
    value = safe_float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.0e}" if value < 0.001 else f"{value:g}"


def display_rows(rows, columns=None, sort_by=None, ascending=False, max_rows=None):
    working = list(rows)
    if sort_by is not None:
        working.sort(key=lambda row: row.get(sort_by, float("-inf")), reverse=not ascending)
    if max_rows is not None:
        working = working[:max_rows]

    if pd is not None:
        df = pd.DataFrame(working)
        if columns is not None:
            existing = [column for column in columns if column in df.columns]
            df = df[existing]
        display(df.reset_index(drop=True))
        return df

    for row in working:
        if columns is not None:
            row = {column: row.get(column) for column in columns}
        print(row)
    return working


def score_value(row, metric_key="best_relaxed_f1"):
    return row.get(metric_key, row.get("sweep_best_dev_f1", float("-inf")))


def sort_rows_by_metric(rows, metric_key="best_relaxed_f1"):
    return sorted(rows, key=lambda row: score_value(row, metric_key), reverse=True)


def best_by_group(rows, group_key, metric_key="best_relaxed_f1"):
    grouped = {}
    for row in rows:
        group = row[group_key]
        current = grouped.get(group)
        if current is None or score_value(row, metric_key) > score_value(current, metric_key):
            grouped[group] = row
    return list(grouped.values())


def pick_rows(best_rows, ordered_families):
    by_family = {row["family"]: row for row in best_rows}
    picked = []
    for display_name, family in ordered_families:
        if family in by_family:
            row = dict(by_family[family])
            row["display_name"] = display_name
            picked.append(row)
    return picked


def detect_backbone(name: str) -> str:
    for backbone in sorted(KNOWN_BACKBONES, key=len, reverse=True):
        suffix = f"_{backbone}"
        if name.endswith(suffix):
            return backbone
    return "default"


def strip_backbone_suffix(name: str) -> str:
    backbone = detect_backbone(name)
    if backbone == "default":
        return name
    return name[: -(len(backbone) + 1)]


def family_display_name(family: str) -> str:
    return FAMILY_LABELS.get(family, family.replace("_", " ").title())


def classify_family(base_family: str) -> str:
    if base_family.startswith("matrix_"):
        return "ablation_matrix"
    if base_family in {family for _, family in EXTRA_PIPELINE_ORDER}:
        return "extra_pipeline"
    if base_family in {family for _, family in ADVANCED_ORDER}:
        return "advanced_recipe"
    if base_family in {family for _, family in CORE_ABLATION_ORDER}:
        return "core_ablation"
    return "other"


def logical_scope(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    if not relative.parts:
        return ""
    if relative.parts[0] == "checkpoints":
        return ""
    return relative.parts[0] if len(relative.parts) > 1 else ""


def make_run_key(root: Path, scope: str, experiment_name: str) -> str:
    scope_value = scope or "."
    return f"{root.name}:{scope_value}:{experiment_name}"


def add_metadata(row: dict, root: Path, scope: str, family: str, experiment_name: str):
    base_family = strip_backbone_suffix(family)
    backbone = detect_backbone(family)
    row.update(
        {
            "artifact_root": root.name,
            "artifact_scope": scope or ".",
            "run_key": make_run_key(root, scope, experiment_name),
            "family": family,
            "base_family": base_family,
            "backbone": backbone,
            "family_group": classify_family(base_family),
            "family_display_name": family_display_name(base_family),
        }
    )
    return row


def _has_artifacts(root: Path) -> bool:
    for pattern in ("*_log.json", "*_results.json", "*_lr_sweep.json"):
        if any(root.rglob(pattern)):
            return True
    return (root / "checkpoints").exists()


def discover_artifact_dirs(project_root=Path(".")):
    patterns = [
        "outputs_new_final*",
        "outputs_reddit_backbones*",
        "outputs_ablation_matrix*",
        "outputs_hier*",
    ]
    discovered = []
    seen = set()
    for pattern in patterns:
        for path in sorted(project_root.glob(pattern)):
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved in seen or not _has_artifacts(path):
                continue
            seen.add(resolved)
            discovered.append(path)
    return discovered


def summarize_log(path: Path, root: Path):
    entries = read_json(path)
    if not isinstance(entries, list) or not entries:
        return None, []

    experiment_name = path.stem[:-4] if path.stem.endswith("_log") else path.stem
    family = experiment_name.split("_lr", 1)[0] if "_lr" in experiment_name else experiment_name
    best_f1_entry = max(entries, key=lambda row: safe_float(row.get("relaxed_f1"), float("-inf")))
    valid_dev_loss = [row for row in entries if math.isfinite(safe_float(row.get("dev_loss")))]
    best_loss_entry = min(valid_dev_loss, key=lambda row: safe_float(row.get("dev_loss"))) if valid_dev_loss else None
    scope = logical_scope(root, path)

    summary = {
        "experiment_name": experiment_name,
        "epochs_completed": len(entries),
        "best_relaxed_f1": safe_float(best_f1_entry.get("relaxed_f1")),
        "best_strict_f1": safe_float(best_f1_entry.get("strict_f1")),
        "best_epoch_by_f1": best_f1_entry.get("epoch"),
        "best_dev_loss": safe_float(best_loss_entry.get("dev_loss")) if best_loss_entry else float("nan"),
        "best_epoch_by_loss": best_loss_entry.get("epoch") if best_loss_entry else None,
        "final_relaxed_f1": safe_float(entries[-1].get("relaxed_f1")),
        "final_dev_loss": safe_float(entries[-1].get("dev_loss")),
    }
    add_metadata(summary, root, scope, family, experiment_name)
    return summary, entries


def load_log_artifacts(artifact_dirs):
    summaries = []
    logs_by_run_key = {}
    for root in artifact_dirs:
        for path in sorted(root.rglob("*_log.json")):
            summary, entries = summarize_log(path, root)
            if summary is None:
                continue
            summaries.append(summary)
            logs_by_run_key[summary["run_key"]] = entries
    return summaries, logs_by_run_key


def load_sweep_rows(artifact_dirs):
    rows = []
    for root in artifact_dirs:
        for path in sorted(root.rglob("*_lr_sweep.json")):
            data = read_json(path)
            family = data.get("experiment_name", path.stem.replace("_lr_sweep", ""))
            scope = logical_scope(root, path)
            for run in data.get("runs", []):
                experiment_name = run.get("experiment_name")
                if not experiment_name:
                    continue
                best_relaxed_f1 = safe_float(run.get("best_dev_f1", run.get("relaxed_f1")))
                row = {
                    "experiment_name": experiment_name,
                    "lr": run.get("lr"),
                    "lr_display": format_lr(run.get("lr")),
                    "sweep_best_dev_f1": safe_float(run.get("best_dev_f1")),
                    "best_relaxed_f1": best_relaxed_f1,
                    "best_strict_f1": safe_float(run.get("strict_f1")),
                    "relaxed_precision": safe_float(run.get("relaxed_precision")),
                    "relaxed_recall": safe_float(run.get("relaxed_recall")),
                    "num_logged_epochs": run.get("num_logged_epochs"),
                }
                add_metadata(row, root, scope, family, experiment_name)
                rows.append(row)
    return rows


def load_result_rows(artifact_dirs):
    rows = []
    for root in artifact_dirs:
        for path in sorted(root.rglob("*_results.json")):
            if path.name.endswith("_soup_results.json"):
                continue
            data = read_json(path)
            if not isinstance(data, dict):
                continue
            if "best_overall" in data and "best_by_size" in data:
                continue
            if "relaxed_f1" not in data and "strict_f1" not in data:
                continue

            experiment_name = data.get("experiment_name", path.stem.replace("_results", ""))
            family = experiment_name.split("_lr", 1)[0] if "_lr" in experiment_name else experiment_name
            scope = logical_scope(root, path)
            row = {
                "experiment_name": experiment_name,
                "result_path": str(path),
                "best_relaxed_f1": safe_float(data.get("relaxed_f1")),
                "best_strict_f1": safe_float(data.get("strict_f1")),
                "relaxed_precision": safe_float(data.get("relaxed_precision")),
                "relaxed_recall": safe_float(data.get("relaxed_recall")),
                "strict_precision": safe_float(data.get("strict_precision")),
                "strict_recall": safe_float(data.get("strict_recall")),
            }
            add_metadata(row, root, scope, family, experiment_name)
            rows.append(row)
    return rows


def load_topk_rows(artifact_dirs):
    rows = {}
    for root in artifact_dirs:
        for path in sorted(root.rglob("topk_summary.json")):
            data = read_json(path)
            experiment_name = data.get("experiment_name", path.parent.name)
            family = experiment_name.split("_lr", 1)[0] if "_lr" in experiment_name else experiment_name
            scope = logical_scope(root, path)
            checkpoints = data.get("checkpoints", [])
            row = {
                "topk_saved": len(checkpoints),
                "top_checkpoint_path": checkpoints[0]["path"] if checkpoints else None,
            }
            add_metadata(row, root, scope, family, experiment_name)
            rows[row["run_key"]] = row
    return rows


def load_model_soup_rows(artifact_dirs):
    soup_rows = []
    best_soup_by_model_type = {}
    for root in artifact_dirs:
        for path in sorted(root.rglob("*_soup_results.json")):
            data = read_json(path)
            if not isinstance(data, dict):
                continue
            experiment_name = data.get("experiment_name", path.stem.replace("_results", ""))
            family = experiment_name.replace("_soup", "")
            scope = logical_scope(root, path)
            source_details = data.get("source_experiments", [])
            source_names = [
                detail.get("source_experiment")
                for detail in source_details
                if detail.get("source_experiment")
            ]
            row = {
                "experiment_name": experiment_name,
                "display_name": experiment_name,
                "model_type": data.get("model_type", data.get("metadata", {}).get("model_type", "unknown")),
                "best_relaxed_f1": safe_float(data.get("relaxed_f1")),
                "best_strict_f1": safe_float(data.get("strict_f1")),
                "relaxed_precision": safe_float(data.get("relaxed_precision")),
                "relaxed_recall": safe_float(data.get("relaxed_recall")),
                "num_checkpoints": len(data.get("checkpoint_paths", [])),
                "num_source_experiments": len(source_details),
                "source_experiments": ", ".join(source_names),
                "saved_model_path": data.get("saved_model_path"),
            }
            add_metadata(row, root, scope, family, experiment_name)
            soup_rows.append(row)

            current = best_soup_by_model_type.get(row["model_type"])
            if current is None or row["best_relaxed_f1"] > current["best_relaxed_f1"]:
                best_soup_by_model_type[row["model_type"]] = row

    soup_rows = sort_rows_by_metric(soup_rows, "best_relaxed_f1")
    best_rows = sort_rows_by_metric(best_soup_by_model_type.values(), "best_relaxed_f1")
    return soup_rows, list(best_rows)


def load_ensemble_search_rows(artifact_dirs):
    ensemble_rows = []
    best_by_size_rows = []
    grouped_candidates = {}

    for root in artifact_dirs:
        for path in sorted(root.rglob("*_results.json")):
            data = read_json(path)
            if not isinstance(data, dict):
                continue
            if "best_overall" not in data or "best_by_size" not in data:
                continue

            experiment_name = data.get("experiment_name", path.stem.replace("_results", ""))
            scope = logical_scope(root, path)
            best_overall = data.get("best_overall", {})
            ensemble_rows.append(
                {
                    "artifact_root": root.name,
                    "artifact_scope": scope or ".",
                    "experiment_name": experiment_name,
                    "vote_method": data.get("vote_method", "unknown"),
                    "candidate_count": data.get("candidate_count"),
                    "total_combinations": data.get("search_space", {}).get("total_combinations"),
                    "best_overall_relaxed_f1": safe_float(best_overall.get("relaxed_f1")),
                    "best_overall_strict_f1": safe_float(best_overall.get("strict_f1")),
                    "best_overall_models": ", ".join(best_overall.get("models", [])),
                    "combination_results_dir": data.get("combination_results_dir"),
                }
            )

            for size, record in sorted(data.get("best_by_size", {}).items(), key=lambda item: int(item[0])):
                size_num = int(size)
                best_by_size_rows.append(
                    {
                        "artifact_root": root.name,
                        "artifact_scope": scope or ".",
                        "ensemble_experiment_name": experiment_name,
                        "family": f"ensemble_size_{size_num}",
                        "display_name": f"{size_num} models",
                        "num_models": size_num,
                        "vote_method": data.get("vote_method", "unknown"),
                        "best_relaxed_f1": safe_float(record.get("relaxed_f1")),
                        "best_strict_f1": safe_float(record.get("strict_f1")),
                        "relaxed_precision": safe_float(record.get("relaxed_precision")),
                        "relaxed_recall": safe_float(record.get("relaxed_recall")),
                        "ci_lower": safe_float(record.get("ci_lower")),
                        "ci_upper": safe_float(record.get("ci_upper")),
                        "annotation_label": data.get("vote_method", "unknown"),
                        "models": ", ".join(record.get("models", [])),
                    }
                )

            for record in data.get("all_results", []):
                size_num = int(record.get("num_models", 0))
                vote_method = data.get("vote_method", record.get("vote_method", "unknown"))
                group_key = (root.name, scope or ".", vote_method, size_num)
                grouped_candidates.setdefault(group_key, []).append(
                    {
                        "artifact_root": root.name,
                        "artifact_scope": scope or ".",
                        "ensemble_experiment_name": experiment_name,
                        "family": f"ensemble_size_{size_num}",
                        "num_models": size_num,
                        "vote_method": vote_method,
                        "best_relaxed_f1": safe_float(record.get("relaxed_f1")),
                        "best_strict_f1": safe_float(record.get("strict_f1")),
                        "relaxed_precision": safe_float(record.get("relaxed_precision")),
                        "relaxed_recall": safe_float(record.get("relaxed_recall")),
                        "models": ", ".join(record.get("models", [])),
                    }
                )

    ensemble_rows = sort_rows_by_metric(ensemble_rows, "best_overall_relaxed_f1")
    best_by_size_rows.sort(key=lambda row: (row["ensemble_experiment_name"], row["num_models"]))

    top_two_per_group_rows = []
    for (artifact_root, artifact_scope, vote_method, size_num), rows in sorted(
        grouped_candidates.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]),
    ):
        ranked = sort_rows_by_metric(rows, "best_relaxed_f1")[:2]
        for rank_index, row in enumerate(ranked, start=1):
            top_row = dict(row)
            top_row["artifact_root"] = artifact_root
            top_row["artifact_scope"] = artifact_scope
            top_row["rank_within_group"] = rank_index
            top_row["display_name"] = f"{size_num} models #{rank_index}"
            top_row["annotation_label"] = vote_method
            top_two_per_group_rows.append(top_row)

    return ensemble_rows, best_by_size_rows, top_two_per_group_rows


def merge_run_rows(artifact_dirs):
    log_rows, logs_by_run_key = load_log_artifacts(artifact_dirs)
    sweep_rows = load_sweep_rows(artifact_dirs)
    result_rows = load_result_rows(artifact_dirs)
    topk_rows = load_topk_rows(artifact_dirs)

    logs_by_key = {row["run_key"]: row for row in log_rows}
    results_by_key = {row["run_key"]: row for row in result_rows}

    merged_rows = {}
    for row in sweep_rows:
        merged = dict(row)
        merged.update(logs_by_key.get(row["run_key"], {}))
        merged.update(results_by_key.get(row["run_key"], {}))
        merged.update(topk_rows.get(row["run_key"], {}))
        merged_rows[row["run_key"]] = merged

    for row in result_rows:
        existing = merged_rows.setdefault(row["run_key"], dict(row))
        existing.update(row)
        existing.update(topk_rows.get(row["run_key"], {}))

    merged = sort_rows_by_metric(merged_rows.values(), "best_relaxed_f1")
    return merged, logs_by_run_key, log_rows, sweep_rows, topk_rows, result_rows


def best_by_family(rows, metric_key="best_relaxed_f1", backbone=None):
    working = list(rows)
    if backbone is not None:
        working = [row for row in working if row.get("backbone") == backbone]
    return best_by_group(working, "family", metric_key=metric_key)


def filter_rows(rows, *, backbone=None, family_group=None, artifact_root=None):
    filtered = list(rows)
    if backbone is not None:
        filtered = [row for row in filtered if row.get("backbone") == backbone]
    if family_group is not None:
        filtered = [row for row in filtered if row.get("family_group") == family_group]
    if artifact_root is not None:
        filtered = [row for row in filtered if row.get("artifact_root") == artifact_root]
    return filtered


def plot_barh(rows, title, subtitle=None, metric_key="best_relaxed_f1", add_reference_lines=True):
    if plt is None:
        print("matplotlib is not installed; skipping plot.")
        return
    if not rows:
        print("No rows available for plotting.")
        return

    labels = [row["display_name"] for row in rows]
    scores = [score_value(row, metric_key) for row in rows]
    colors = [COLOR_MAP.get(row.get("base_family", row.get("family")), "#78909C") for row in rows]

    fig_height = max(4, 0.5 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bars = ax.barh(labels, scores, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(metric_key.replace("_", " ").title())
    ax.set_title(title if subtitle is None else f"{title}\n{subtitle}")

    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{score_value(row, metric_key):.3f}",
            va="center",
            fontsize=10,
        )

    if add_reference_lines:
        ax.axvline(PUBLISHED_SOTA, color="#455A64", linestyle="--", linewidth=1.25, label="Published SOTA")
        ax.axvline(GPT4O_THREE_SHOT, color="#90A4AE", linestyle=":", linewidth=1.25, label="GPT-4o 3-shot")
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def plot_learning_curves(best_rows, logs_by_run_key, families, backbone="default", metric_key="relaxed_f1", title="Learning Curves"):
    if plt is None:
        print("matplotlib is not installed; skipping plot.")
        return

    selected_rows = []
    for family in families:
        candidates = [row for row in best_rows if row.get("base_family") == family and row.get("backbone") == backbone]
        if candidates:
            selected_rows.append(sort_rows_by_metric(candidates)[0])

    if not selected_rows:
        print("No matching best-run rows found for learning curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for row in selected_rows:
        entries = logs_by_run_key.get(row["run_key"])
        if not entries:
            continue
        epochs = [entry.get("epoch") for entry in entries]
        dev_loss = [safe_float(entry.get("dev_loss")) for entry in entries]
        tracked_metric = [safe_float(entry.get(metric_key)) for entry in entries]
        label = row.get("family_display_name", row["family"])
        axes[0].plot(epochs, tracked_metric, marker="o", linewidth=2, label=label)
        axes[1].plot(epochs, dev_loss, marker="o", linewidth=2, label=label)

    axes[0].set_title(f"{title}: {metric_key.replace('_', ' ').title()}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(metric_key.replace("_", " ").title())
    axes[0].legend()

    axes[1].set_title(f"{title}: Dev Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dev Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
