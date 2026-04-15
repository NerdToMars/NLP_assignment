"""Utilities for the results and ablation notebook."""

from __future__ import annotations

import json
import math
from pathlib import Path

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = None
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

CONTRIBUTION_GROUPS = [
    {
        "group": "Core Model Family",
        "kind": "run",
        "families": [family for _, family in CORE_ABLATION_ORDER],
        "description": "Baseline encoder and its main ablation variants.",
    },
    {
        "group": "Advanced Training Recipes",
        "kind": "run",
        "families": [family for _, family in ADVANCED_ORDER],
        "description": "Recall-oriented and regularized training recipes.",
    },
    {
        "group": "Pipeline Additions",
        "kind": "run",
        "families": [family for _, family in EXTRA_PIPELINE_ORDER],
        "description": "Hierarchical, two-step, and span-oriented alternatives.",
    },
    {
        "group": "Domain Backbones",
        "kind": "backbone",
        "description": "Reddit / mental-health backbone swaps evaluated with the same recipes.",
    },
    {
        "group": "Model Soup",
        "kind": "soup",
        "description": "Weight-averaged checkpoints from compatible runs.",
    },
    {
        "group": "Ensemble Search",
        "kind": "ensemble",
        "description": "Multi-model voting or probability-averaging combinations.",
    },
]

CONTRIBUTION_EXPLANATION_SPECS = [
    {
        "contribution": "BiLSTM-CRF baseline",
        "kind": "run",
        "families": ["bilstm_crf"],
        "rationale": "Classical sequence-labeling baseline used to measure how much the transformer encoders help on noisy Reddit NER.",
    },
    {
        "contribution": "DeBERTa baseline",
        "kind": "run",
        "families": ["deberta_baseline"],
        "rationale": "Reference transformer token classifier that anchors all of the later comparisons.",
    },
    {
        "contribution": "Focal loss",
        "kind": "run",
        "families": ["deberta_focal"],
        "rationale": "Tests whether reweighting hard examples helps with the strong O-class imbalance.",
    },
    {
        "contribution": "Definition prompting",
        "kind": "run",
        "families": ["deberta_definition"],
        "rationale": "Injects lightweight task definitions so the encoder sees more explicit semantics for Clinical vs Social impacts.",
    },
    {
        "contribution": "Multi-task objective",
        "kind": "run",
        "families": ["deberta_multitask"],
        "rationale": "Adds auxiliary supervision so the model learns impact presence/context as well as token tags.",
    },
    {
        "contribution": "Synthetic + curriculum data",
        "kind": "run",
        "families": ["deberta_synthetic_curriculum"],
        "rationale": "Expands training signal and orders examples to test whether staged exposure improves span learning.",
    },
    {
        "contribution": "Combined recipes",
        "kind": "run",
        "families": ["deberta_combined_no_synth", "deberta_combined"],
        "rationale": "Measures whether stacking the stronger ideas compounds or causes interference.",
    },
    {
        "contribution": "Recall-boost / regularized recipes",
        "kind": "run",
        "families": ["recall_boost_ow02_s42", "recall_boost_ow02_s123", "rdrop_a1_s42", "rdrop_a1_s123", "fgm05_swa_s42"],
        "rationale": "Explores whether recall-oriented losses and regularization stabilize extraction on sparse entity spans.",
    },
    {
        "contribution": "Hierarchical and staged pipelines",
        "kind": "run",
        "families": ["hierarchical_deberta", "hierarchical_deberta_0.1_no_impact", "two_step_impact_pipeline", "sentence_token_hierarchy"],
        "rationale": "Tests whether sentence-level gating or multi-stage extraction reduces false positives before token decoding.",
    },
    {
        "contribution": "GLiNER span models",
        "kind": "run",
        "families": ["gliner", "gliner_finetune", "span_nested_gliner"],
        "rationale": "Checks whether span-based extraction helps with overlap-like cases and stricter boundary decisions.",
    },
    {
        "contribution": "Reddit-domain backbones",
        "kind": "backbone",
        "rationale": "Swaps in domain-adapted encoders to see whether Reddit-specific language modeling transfers to this task.",
    },
    {
        "contribution": "Model soup",
        "kind": "soup",
        "rationale": "Averages compatible checkpoints to smooth optimization noise without requiring multi-model inference.",
    },
    {
        "contribution": "Ensemble search",
        "kind": "ensemble",
        "rationale": "Combines complementary models to capture different precision/recall trade-offs at inference time.",
    },
]


def configure_plot_style():
    if plt is not None:
        plt.style.use("seaborn-v0_8-whitegrid")


def display_markdown(text: str):
    if Markdown is not None:
        display(Markdown(text))
    else:
        print(text)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_finite_score(value) -> bool:
    return math.isfinite(safe_float(value))


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


def metric_label(metric_key: str) -> str:
    return metric_key.replace("_", " ").title()


def format_metric(value, digits=3, nan="n/a"):
    value = safe_float(value)
    if not math.isfinite(value):
        return nan
    return f"{value:.{digits}f}"


def format_delta(value, digits=3, nan="n/a"):
    value = safe_float(value)
    if not math.isfinite(value):
        return nan
    return f"{value:+.{digits}f}"


def score_value(row, metric_key="best_relaxed_f1"):
    return row.get(metric_key, row.get("sweep_best_dev_f1", float("-inf")))


def sort_rows_by_metric(rows, metric_key="best_relaxed_f1"):
    return sorted(rows, key=lambda row: score_value(row, metric_key), reverse=True)


def top_row(rows, metric_key="best_relaxed_f1"):
    ranked = sort_rows_by_metric(rows, metric_key)
    return ranked[0] if ranked else None


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
        "outputs",
        "outputs_*",
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
            all_results = data.get("all_results", [])
            best_overall_relaxed = (
                data.get("best_overall_by_relaxed")
                or data.get("best_overall")
                or top_row(all_results, "relaxed_f1")
                or {}
            )
            best_overall_strict = (
                data.get("best_overall_by_strict")
                or top_row(all_results, "strict_f1")
                or best_overall_relaxed
                or {}
            )
            ensemble_rows.append(
                {
                    "artifact_root": root.name,
                    "artifact_scope": scope or ".",
                    "experiment_name": experiment_name,
                    "vote_method": data.get("vote_method", "unknown"),
                    "candidate_count": data.get("candidate_count"),
                    "total_combinations": data.get("search_space", {}).get("total_combinations"),
                    "best_overall_relaxed_f1": safe_float(best_overall_relaxed.get("relaxed_f1")),
                    "best_overall_strict_f1": safe_float(best_overall_strict.get("strict_f1")),
                    "best_overall_by_relaxed_f1": safe_float(best_overall_relaxed.get("relaxed_f1")),
                    "best_overall_by_relaxed_models": ", ".join(best_overall_relaxed.get("models", [])),
                    "best_overall_by_strict_f1": safe_float(best_overall_strict.get("strict_f1")),
                    "best_overall_by_strict_models": ", ".join(best_overall_strict.get("models", [])),
                    "best_overall_models": ", ".join(best_overall_relaxed.get("models", [])),
                    "combination_results_dir": data.get("combination_results_dir"),
                }
            )

            best_by_size_relaxed = data.get("best_by_size_by_relaxed", data.get("best_by_size", {}))
            best_by_size_strict = data.get("best_by_size_by_strict", {})
            for size, record in sorted(best_by_size_relaxed.items(), key=lambda item: int(item[0])):
                size_num = int(size)
                strict_record = best_by_size_strict.get(str(size), record)
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
                        "best_strict_f1": safe_float(strict_record.get("strict_f1", record.get("strict_f1"))),
                        "relaxed_precision": safe_float(record.get("relaxed_precision")),
                        "relaxed_recall": safe_float(record.get("relaxed_recall")),
                        "ci_lower": safe_float(record.get("ci_lower")),
                        "ci_upper": safe_float(record.get("ci_upper")),
                        "annotation_label": data.get("vote_method", "unknown"),
                        "models": ", ".join(record.get("models", [])),
                        "strict_models": ", ".join(strict_record.get("models", record.get("models", []))),
                    }
                )

            for record in all_results:
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
            ranked_row = dict(row)
            ranked_row["artifact_root"] = artifact_root
            ranked_row["artifact_scope"] = artifact_scope
            ranked_row["rank_within_group"] = rank_index
            ranked_row["display_name"] = f"{size_num} models #{rank_index}"
            ranked_row["annotation_label"] = vote_method
            top_two_per_group_rows.append(ranked_row)

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


def describe_row(row, metric_key="best_relaxed_f1"):
    if not row:
        return "n/a"
    family = row.get("family_display_name") or family_display_name(row.get("base_family", row.get("family", "run")))
    backbone = row.get("backbone")
    label = family if backbone in (None, "default") else f"{family} [{backbone}]"
    experiment_name = row.get("experiment_name")
    score = row.get(metric_key)
    if experiment_name:
        return f"{label} (`{experiment_name}`; {metric_label(metric_key)} {format_metric(score)})"
    return f"{label} ({metric_label(metric_key)} {format_metric(score)})"


def add_gain_columns(rows, reference_row, relaxed_key="best_relaxed_f1", strict_key="best_strict_f1"):
    reference_relaxed = safe_float(reference_row.get(relaxed_key)) if reference_row else float("nan")
    reference_strict = safe_float(reference_row.get(strict_key)) if reference_row else float("nan")
    enriched = []
    for row in rows:
        enriched_row = dict(row)
        enriched_row["delta_relaxed_vs_reference"] = safe_float(row.get(relaxed_key)) - reference_relaxed
        enriched_row["delta_strict_vs_reference"] = safe_float(row.get(strict_key)) - reference_strict
        enriched.append(enriched_row)
    return enriched


def summarize_artifact_coverage(run_rows, soup_rows, ensemble_rows):
    coverage_rows = []
    roots = sorted(
        {
            row.get("artifact_root")
            for row in list(run_rows) + list(soup_rows) + list(ensemble_rows)
            if row.get("artifact_root")
        }
    )
    for artifact_root in roots:
        root_run_rows = [row for row in run_rows if row.get("artifact_root") == artifact_root]
        root_soup_rows = [row for row in soup_rows if row.get("artifact_root") == artifact_root]
        root_ensemble_rows = [row for row in ensemble_rows if row.get("artifact_root") == artifact_root]
        coverage_rows.append(
            {
                "artifact_root": artifact_root,
                "num_runs": len(root_run_rows),
                "num_families": len({row.get("family") for row in root_run_rows}),
                "num_backbones": len({row.get("backbone") for row in root_run_rows}),
                "num_soups": len(root_soup_rows),
                "num_ensemble_searches": len(root_ensemble_rows),
                "best_relaxed_f1": score_value(top_row(root_run_rows, "best_relaxed_f1") or {}, "best_relaxed_f1"),
                "best_strict_f1": score_value(top_row(root_run_rows, "best_strict_f1") or {}, "best_strict_f1"),
            }
        )
    return coverage_rows


def build_contribution_rows(run_rows, soup_rows, ensemble_rows, best_rows=None):
    best_rows = list(best_rows) if best_rows is not None else best_by_family(run_rows)
    baseline_row = top_row(
        [
            row for row in best_rows
            if row.get("base_family") == "deberta_baseline" and row.get("backbone") == "default"
        ],
        "best_relaxed_f1",
    )
    contribution_rows = []
    for group in CONTRIBUTION_GROUPS:
        group_name = group["group"]
        description = group["description"]
        kind = group["kind"]

        if kind == "run":
            candidates = [row for row in best_rows if row.get("base_family") in group["families"]]
            included_items = ", ".join(
                family_display_name(family)
                for family in group["families"]
                if any(row.get("base_family") == family for row in best_rows)
            )
        elif kind == "backbone":
            candidates = [row for row in best_rows if row.get("backbone") not in (None, "default")]
            included_items = ", ".join(sorted({row.get("backbone") for row in candidates if row.get("backbone")}))
        elif kind == "soup":
            candidates = list(soup_rows)
            included_items = ", ".join(sorted({row.get("model_type", "unknown") for row in candidates}))
        elif kind == "ensemble":
            candidates = []
            for row in ensemble_rows:
                candidate = {
                    "artifact_root": row.get("artifact_root"),
                    "artifact_scope": row.get("artifact_scope"),
                    "experiment_name": row.get("experiment_name"),
                    "family": "ensemble_search",
                    "base_family": "ensemble_search",
                    "family_display_name": "Ensemble Search",
                    "best_relaxed_f1": safe_float(
                        row.get("best_overall_by_relaxed_f1", row.get("best_overall_relaxed_f1"))
                    ),
                    "best_strict_f1": safe_float(
                        row.get("best_overall_by_strict_f1", row.get("best_overall_strict_f1"))
                    ),
                    "summary": row.get("best_overall_by_relaxed_models", row.get("best_overall_models")),
                }
                candidates.append(candidate)
            included_items = ", ".join(sorted({row.get("vote_method", "unknown") for row in ensemble_rows}))
        else:
            candidates = []
            included_items = ""

        best_relaxed = top_row(candidates, "best_relaxed_f1")
        best_strict = top_row(candidates, "best_strict_f1")
        if best_relaxed is None and best_strict is None:
            continue

        contribution_rows.append(
            {
                "group": group_name,
                "description": description,
                "included_items": included_items,
                "best_relaxed_experiment": best_relaxed.get("experiment_name") if best_relaxed else None,
                "best_relaxed_display": describe_row(best_relaxed, "best_relaxed_f1") if best_relaxed else "n/a",
                "best_relaxed_f1": safe_float(best_relaxed.get("best_relaxed_f1")) if best_relaxed else float("nan"),
                "delta_relaxed_vs_baseline": (
                    safe_float(best_relaxed.get("best_relaxed_f1")) - safe_float(baseline_row.get("best_relaxed_f1"))
                    if best_relaxed and baseline_row
                    else float("nan")
                ),
                "best_strict_experiment": best_strict.get("experiment_name") if best_strict else None,
                "best_strict_display": describe_row(best_strict, "best_strict_f1") if best_strict else "n/a",
                "best_strict_f1": safe_float(best_strict.get("best_strict_f1")) if best_strict else float("nan"),
                "delta_strict_vs_baseline": (
                    safe_float(best_strict.get("best_strict_f1")) - safe_float(baseline_row.get("best_strict_f1"))
                    if best_strict and baseline_row
                    else float("nan")
                ),
            }
        )
    return contribution_rows


def build_experiment_gain_rows(best_rows, ordered_families, baseline_family="deberta_baseline", backbone="default"):
    backbone_rows = [row for row in best_rows if row.get("backbone") == backbone]
    relevant_rows = [
        row for row in backbone_rows
        if row.get("base_family") in {family for _, family in ordered_families}
    ]
    baseline_row = top_row(
        [row for row in backbone_rows if row.get("base_family") == baseline_family],
        "best_relaxed_f1",
    )
    if baseline_row is None:
        return []

    ordered_rows = pick_rows(relevant_rows, ordered_families)
    return add_gain_columns(ordered_rows, baseline_row)


def build_findings_summary(run_rows, soup_rows, ensemble_rows, best_rows=None):
    best_rows = list(best_rows) if best_rows is not None else best_by_family(run_rows)
    findings = []

    best_single_relaxed = top_row(run_rows, "best_relaxed_f1")
    if best_single_relaxed:
        findings.append(
            f"Best single-run relaxed result: {describe_row(best_single_relaxed, 'best_relaxed_f1')} from `{best_single_relaxed.get('artifact_root')}`."
        )

    best_single_strict = top_row(run_rows, "best_strict_f1")
    if best_single_strict:
        findings.append(
            f"Best single-run strict result: {describe_row(best_single_strict, 'best_strict_f1')} from `{best_single_strict.get('artifact_root')}`."
        )

    baseline_row = top_row(
        [
            row for row in best_rows
            if row.get("base_family") == "deberta_baseline" and row.get("backbone") == "default"
        ],
        "best_relaxed_f1",
    )
    if baseline_row:
        best_core = top_row(
            [
                row for row in best_rows
                if row.get("backbone") == "default" and row.get("family_group") == "core_ablation"
            ],
            "best_relaxed_f1",
        )
        if best_core and best_core.get("experiment_name") != baseline_row.get("experiment_name"):
            delta = safe_float(best_core.get("best_relaxed_f1")) - safe_float(baseline_row.get("best_relaxed_f1"))
            findings.append(
                f"Within the core default-backbone ablations, the strongest variant improves relaxed F1 over the baseline by {format_delta(delta)}."
            )

        best_extra = top_row(
            [
                row for row in best_rows
                if row.get("backbone") == "default" and row.get("family_group") == "extra_pipeline"
            ],
            "best_relaxed_f1",
        )
        if best_extra:
            delta = safe_float(best_extra.get("best_relaxed_f1")) - safe_float(baseline_row.get("best_relaxed_f1"))
            findings.append(
                f"The best extra pipeline changes relaxed F1 by {format_delta(delta)} relative to the default DeBERTa baseline."
            )

        best_backbone = top_row([row for row in best_rows if row.get("backbone") not in (None, 'default')], "best_relaxed_f1")
        if best_backbone:
            delta = safe_float(best_backbone.get("best_relaxed_f1")) - safe_float(baseline_row.get("best_relaxed_f1"))
            findings.append(
                f"The strongest non-default backbone run changes relaxed F1 by {format_delta(delta)} relative to the default DeBERTa baseline."
            )

    best_soup = top_row(soup_rows, "best_relaxed_f1")
    if best_soup and baseline_row:
        delta = safe_float(best_soup.get("best_relaxed_f1")) - safe_float(baseline_row.get("best_relaxed_f1"))
        findings.append(
            f"The best model soup reaches relaxed F1 {format_metric(best_soup.get('best_relaxed_f1'))}, which is {format_delta(delta)} relative to the default DeBERTa baseline."
        )

    ensemble_candidates = []
    for row in ensemble_rows:
        ensemble_candidates.append(
            {
                "experiment_name": row.get("experiment_name"),
                "best_relaxed_f1": safe_float(row.get("best_overall_by_relaxed_f1", row.get("best_overall_relaxed_f1"))),
                "best_strict_f1": safe_float(row.get("best_overall_by_strict_f1", row.get("best_overall_strict_f1"))),
                "summary": row.get("best_overall_by_relaxed_models", row.get("best_overall_models")),
            }
        )

    best_ensemble_relaxed = top_row(ensemble_candidates, "best_relaxed_f1")
    if best_ensemble_relaxed and best_single_relaxed:
        delta = safe_float(best_ensemble_relaxed.get("best_relaxed_f1")) - safe_float(best_single_relaxed.get("best_relaxed_f1"))
        findings.append(
            f"The best ensemble improves on the best single run by {format_delta(delta)} in relaxed F1."
        )

    best_ensemble_strict = top_row(ensemble_candidates, "best_strict_f1")
    if best_ensemble_strict and best_single_strict:
        delta = safe_float(best_ensemble_strict.get("best_strict_f1")) - safe_float(best_single_strict.get("best_strict_f1"))
        findings.append(
            f"Under strict F1, the best ensemble changes performance by {format_delta(delta)} compared with the best single run."
        )

    if not findings:
        findings.append("No structured experiment findings are available yet.")
    return findings


def build_contribution_explanation_rows(run_rows, soup_rows, ensemble_rows, best_rows=None):
    best_rows = list(best_rows) if best_rows is not None else best_by_family(run_rows)
    baseline_row = top_row(
        [
            row for row in best_rows
            if row.get("base_family") == "deberta_baseline" and row.get("backbone") == "default"
        ],
        "best_relaxed_f1",
    )
    baseline_relaxed = safe_float(baseline_row.get("best_relaxed_f1")) if baseline_row else float("nan")
    baseline_strict = safe_float(baseline_row.get("best_strict_f1")) if baseline_row else float("nan")

    ensemble_candidates = [
        {
            "experiment_name": row.get("experiment_name"),
            "best_relaxed_f1": safe_float(row.get("best_overall_by_relaxed_f1", row.get("best_overall_relaxed_f1"))),
            "best_strict_f1": safe_float(row.get("best_overall_by_strict_f1", row.get("best_overall_strict_f1"))),
            "artifact_root": row.get("artifact_root"),
            "family_display_name": "Ensemble Search",
        }
        for row in ensemble_rows
    ]

    explanation_rows = []
    for spec in CONTRIBUTION_EXPLANATION_SPECS:
        kind = spec["kind"]
        if kind == "run":
            families = set(spec["families"])
            candidates = [row for row in best_rows if row.get("base_family") in families]
        elif kind == "backbone":
            candidates = [row for row in best_rows if row.get("backbone") not in (None, "default")]
        elif kind == "soup":
            candidates = list(soup_rows)
        elif kind == "ensemble":
            candidates = ensemble_candidates
        else:
            candidates = []

        best_relaxed = top_row(candidates, "best_relaxed_f1")
        best_strict = top_row(candidates, "best_strict_f1")
        if best_relaxed is None and best_strict is None:
            continue

        explanation_rows.append(
            {
                "contribution": spec["contribution"],
                "rationale": spec["rationale"],
                "best_relaxed_experiment": best_relaxed.get("experiment_name") if best_relaxed else None,
                "best_relaxed_f1": safe_float(best_relaxed.get("best_relaxed_f1")) if best_relaxed else float("nan"),
                "delta_relaxed_vs_baseline": (
                    safe_float(best_relaxed.get("best_relaxed_f1")) - baseline_relaxed
                    if best_relaxed and math.isfinite(baseline_relaxed)
                    else float("nan")
                ),
                "best_strict_experiment": best_strict.get("experiment_name") if best_strict else None,
                "best_strict_f1": safe_float(best_strict.get("best_strict_f1")) if best_strict else float("nan"),
                "delta_strict_vs_baseline": (
                    safe_float(best_strict.get("best_strict_f1")) - baseline_strict
                    if best_strict and math.isfinite(baseline_strict)
                    else float("nan")
                ),
                "artifact_root": (
                    best_relaxed.get("artifact_root")
                    if best_relaxed and best_relaxed.get("artifact_root")
                    else (best_strict.get("artifact_root") if best_strict else None)
                ),
            }
        )
    return explanation_rows


def build_soup_gain_rows(soup_rows, run_rows):
    run_rows_by_experiment = {row.get("experiment_name"): row for row in run_rows}
    gain_rows = []
    for soup_row in soup_rows:
        source_names = [
            source.strip()
            for source in (soup_row.get("source_experiments") or "").split(",")
            if source.strip()
        ]
        source_rows = [
            run_rows_by_experiment[source_name]
            for source_name in source_names
            if source_name in run_rows_by_experiment
        ]
        best_source_relaxed = top_row(source_rows, "best_relaxed_f1")
        best_source_strict = top_row(source_rows, "best_strict_f1")

        enriched = dict(soup_row)
        enriched["best_source_relaxed_experiment"] = (
            best_source_relaxed.get("experiment_name") if best_source_relaxed else None
        )
        enriched["best_source_relaxed_f1"] = (
            safe_float(best_source_relaxed.get("best_relaxed_f1")) if best_source_relaxed else float("nan")
        )
        enriched["delta_relaxed_vs_best_source"] = (
            safe_float(soup_row.get("best_relaxed_f1")) - safe_float(best_source_relaxed.get("best_relaxed_f1"))
            if best_source_relaxed
            else float("nan")
        )
        enriched["best_source_strict_experiment"] = (
            best_source_strict.get("experiment_name") if best_source_strict else None
        )
        enriched["best_source_strict_f1"] = (
            safe_float(best_source_strict.get("best_strict_f1")) if best_source_strict else float("nan")
        )
        enriched["delta_strict_vs_best_source"] = (
            safe_float(soup_row.get("best_strict_f1")) - safe_float(best_source_strict.get("best_strict_f1"))
            if best_source_strict
            else float("nan")
        )
        gain_rows.append(enriched)

    return gain_rows


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


def plot_learning_curves(
    best_rows,
    logs_by_run_key,
    families,
    backbone="default",
    metrics=None,
    title="Learning Curves",
):
    if plt is None:
        print("matplotlib is not installed; skipping plot.")
        return

    if metrics is None:
        metrics = ["relaxed_f1", "strict_f1", "dev_loss", "train_loss"]

    selected_rows = []
    for family in families:
        candidates = [row for row in best_rows if row.get("base_family") == family and row.get("backbone") == backbone]
        if candidates:
            selected_rows.append(sort_rows_by_metric(candidates)[0])

    if not selected_rows:
        print("No matching best-run rows found for learning curves.")
        return

    available_metrics = []
    for metric in metrics:
        for row in selected_rows:
            entries = logs_by_run_key.get(row["run_key"])
            if not entries:
                continue
            values = [safe_float(entry.get(metric)) for entry in entries]
            if any(math.isfinite(value) for value in values):
                available_metrics.append(metric)
                break

    if not available_metrics:
        print("No learning-curve metrics are available for the selected runs.")
        return

    n_metrics = len(available_metrics)
    ncols = 2 if n_metrics > 1 else 1
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.flatten()
        except AttributeError:
            axes = [axes]

    axis_by_metric = dict(zip(available_metrics, axes))
    for row in selected_rows:
        entries = logs_by_run_key.get(row["run_key"])
        if not entries:
            continue
        epochs = [entry.get("epoch") for entry in entries]
        label = row.get("family_display_name", row["family"])
        for metric in available_metrics:
            values = [safe_float(entry.get(metric)) for entry in entries]
            axis_by_metric[metric].plot(epochs, values, marker="o", linewidth=2, label=label)

    for metric in available_metrics:
        axis = axis_by_metric[metric]
        axis.set_title(f"{title}: {metric_label(metric)}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_label(metric))
        axis.legend()

    for axis in axes[len(available_metrics):]:
        axis.set_visible(False)

    plt.tight_layout()
    plt.show()
