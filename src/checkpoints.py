"""Utilities for managing top-k experiment checkpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _score_tag(score: float) -> str:
    return f"{score:.6f}".replace(".", "p")


class TopKCheckpointManager:
    """Keep the top-k checkpoints for a single experiment."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: str | Path,
        top_k: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir).resolve()
        self.top_k = max(0, int(top_k))
        self.metadata = metadata or {}
        self.checkpoint_dir = self.output_dir / "checkpoints" / experiment_name
        self.records: list[dict[str, Any]] = []

        if self.top_k > 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._clear_existing_artifacts()
            self._write_summary()

    @property
    def enabled(self) -> bool:
        return self.top_k > 0

    def _clear_existing_artifacts(self) -> None:
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def _write_summary(self) -> None:
        if not self.enabled:
            return

        summary = {
            "experiment_name": self.experiment_name,
            "top_k": self.top_k,
            "metadata": _json_ready(self.metadata),
            "checkpoints": _json_ready(self.records),
        }
        with (self.checkpoint_dir / "topk_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _qualifies(self, score: float) -> bool:
        if not self.enabled:
            return False
        if len(self.records) < self.top_k:
            return True
        return score > min(record["score"] for record in self.records)

    def _prune_to_top_k(self) -> None:
        self.records.sort(key=lambda record: (record["score"], record["epoch"]), reverse=True)
        keep = self.records[: self.top_k]
        remove = self.records[self.top_k :]

        for record in remove:
            path = Path(record["path"])
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

        self.records = keep
        self._write_summary()

    def maybe_save_state_dict(
        self,
        state_dict: dict[str, Any],
        score: float,
        epoch: int,
        metrics: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save a PyTorch state dict if it belongs in the current top-k."""

        if not self._qualifies(score):
            return None

        import torch

        checkpoint_path = self.checkpoint_dir / f"epoch{epoch:03d}_f1_{_score_tag(score)}.pt"
        torch.save(state_dict, checkpoint_path)
        self.records.append(
            {
                "epoch": epoch,
                "score": score,
                "path": str(checkpoint_path),
                "metrics": metrics or {},
            }
        )
        self._prune_to_top_k()
        return checkpoint_path

    def maybe_save_directory(
        self,
        save_fn: Callable[[Path], None],
        score: float,
        epoch: int,
        metrics: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save a directory-based checkpoint if it belongs in the current top-k."""

        if not self._qualifies(score):
            return None

        checkpoint_path = self.checkpoint_dir / f"epoch{epoch:03d}_f1_{_score_tag(score)}"
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        save_fn(checkpoint_path)
        self.records.append(
            {
                "epoch": epoch,
                "score": score,
                "path": str(checkpoint_path),
                "metrics": metrics or {},
            }
        )
        self._prune_to_top_k()
        return checkpoint_path
