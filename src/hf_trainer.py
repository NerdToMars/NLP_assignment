"""Hugging Face Trainer helpers for custom token-classification losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from transformers import Trainer


class WeightedTokenClassificationTrainer(Trainer):
    """Trainer that applies weighted cross-entropy to token-classification logits."""

    def __init__(
        self,
        *args,
        class_weights: list[float] | torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        del num_items_in_batch  # Unused, but part of the Trainer hook signature.

        if "labels" not in inputs:
            raise KeyError("WeightedTokenClassificationTrainer requires a 'labels' field in inputs.")

        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)

        if isinstance(outputs, Mapping):
            logits = outputs["logits"]
        else:
            logits = outputs.logits

        num_labels = logits.shape[-1]
        weight_tensor = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-100)
        loss = loss_fn(
            logits.reshape(-1, num_labels),
            labels.reshape(-1),
        )

        if return_outputs:
            return loss, outputs
        return loss
