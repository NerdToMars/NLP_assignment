"""DeBERTa-based NER models with innovations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha  # Per-class weights tensor
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B, seq_len, num_classes), targets: (B, seq_len)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DeBERTaNER(nn.Module):
    """DeBERTa with token classification head for NER."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        num_labels: int = 5,
        dropout: float = 0.1,
        use_focal_loss: bool = False,
        focal_alpha: list = None,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.num_labels = num_labels

        if use_focal_loss:
            alpha = torch.tensor(focal_alpha, dtype=torch.float32) if focal_alpha else None
            self.loss_fn = FocalLoss(alpha=alpha, gamma=focal_gamma)
        else:
            self.loss_fn = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state.float())
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.loss_fn is not None:
                loss = self.loss_fn(logits, labels)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels.view(-1),
                    ignore_index=-100,
                )

        return {"loss": loss, "logits": logits}


class DeBERTaNERMultiTask(nn.Module):
    """DeBERTa NER with auxiliary tasks: entity presence detection and negation scope."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        num_labels: int = 5,
        dropout: float = 0.1,
        use_focal_loss: bool = False,
        focal_alpha: list = None,
        focal_gamma: float = 2.0,
        aux_weight: float = 0.3,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.aux_weight = aux_weight

        # Main NER head
        self.ner_classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Auxiliary head 1: entity presence per sentence (binary)
        self.entity_presence_head = nn.Linear(self.config.hidden_size, 2)

        # Auxiliary head 2: negation scope per token (binary)
        self.negation_head = nn.Linear(self.config.hidden_size, 2)

        if use_focal_loss:
            alpha = torch.tensor(focal_alpha, dtype=torch.float32) if focal_alpha else None
            self.ner_loss_fn = FocalLoss(alpha=alpha, gamma=focal_gamma)
        else:
            self.ner_loss_fn = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        entity_presence_labels=None,
        negation_labels=None,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state.float())

        # Main NER
        logits = self.ner_classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.ner_loss_fn is not None:
                loss = self.ner_loss_fn(logits, labels)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels.view(-1),
                    ignore_index=-100,
                )

            # Auxiliary task 1: entity presence (use CLS token)
            if entity_presence_labels is not None:
                cls_output = sequence_output[:, 0]
                presence_logits = self.entity_presence_head(cls_output)
                presence_loss = F.cross_entropy(presence_logits, entity_presence_labels)
                loss = loss + self.aux_weight * presence_loss

            # Auxiliary task 2: negation scope
            if negation_labels is not None:
                neg_logits = self.negation_head(sequence_output)
                neg_loss = F.cross_entropy(
                    neg_logits.view(-1, 2),
                    negation_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss + self.aux_weight * neg_loss

        return {"loss": loss, "logits": logits}


class SpanNER(nn.Module):
    """Span-based NER: enumerate spans and classify each."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        num_labels: int = 3,  # None, ClinicalImpacts, SocialImpacts
        max_span_length: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.max_span_length = max_span_length
        self.num_labels = num_labels

        # Span classifier takes first + last token repr + width embedding
        self.width_embedding = nn.Embedding(max_span_length, 64)
        self.span_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2 + 64, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size, num_labels),
        )

    def _enumerate_spans(self, seq_len, attention_mask=None):
        """Generate all candidate spans up to max_span_length."""
        spans = []
        for start in range(seq_len):
            for end in range(start, min(start + self.max_span_length, seq_len)):
                spans.append((start, end))
        return spans

    def forward(self, input_ids, attention_mask=None, span_labels=None, spans=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        batch_size, seq_len, hidden = sequence_output.shape

        if spans is None:
            spans = self._enumerate_spans(seq_len)

        # Compute span representations
        all_span_reprs = []
        for start, end in spans:
            start_repr = sequence_output[:, start]
            end_repr = sequence_output[:, end]
            width = torch.tensor([end - start], device=input_ids.device).expand(batch_size)
            width_repr = self.width_embedding(width)
            span_repr = torch.cat([start_repr, end_repr, width_repr], dim=-1)
            all_span_reprs.append(span_repr)

        span_reprs = torch.stack(all_span_reprs, dim=1)  # (B, num_spans, hidden*2+64)
        span_logits = self.span_classifier(span_reprs)  # (B, num_spans, num_labels)

        loss = None
        if span_labels is not None:
            loss = F.cross_entropy(
                span_logits.view(-1, self.num_labels),
                span_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": span_logits, "spans": spans}

    def decode_to_bio(self, span_logits, spans, seq_len):
        """Convert span predictions to BIO tags."""
        preds = span_logits.argmax(dim=-1)  # (B, num_spans)
        batch_size = preds.shape[0]
        span_label_map = {0: None, 1: "ClinicalImpacts", 2: "SocialImpacts"}

        all_bio = []
        for b in range(batch_size):
            bio = ["O"] * seq_len
            # Collect predicted spans with scores
            predicted_spans = []
            for i, (start, end) in enumerate(spans):
                label_idx = preds[b, i].item()
                if label_idx != 0:
                    score = span_logits[b, i, label_idx].item()
                    predicted_spans.append((start, end, span_label_map[label_idx], score))

            # Sort by score descending, apply greedily (no overlap)
            predicted_spans.sort(key=lambda x: x[3], reverse=True)
            occupied = set()
            for start, end, label, score in predicted_spans:
                span_positions = set(range(start, end + 1))
                if span_positions & occupied:
                    continue
                occupied |= span_positions
                bio[start] = f"B-{label}"
                for pos in range(start + 1, end + 1):
                    bio[pos] = f"I-{label}"

            all_bio.append(bio)
        return all_bio
