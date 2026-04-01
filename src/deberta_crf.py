"""DeBERTa + CRF: addresses boundary errors by enforcing valid BIO transitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from .data import NUM_LABELS, ID2LABEL


class LinearCRF(nn.Module):
    """Lightweight CRF on top of transformer emissions."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        self._init_constraints()

    def _init_constraints(self):
        """Initialize transition constraints for BIO scheme."""
        with torch.no_grad():
            for i in range(self.num_tags):
                for j in range(self.num_tags):
                    to_tag = ID2LABEL.get(j, "O")
                    from_tag = ID2LABEL.get(i, "O")
                    if to_tag.startswith("I-"):
                        to_entity = to_tag[2:]
                        if from_tag == "O":
                            self.transitions.data[j, i] = -10000.0
                        elif from_tag.endswith(to_entity):
                            pass  # Valid
                        else:
                            self.transitions.data[j, i] = -10000.0
            # I- cannot start a sequence
            for j in range(self.num_tags):
                if ID2LABEL.get(j, "O").startswith("I-"):
                    self.start_transitions.data[j] = -10000.0

    def forward(self, emissions, tags, mask):
        """Negative log-likelihood."""
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        return (forward_score - gold_score).mean()

    def _forward_algorithm(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            m = mask[:, i].bool()
            trans = self.transitions[tags[:, i], tags[:, i - 1]]
            emit = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            score = score + (trans + emit) * m.float()

        last_idx = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def decode(self, emissions, mask):
        """Viterbi decoding."""
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        score += self.end_transitions
        _, best_last_tag = score.max(dim=1)

        best_paths = [best_last_tag]
        for hist in reversed(history):
            best_last_tag = hist.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last_tag)
        best_paths.reverse()
        return torch.stack(best_paths, dim=1)


class DeBERTaCRF(nn.Module):
    """DeBERTa with CRF layer for boundary-aware NER."""

    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        num_labels=NUM_LABELS,
        dropout=0.1,
        use_lstm=False,
        lstm_hidden=256,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels

        hidden_size = self.config.hidden_size
        if use_lstm:
            self.lstm = nn.LSTM(
                hidden_size, lstm_hidden, num_layers=1,
                bidirectional=True, batch_first=True,
            )
            self.emission = nn.Linear(lstm_hidden * 2, num_labels)
        else:
            self.lstm = None
            self.emission = nn.Linear(hidden_size, num_labels)

        self.crf = LinearCRF(num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state.float())

        if self.lstm is not None:
            sequence_output, _ = self.lstm(sequence_output)
            sequence_output = self.dropout(sequence_output)

        emissions = self.emission(sequence_output)

        if labels is not None:
            # Replace -100 with 0 for CRF (handle with mask)
            crf_labels = labels.clone()
            label_mask = (labels != -100).float()
            crf_labels[labels == -100] = 0

            loss = self.crf(emissions, crf_labels, label_mask)

            # Also return logits for evaluation
            return {"loss": loss, "logits": emissions}
        else:
            mask = attention_mask.float() if attention_mask is not None else torch.ones_like(input_ids).float()
            preds = self.crf.decode(emissions, mask)
            return {"logits": emissions, "predictions": preds}


class DeBERTaCRFMultiTask(nn.Module):
    """DeBERTa + CRF + auxiliary entity presence head."""

    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        num_labels=NUM_LABELS,
        dropout=0.1,
        aux_weight=0.3,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.aux_weight = aux_weight

        self.emission = nn.Linear(self.config.hidden_size, num_labels)
        self.crf = LinearCRF(num_labels)

        # Auxiliary: entity presence
        self.entity_presence_head = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, labels=None, entity_presence_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state.float())
        emissions = self.emission(sequence_output)

        if labels is not None:
            crf_labels = labels.clone()
            label_mask = (labels != -100).float()
            crf_labels[labels == -100] = 0

            loss = self.crf(emissions, crf_labels, label_mask)

            if entity_presence_labels is not None:
                cls_output = sequence_output[:, 0]
                presence_logits = self.entity_presence_head(cls_output)
                presence_loss = F.cross_entropy(presence_logits, entity_presence_labels)
                loss = loss + self.aux_weight * presence_loss

            return {"loss": loss, "logits": emissions}
        else:
            mask = attention_mask.float() if attention_mask is not None else torch.ones_like(input_ids).float()
            preds = self.crf.decode(emissions, mask)
            return {"logits": emissions, "predictions": preds}
