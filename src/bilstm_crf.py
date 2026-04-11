"""BiLSTM-CRF model for NER."""

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Conditional Random Field layer."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        # Enforce BIO constraints with large negative scores
        with torch.no_grad():
            # I- cannot follow O or a different entity type's B/I
            for i in range(num_tags):
                for j in range(num_tags):
                    from_tag = self._tag_name(i)
                    to_tag = self._tag_name(j)
                    if to_tag.startswith("I-"):
                        to_entity = to_tag[2:]
                        if from_tag == "O":
                            self.transitions.data[i, j] = -10000.0
                        elif from_tag.startswith("B-") or from_tag.startswith("I-"):
                            from_entity = from_tag[2:]
                            if from_entity != to_entity:
                                self.transitions.data[i, j] = -10000.0

    def _tag_name(self, idx):
        from .data import ID2LABEL
        return ID2LABEL.get(idx, "O")

    def forward(self, emissions, tags, mask):
        """Compute negative log likelihood."""
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        return (forward_score - gold_score).mean()

    def _forward_algorithm(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # (B, T, 1)
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # (B, 1, T)
            next_score = broadcast_score + self.transitions + broadcast_emissions  # (B, T, T)
            next_score = torch.logsumexp(next_score, dim=1)  # (B, T)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            m = mask[:, i].bool()
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            emit = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            score = score + (trans + emit) * m.float()

        last_tag_idx = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)
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


class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF for NER with pretrained embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_tags: int = 5,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Freeze GloVe

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, input_ids, labels=None, length=None):
        mask = (input_ids != 0).float()
        embeds = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(self.dropout(lstm_out))

        if labels is not None:
            loss = self.crf(emissions, labels, mask)
            return {"loss": loss, "emissions": emissions}
        else:
            preds = self.crf.decode(emissions, mask)
            return {"predictions": preds, "emissions": emissions}
