"""Data loading and preprocessing for Reddit Impacts NER task."""

import ast
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional


LABEL2ID = {"O": 0, "B-ClinicalImpacts": 1, "I-ClinicalImpacts": 2, "B-SocialImpacts": 3, "I-SocialImpacts": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# Entity definitions from annotation guidelines
CLINICAL_DEFINITION = (
    "Clinical Impacts are physical or psychological consequences of substance use "
    "personally experienced by the post author, such as overdose, withdrawal, "
    "addiction, depression, anxiety, hospitalization, or medical treatment."
)
SOCIAL_DEFINITION = (
    "Social Impacts are social, occupational, legal, or relational consequences of "
    "substance use personally experienced by the post author, such as job loss, "
    "arrest, criminal charges, relationship breakdown, financial hardship, or homelessness."
)
ENTITY_DEFINITION_PREFIX = f"{CLINICAL_DEFINITION} {SOCIAL_DEFINITION}"


def parse_list_col(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x.strip())
    raise TypeError(f"Unsupported type: {type(x)}")


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tokens"] = df["tokens"].apply(parse_list_col)
    df["ner_tags"] = df["ner_tags"].apply(parse_list_col)
    return df


def preprocess_tokens(tokens: list[str]) -> list[str]:
    """Minimal preprocessing: replace usernames and URLs with placeholders."""
    out = []
    for t in tokens:
        if t.startswith("u/") or t.startswith("/u/"):
            out.append("[USER]")
        elif re.match(r"https?://", t):
            out.append("[URL]")
        else:
            out.append(t)
    return out


class NERDataset(Dataset):
    """Token classification dataset for transformer models."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        definition_prompting: bool = False,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.definition_prompting = definition_prompting

        for _, row in df.iterrows():
            tokens = preprocess_tokens(row["tokens"])
            ner_tags = row["ner_tags"]
            label_ids = [LABEL2ID[t] for t in ner_tags]

            if definition_prompting:
                encoding = self._encode_with_definition(tokens, label_ids)
            else:
                encoding = self._encode(tokens, label_ids)

            if encoding is not None:
                encoding["id"] = row["ID"]
                encoding["raw_tokens"] = tokens
                encoding["raw_tags"] = ner_tags
                self.samples.append(encoding)

    def _encode(self, tokens, label_ids):
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                aligned_labels.append(label_ids[wid])
            else:
                # For subword continuations, use I- tag if B- tag, else same
                lab = label_ids[wid]
                if ID2LABEL[lab].startswith("B-"):
                    entity_type = ID2LABEL[lab][2:]
                    aligned_labels.append(LABEL2ID[f"I-{entity_type}"])
                else:
                    aligned_labels.append(lab)
            prev_word_id = wid

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
            "word_ids": word_ids,
        }

    def _encode_with_definition(self, tokens, label_ids):
        """Encode with entity definitions prepended, masking definition tokens from loss."""
        # Tokenize the definition prefix
        def_tokens = self.tokenizer.tokenize(ENTITY_DEFINITION_PREFIX)
        sep_token = self.tokenizer.sep_token or "[SEP]"

        # Tokenize the actual text
        text_encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length - len(def_tokens) - 2,  # Leave room for def + sep
            add_special_tokens=False,
        )

        # Build combined input: [CLS] definition [SEP] text tokens [SEP]
        def_input_ids = self.tokenizer.convert_tokens_to_ids(def_tokens)
        cls_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        input_ids = [cls_id] + def_input_ids + [sep_id] + text_encoding["input_ids"] + [sep_id]
        def_len = 1 + len(def_input_ids) + 1  # CLS + def + SEP

        # Align labels for text portion
        word_ids_text = text_encoding.word_ids(batch_index=0)
        aligned_labels = [-100] * def_len  # Mask definition from loss

        prev_word_id = None
        for wid in word_ids_text:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                aligned_labels.append(label_ids[wid])
            else:
                lab = label_ids[wid]
                if ID2LABEL[lab].startswith("B-"):
                    entity_type = ID2LABEL[lab][2:]
                    aligned_labels.append(LABEL2ID[f"I-{entity_type}"])
                else:
                    aligned_labels.append(lab)
            prev_word_id = wid

        aligned_labels.append(-100)  # Final SEP

        # Pad
        pad_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id or 0] * pad_len
        aligned_labels = aligned_labels + [-100] * pad_len

        # Build word_ids including definition portion (None for definition)
        word_ids = [None] * def_len + word_ids_text + [None] * (1 + pad_len)

        return {
            "input_ids": torch.tensor(input_ids[:self.max_length], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:self.max_length], dtype=torch.long),
            "labels": torch.tensor(aligned_labels[:self.max_length], dtype=torch.long),
            "word_ids": word_ids[:self.max_length],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": s["input_ids"],
            "attention_mask": s["attention_mask"],
            "labels": s["labels"],
        }

    def get_full_sample(self, idx):
        return self.samples[idx]


class BiLSTMDataset(Dataset):
    """Dataset for BiLSTM-CRF using word-level tokens."""

    def __init__(self, df: pd.DataFrame, word2idx: dict, max_length: int = 256):
        self.samples = []
        self.max_length = max_length

        for _, row in df.iterrows():
            tokens = preprocess_tokens(row["tokens"])
            ner_tags = row["ner_tags"]

            token_ids = [word2idx.get(t.lower(), word2idx.get("<UNK>", 1)) for t in tokens]
            label_ids = [LABEL2ID[t] for t in ner_tags]

            length = min(len(token_ids), max_length)
            token_ids = token_ids[:max_length]
            label_ids = label_ids[:max_length]

            # Pad
            pad_len = max_length - length
            token_ids += [0] * pad_len
            label_ids += [0] * pad_len

            self.samples.append({
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "labels": torch.tensor(label_ids, dtype=torch.long),
                "length": length,
                "id": row["ID"],
                "raw_tokens": tokens,
                "raw_tags": ner_tags,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": s["input_ids"],
            "labels": s["labels"],
            "length": s["length"],
        }


def build_vocab(df: pd.DataFrame, min_freq: int = 1) -> dict:
    """Build vocabulary from training data."""
    from collections import Counter
    counter = Counter()
    for _, row in df.iterrows():
        for token in row["tokens"]:
            counter[token.lower()] += 1

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)
    return word2idx


def load_glove_embeddings(glove_path: str, word2idx: dict, dim: int = 300):
    """Load GloVe embeddings for vocabulary."""
    import numpy as np
    embeddings = np.random.normal(0, 0.1, (len(word2idx), dim)).astype(np.float32)
    embeddings[0] = 0  # PAD

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"GloVe: found {found}/{len(word2idx)} words")
    return torch.tensor(embeddings)
