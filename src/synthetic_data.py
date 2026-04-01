"""Synthetic data generation for targeted augmentation."""

import random
import json
from .data import LABEL2ID


# Templates for synthetic data targeting specific failure modes
IMPLICIT_TEMPLATES = [
    {
        "tokens": ["I", "lost", "everything", "because", "of", "the", "pills", "."],
        "ner_tags": ["O", "B-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["Those", "weeks", "were", "absolute", "hell", "."],
        "ner_tags": ["O", "O", "O", "O", "B-ClinicalImpacts", "O"],
    },
    {
        "tokens": ["I", "couldn't", "even", "get", "out", "of", "bed", "for", "months", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "B-ClinicalImpacts", "O"],
    },
    {
        "tokens": ["My", "whole", "life", "fell", "apart", "after", "I", "started", "using", "."],
        "ner_tags": ["O", "O", "O", "B-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "was", "a", "shell", "of", "my", "former", "self", "."],
        "ner_tags": ["O", "O", "O", "B-ClinicalImpacts", "I-ClinicalImpacts", "I-ClinicalImpacts", "I-ClinicalImpacts", "I-ClinicalImpacts", "O"],
    },
    {
        "tokens": ["Nobody", "wanted", "to", "be", "around", "me", "anymore", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "B-SocialImpacts", "O"],
    },
    {
        "tokens": ["I", "hit", "rock", "bottom", "and", "had", "nothing", "left", "."],
        "ner_tags": ["O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["The", "darkness", "consumed", "me", "for", "years", "."],
        "ner_tags": ["O", "B-ClinicalImpacts", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "pushed", "everyone", "away", "during", "that", "time", "."],
        "ner_tags": ["O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "woke", "up", "in", "the", "ER", "not", "knowing", "how", "I", "got", "there", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "B-ClinicalImpacts", "O", "O", "O", "O", "O", "O", "O"],
    },
]

NEGATED_TEMPLATES = [
    {
        "tokens": ["I", "have", "no", "criminal", "record", "thankfully", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "never", "lost", "my", "job", "over", "this", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["Luckily", "I", "didn't", "overdose", "that", "time", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "was", "never", "arrested", "for", "anything", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["My", "family", "didn't", "abandon", "me", "through", "it", "all", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "haven't", "been", "to", "the", "hospital", "for", "it", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["Not", "once", "did", "I", "get", "into", "trouble", "with", "the", "law", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "managed", "to", "avoid", "any", "serious", "withdrawals", "."],
        "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O"],
    },
]

BOUNDARY_TEMPLATES = [
    {
        "tokens": ["I", "got", "arrested", "and", "it", "sent", "me", "into", "a", "deep", "depression", "."],
        "ner_tags": ["O", "B-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "O", "O", "B-ClinicalImpacts", "I-ClinicalImpacts", "O"],
    },
    {
        "tokens": ["The", "withdrawal", "cost", "me", "my", "job", "because", "I", "couldn't", "show", "up", "."],
        "ner_tags": ["O", "B-ClinicalImpacts", "O", "O", "O", "B-SocialImpacts", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "was", "hospitalized", "and", "then", "lost", "custody", "of", "my", "kids", "."],
        "ner_tags": ["O", "O", "B-ClinicalImpacts", "O", "O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O"],
    },
    {
        "tokens": ["My", "anxiety", "made", "it", "impossible", "to", "hold", "down", "a", "job", "."],
        "ner_tags": ["O", "B-ClinicalImpacts", "O", "O", "O", "O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O"],
    },
    {
        "tokens": ["I", "overdosed", "twice", "and", "my", "wife", "left", "me", "."],
        "ner_tags": ["O", "B-ClinicalImpacts", "O", "O", "O", "O", "B-SocialImpacts", "I-SocialImpacts", "O"],
    },
]

SOCIAL_HEAVY_TEMPLATES = [
    {
        "tokens": ["I", "got", "fired", "from", "my", "job", "because", "I", "kept", "calling", "in", "sick", "."],
        "ner_tags": ["O", "B-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "was", "evicted", "and", "ended", "up", "homeless", "for", "a", "while", "."],
        "ner_tags": ["O", "O", "B-SocialImpacts", "O", "O", "O", "B-SocialImpacts", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "spent", "all", "my", "savings", "on", "dope", "and", "went", "bankrupt", "."],
        "ner_tags": ["O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "B-SocialImpacts", "O"],
    },
    {
        "tokens": ["My", "parents", "disowned", "me", "after", "they", "found", "out", "."],
        "ner_tags": ["O", "O", "B-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "was", "charged", "with", "possession", "and", "got", "3", "months", "in", "jail", "."],
        "ner_tags": ["O", "O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O"],
    },
    {
        "tokens": ["I", "lost", "my", "license", "after", "the", "DUI", "."],
        "ner_tags": ["O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O", "O", "B-SocialImpacts", "O"],
    },
    {
        "tokens": ["I", "got", "expelled", "from", "school", "for", "being", "caught", "with", "paraphernalia", "."],
        "ner_tags": ["O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "O", "O"],
    },
    {
        "tokens": ["My", "wife", "divorced", "me", "and", "I", "don't", "see", "my", "kids", "anymore", "."],
        "ner_tags": ["O", "O", "B-SocialImpacts", "I-SocialImpacts", "O", "O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O"],
    },
    {
        "tokens": ["I", "ended", "up", "on", "probation", "for", "2", "years", "."],
        "ner_tags": ["O", "O", "O", "O", "B-SocialImpacts", "O", "O", "O", "O"],
    },
    {
        "tokens": ["I", "had", "to", "drop", "out", "of", "college", "because", "I", "couldn't", "function", "."],
        "ner_tags": ["O", "O", "O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "O"],
    },
]


def generate_synthetic_data(n_per_category: int = 50, seed: int = 42) -> list[dict]:
    """Generate synthetic training examples targeting failure modes."""
    rng = random.Random(seed)
    all_templates = {
        "implicit": IMPLICIT_TEMPLATES,
        "negated": NEGATED_TEMPLATES,
        "boundary": BOUNDARY_TEMPLATES,
        "social_heavy": SOCIAL_HEAVY_TEMPLATES,
    }

    synthetic = []
    for category, templates in all_templates.items():
        for i in range(n_per_category):
            template = rng.choice(templates)
            sample = {
                "tokens": template["tokens"].copy(),
                "ner_tags": template["ner_tags"].copy(),
                "ID": f"synth_{category}_{i}",
                "labels": [],
                "category": category,
            }
            # Derive labels column
            for tag in template["ner_tags"]:
                if tag == "O":
                    sample["labels"].append("_")
                elif "Clinical" in tag:
                    sample["labels"].append("ClinicalImpacts")
                else:
                    sample["labels"].append("SocialImpacts")
            synthetic.append(sample)

    rng.shuffle(synthetic)
    return synthetic


def get_curriculum_order(samples: list[dict], epoch: int, total_epochs: int) -> list[int]:
    """Order samples by difficulty for curriculum learning.
    Early epochs: easy (explicit, Clinical-heavy) -> hard (implicit, Social, boundary)
    Later epochs: all mixed.
    """
    easy_indices = []
    medium_indices = []
    hard_indices = []

    for i, s in enumerate(samples):
        tags = s.get("ner_tags", s.get("raw_tags", []))
        has_clinical = any("Clinical" in t for t in tags)
        has_social = any("Social" in t for t in tags)
        is_synthetic = str(s.get("ID", "")).startswith("synth_")
        category = s.get("category", "")

        if category in ("negated", "boundary") or is_synthetic and category == "implicit":
            hard_indices.append(i)
        elif has_social and not has_clinical:
            medium_indices.append(i)
        else:
            easy_indices.append(i)

    progress = epoch / max(total_epochs - 1, 1)

    if progress < 0.33:
        # Easy first
        return easy_indices + medium_indices[:int(len(medium_indices) * progress * 3)]
    elif progress < 0.66:
        # Add medium
        return easy_indices + medium_indices + hard_indices[:int(len(hard_indices) * (progress - 0.33) * 3)]
    else:
        # All data
        indices = easy_indices + medium_indices + hard_indices
        random.shuffle(indices)
        return indices
