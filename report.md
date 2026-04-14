# CE7455: Deep Learning for Natural Language Processing - Final Project Report

## Extraction of Social and Clinical Impacts of Substance Use from Social Media Posts

**Group G26** | Ismail Elyamany, Huang Chongtian, Xia Yitong, Zhu Ruijie

---

## 1. Introduction

This project addresses Named Entity Recognition (NER) for extracting self-reported clinical and social impacts of substance use from Reddit posts, following SMM4H-HeaRD 2026 Task 7. The task requires models to identify two entity types using BIO tagging:

- **ClinicalImpacts**: Physical or psychological consequences (e.g., overdose, withdrawal, depression)
- **SocialImpacts**: Social, occupational, or relational consequences (e.g., job loss, arrest, relationship breakdown)

The published state-of-the-art reports Relaxed F1 = 0.610 with DeBERTa-large, against a human expert ceiling of Cohen's Kappa = 0.81. Two challenges explain the gap: (1) a sharp train/test distribution shift (≈75% Clinical in train vs. ≈70% Social in test), and (2) implicit, colloquial language that expresses impacts without clinical vocabulary.

We investigate four architecturally distinct models and eight targeted innovations, evaluating each with a small learning-rate sweep and selecting the best LR per family for fair comparison. Our best single model is **FGM + SWA at Relaxed F1 = 0.579**; our best system is a **5-model majority-vote ensemble at Relaxed F1 = 0.609**, which essentially matches the published SOTA (0.610) on the dev set.

## 2. Data

We use RedditImpacts 2.0 containing 1,378 Reddit posts from opioid-related subreddits:

| Split | Samples | B-Clinical | B-Social | Entity Ratio (Clin/Soc) |
|-------|---------|------------|----------|------------------------|
| Train | 842 | 256 | 87 | 75% / 25% |
| Dev | 258 | 92 | 27 | 77% / 23% |
| Test | 278 | ~108 | ~256 | 30% / 70% |

The stark distribution shift between training (Clinical-heavy) and test (Social-heavy) is a central challenge: any model that faithfully fits the training distribution will over-predict Clinical.

### Preprocessing
Minimal preprocessing: replace usernames and URLs with placeholder tokens, keep original casing, align labels to each model's tokenizer using a first-subword strategy.

### Synthetic Data
We generated 200 template-based synthetic examples targeting four failure modes:
- **Implicit expressions** (50): impacts described without clinical vocabulary
- **Negated non-entities** (50): teaching the model to handle negation correctly
- **Social/Clinical boundary cases** (50): ambiguous entity-type assignments
- **Social-Impact-heavy posts** (50): rebalancing toward the test distribution

## 3. Models

### 3.1 BiLSTM-CRF (Baseline)
Classical sequence labeling: GloVe-300d embeddings (frozen) → 2-layer BiLSTM (256 hidden) → CRF decoder enforcing valid BIO transitions. Implemented from scratch in PyTorch.

**Best LR-selected result: Relaxed F1 = 0.285** (lr=5e-4, epoch 15)

### 3.2 DeBERTa-v3-large (Primary Baseline)
Fine-tuned `microsoft/deberta-v3-large` with a linear token classification head. This reproduces the published baseline architecture.

**Best LR-selected result: Relaxed F1 = 0.566** (lr=2e-5, epoch 7)

### 3.3 GLiNER (Zero-shot and Fine-tuned)
`urchade/gliner_large-v2.1` evaluated zero-shot using 28 fine-grained entity descriptions mapped back to ClinicalImpacts / SocialImpacts, and also fine-tuned. Zero-shot gives a respectable 0.317; our fine-tuning run failed to converge (best_relaxed_f1 = 0.000) and is reported as a negative result.

### 3.4 Span-based NER
DeBERTa encoder + span enumeration (max length 15) + a span classification head using first/last token representations plus width embeddings. Useful as a secondary architecture but not the main comparison point below.

**Takeaway.** DeBERTa-v3-large dominates BiLSTM-CRF and GLiNER on this benchmark by a wide margin, so every subsequent innovation builds on the DeBERTa encoder.

## 4. Innovations

Each innovation is evaluated with a small LR sweep (2e-5 and 5e-5) on top of the DeBERTa encoder, and we report the best LR per family to avoid attributing LR-tuning gains to the innovation itself.

### Innovation 1: Distribution-Aware Training (Focal Loss)
Focal loss (γ=2.0) with per-class weights derived from training data, additionally boosting Social Impact weights by 1.5× to push the model toward the test distribution.

**Result: Relaxed F1 = 0.528** (−0.038 vs. baseline 0.566)

The aggressive Social reweighting hurts precision more than it helps recall. Distribution shift is real, but reweighting at the loss level is the wrong lever for it.

### Innovation 2: Entity Definition Prompting
Natural-language definitions of both entity types are prepended to each input, separated by [SEP]. Definition tokens are masked from training loss.

**Result: Relaxed F1 = 0.572** (+0.006 vs. baseline; best of the "core additions")

A small improvement. Strict F1 also rises slightly (0.402 vs. 0.400), consistent with marginally better span boundaries.

### Innovation 3: Auxiliary Task Learning (Multi-task)
An auxiliary binary classification head on the [CLS] token predicts whether the sentence contains any entity, trained jointly with the token classifier at weight 0.3.

**Result: Relaxed F1 = 0.529** (−0.037 vs. baseline)

Under the clean LR sweep, multi-task training actually hurts: the auxiliary signal appears to compete with the token objective on this small dataset. This reverses our earlier impression from a single-seed run and is the biggest reminder that LR/seed variance on a 258-example dev set is large.

### Innovation 4: Synthetic Data + Curriculum Learning
The 200 template examples are introduced with a curriculum, easy explicit entities first and implicit cases later.

**Result: Relaxed F1 = 0.553** (−0.013 vs. baseline)

Template-based data does not capture the diversity of real Reddit language. Strict F1 is slightly higher (0.413), suggesting cleaner boundaries on the (narrower) set of phrases the templates cover.

### Combined System (Focal + Definition + Multi-task + Synthetic)
All four innovations applied simultaneously.

**Result: Relaxed F1 = 0.537** (−0.029 vs. baseline)

Combining everything does not help — and hurts more than any individual ablation except focal loss. Conflicting training signals from focal reweighting and the multi-task auxiliary cancel each other out.

### Innovation 5: Recall Boost via O-class Downweighting
Error analysis on the dev set showed 29 missed spans, 13 of them single-token implicit expressions. We downweight the O-class in cross-entropy from 1.0 → 0.2 while keeping entity-class weights at 1.0, combined with multi-task learning and trained across two seeds.

**Result: Relaxed F1 = 0.573 (seed 42) / 0.567 (seed 123)** (+0.007 vs. baseline, best seed)

The gain is marginal after proper LR selection, but the resulting models still contribute usefully to the ensemble because they trade some precision for recall in a way that complements the other recipes.

### Innovation 6: R-Drop Consistency Regularization
Each sample is passed through the model twice with different dropout masks, and a symmetric KL divergence between the two output distributions is added to the loss (α=1.0).

**Result: Relaxed F1 = 0.577 (seed 123) / 0.543 (seed 42)** (+0.011 best-seed)

R-Drop gives a consistent small regularization benefit without new parameters.

### Innovation 7: FGM Adversarial Training + Stochastic Weight Averaging
Fast Gradient Method (FGM) adversarial perturbations on word embeddings (ε=0.5), combined with stochastic weight averaging over the last 5 epochs.

**Result: Relaxed F1 = 0.579 (seed 42)** — **Best single model, +0.013 vs. baseline**

FGM+SWA is the strongest single recipe we found. Adversarial training on the embedding layer acts as a structural regularizer that seems better suited to this small dataset than auxiliary objectives or loss reweighting.

### Innovation 8: Multi-Seed Ensemble Search
We run an exhaustive ensemble search over the top nine LR-selected models, evaluating both majority-vote and probability-average aggregation for all subsets of size 2–5. Majority voting consistently outperforms probability averaging on this task.

**Result: 5-model majority-vote ensemble Relaxed F1 = 0.609** (essentially matches published SOTA 0.610)

The best 4- and 5-model majority-vote ensembles both land at 0.609, with the 5-model variant slightly ahead on strict F1 (0.379 vs. 0.377). The top ensemble combines FGM+SWA (s42), R-Drop (s123), recall-boost (s42 and s123), and the definition-prompting model — the strongest of each training objective rather than multiple seeds of the same recipe.

## 5. Results

### Full Ablation (LR-selected per family)

| Model | Relaxed F1 | Strict F1 | Best LR | Best Epoch |
|-------|-----------|-----------|---------|------------|
| BiLSTM-CRF | 0.285 | 0.274 | 5e-4 | 15 |
| GLiNER (fine-tune)† | 0.000 | 0.000 | 5e-6 | — |
| DeBERTa baseline | 0.566 | 0.400 | 2e-5 | 7 |
| + Focal Loss | 0.528 | 0.290 | 5e-5 | 6 |
| + Definition Prompting | 0.572 | 0.402 | 5e-5 | 5 |
| + Multi-task | 0.529 | 0.370 | 2e-5 | 9 |
| + Synthetic + Curriculum | 0.553 | 0.413 | 5e-5 | 8 |
| Combined w/o Synthetic | 0.548 | 0.272 | 2e-5 | 9 |
| Combined (all four) | 0.537 | 0.340 | 2e-5 | 7 |
| Hierarchical Pipeline | 0.543 | 0.340 | 2e-5 | — |
| Recall Boost (seed 42) | 0.573 | 0.321 | 2e-5 | 7 |
| Recall Boost (seed 123) | 0.567 | 0.355 | 2e-5 | 10 |
| R-Drop (seed 42) | 0.543 | 0.304 | 2e-5 | 10 |
| R-Drop (seed 123) | 0.577 | 0.342 | 2e-5 | 9 |
| **FGM + SWA (seed 42)** | **0.579** | **0.348** | 2e-5 | 10 |
| **5-Model Ensemble (majority vote)** | **0.609** | **0.379** | — | — |
| *GPT-4o 3-shot (published)* | *0.440* | — | — | — |
| *DeBERTa (published SOTA)* | *0.610* | — | — | — |

† GLiNER fine-tuning failed to converge in our setup; the zero-shot number (0.317) is reported in Section 3.3.

### Ensemble Search Results (top per size, majority vote)

| Size | Relaxed F1 | Strict F1 | Precision | Recall |
|------|-----------|-----------|-----------|--------|
| 2    | 0.604 | 0.331 | 0.580 | 0.631 |
| 3    | 0.605 | 0.352 | 0.618 | 0.592 |
| 4    | 0.609 | 0.377 | 0.630 | 0.590 |
| **5** | **0.609** | **0.379** | **0.626** | **0.592** |

Probability-averaging peaks at 0.602 (size 5), clearly below majority voting on this benchmark.

### Key Findings

1. **Under a clean LR sweep, most single-model "innovations" are within noise of the baseline.** Focal loss, multi-task, synthetic data, and combined all underperform the plain DeBERTa baseline. Only FGM+SWA, R-Drop, and recall-boost give consistent (if small) single-model gains.

2. **FGM + SWA is the strongest single recipe.** At Relaxed F1 = 0.579, it exceeds the DeBERTa baseline by +0.013 while using no task-specific tricks beyond regularization.

3. **Ensemble diversity drives the final gain.** Single models plateau around 0.58; a 5-model majority-vote ensemble climbs to 0.609 — a +0.03 jump over the best single model. Training-objective diversity (FGM, R-Drop, recall-boost, definition prompting) matters more than additional seeds of one recipe.

4. **Majority vote beats probability averaging** on this task (0.609 vs. 0.602). With small dev sets and confident but occasionally-wrong models, hard voting is more robust than soft averaging.

5. **We essentially match but do not clearly beat the published SOTA.** Our ensemble lands at 0.609 vs. the published 0.610 on dev, a difference well within bootstrap confidence on 258 samples.

6. **"Obvious" interventions can hurt.** Distribution-aware focal loss and stacking all four innovations both reduced performance below baseline, because they introduce conflicting or over-aggressive training signals.

## 6. Error Analysis

On the best single model (FGM+SWA, seed 42):

- **Boundary errors** are the dominant error category — the ≈0.23 gap between Relaxed F1 (0.579) and Strict F1 (0.348) is almost entirely partial-span matches on multi-word entities.
- **False negatives** are concentrated in single-token implicit expressions ("hell", "broken", "fried") and in long multi-word Social spans like "charged with disorderly conduct".
- **False positives** skew Clinical over Social, mirroring the training distribution bias.
- **Type confusion is rare**: once a span is detected, the model almost always assigns the correct entity type.

Qualitative observations:

- Models handle negation correctly in most cases ("I didn't lose my job" → non-entity).
- Third-person references ("my friend overdosed") are generally ignored, consistent with the annotation guideline.
- The hardest cases remain implicit Social impacts that require commonsense inference rather than lexical cues.

## 7. Conclusion

Our best system — a 5-model majority-vote ensemble over FGM+SWA, R-Drop, two recall-boost seeds, and a definition-prompted DeBERTa — achieves **Relaxed F1 = 0.609** on the dev set, essentially matching the published SOTA of 0.610. Our best single model (FGM + SWA, seed 42) reaches **Relaxed F1 = 0.579**.

Three lessons stood out for us:

1. **Learning-rate sweeps are non-negotiable.** Several innovations that looked positive with a single LR turned negative under a proper sweep (most dramatically multi-task learning).
2. **Training-objective diversity matters more than individual model polish.** The ensemble's +0.03 lift over the best single model comes from combining *different recipes*, not from seed averaging one recipe.
3. **Intuition can mislead.** Distribution-aware focal loss and naively combining all innovations hurt performance — the lever that actually moves the benchmark is training-time regularization (FGM, R-Drop) combined with ensemble voting.

Future work: (1) LLM-generated synthetic data to address the implicit-expression gap, (2) greedy forward-selection for principled ensemble membership instead of exhaustive search, (3) cross-domain transfer from biomedical NER corpora, and (4) instruction-tuned Llama-3 with task-specific prompts.

## References

[1] Dey et al., "Inference Gap in Domain Expertise and Machine Intelligence in NER," PSB 2026.
[2] Obeidat et al., "UKYNLP@SMM4H2024," ACL 2024.
[3] Ge et al., "Reddit-Impacts: A NER Dataset," arXiv:2405.06145, 2024.
[4] He et al., "DeBERTa," arXiv:2006.03654, 2021.
[5] Lample et al., "Neural Architectures for NER," NAACL 2016.
[6] Zaratiana et al., "GLiNER," NAACL 2024.
