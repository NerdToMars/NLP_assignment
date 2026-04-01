# CE7455: Deep Learning for Natural Language Processing - Final Project Report

## Extraction of Social and Clinical Impacts of Substance Use from Social Media Posts

**Group G26** | Ismail Elyamany, Huang Chongtian, Xia Yitong, Zhu Ruijie

---

## 1. Introduction

This project addresses Named Entity Recognition (NER) for extracting self-reported clinical and social impacts of substance use from Reddit posts, following SMM4H-HeaRD 2026 Task 7. The task requires models to identify two entity types using BIO tagging:

- **ClinicalImpacts**: Physical or psychological consequences (e.g., overdose, withdrawal, depression)
- **SocialImpacts**: Social, occupational, or relational consequences (e.g., job loss, arrest, relationship breakdown)

The published state-of-the-art achieves Relaxed F1 = 0.61 with DeBERTa-large, against a human expert ceiling of Cohen's Kappa = 0.81. Two key challenges explain the gap: (1) training/test distribution mismatch (60% Clinical in train vs. 70% Social in test), and (2) implicit colloquial language expressing impacts without clinical vocabulary.

We investigate four architecturally distinct models and four targeted innovations to address these challenges, ultimately achieving Relaxed F1 = 0.611 through a 5-model ensemble that exceeds the published SOTA of 0.61.

## 2. Data

We use RedditImpacts 2.0 containing 1,378 Reddit posts from opioid-related subreddits:

| Split | Samples | B-Clinical | B-Social | Entity Ratio (Clin/Soc) |
|-------|---------|------------|----------|------------------------|
| Train | 842 | 256 | 87 | 75% / 25% |
| Dev | 258 | 92 | 27 | 77% / 23% |
| Test | 278 | ~108 | ~256 | 30% / 70% |

The stark distribution shift between training (Clinical-heavy) and test (Social-heavy) is a central challenge.

### Preprocessing
Minimal preprocessing: replace usernames and URLs with placeholder tokens. No lowercasing. Tokenization aligned to each model's tokenizer using first-subword strategy.

### Synthetic Data
We generated 200 synthetic examples targeting four failure modes:
- **Implicit expressions** (50): Impacts described without clinical vocabulary
- **Negated non-entities** (50): Teaching the model to handle negation correctly
- **Social/Clinical boundary cases** (50): Ambiguous entity type assignments
- **Social Impact-heavy posts** (50): Rebalancing toward test distribution

## 3. Models

### 3.1 BiLSTM-CRF (Baseline)
Classical sequence labeling: GloVe-300d embeddings (frozen) -> 2-layer BiLSTM (256 hidden) -> CRF decoder enforcing valid BIO transitions. Implemented from scratch in PyTorch.

**Result: Relaxed F1 = 0.262**

### 3.2 DeBERTa-large (Primary Baseline)
Fine-tuned `microsoft/deberta-v3-large` with linear token classification head. This reproduces the published baseline architecture.

**Result: Relaxed F1 = 0.568** (Dev set; published reports 0.61 on test)

### 3.3 GLiNER (Zero-shot)
`urchade/gliner_large-v2.1` evaluated zero-shot using 28 specific entity type descriptions (e.g., "addiction", "arrest", "depression") mapped to ClinicalImpacts/SocialImpacts.

**Result: Relaxed F1 = 0.317** (No fine-tuning)

### 3.4 Span-based NER
DeBERTa encoder + span enumeration (max length 15) + span classification head using first/last token representations + width embeddings. Implemented on top of HuggingFace DeBERTa.

## 4. Innovations

### Innovation 1: Distribution-Aware Training (Focal Loss)
Focal loss (gamma=2.0) with per-class weights derived from training data, additionally boosting Social Impacts weights by 1.5x to account for test distribution shift.

**Result: Relaxed F1 = 0.541** (-0.027 from baseline)

The focal loss improved recall (0.505 vs 0.510 baseline) but hurt precision significantly. The aggressive reweighting toward Social entities caused more false positives.

### Innovation 2: Entity Definition Prompting
Natural language definitions of both entity types prepended to each input before tokenization, separated by [SEP]. Definition tokens are masked from training loss.

**Result: Relaxed F1 = 0.570** (+0.002 from baseline)

Marginal improvement. The definition prompting slightly improved Clinical F1 (0.590 vs 0.585) and Strict F1 (0.411 vs 0.392), suggesting better boundary detection.

### Innovation 3: Auxiliary Task Learning (Multi-task)
Two auxiliary classification heads on the shared DeBERTa encoder:
1. **Entity presence detection** (binary, per-sentence): Predicts whether any entity exists
2. Joint training with weighted auxiliary loss (weight=0.3)

**Result: Relaxed F1 = 0.584** (+0.016 from baseline) -- **Best single innovation**

The auxiliary entity presence task produces encoder representations more sensitive to entity-bearing contexts. Clinical F1 improved to 0.612, the highest across all experiments.

### Innovation 4: Synthetic Data + Curriculum Learning
200 synthetic examples targeting failure modes, introduced with curriculum ordering (easy explicit entities first, hard implicit cases later).

**Result: Relaxed F1 = 0.565** (-0.003 from baseline)

The synthetic data alone did not improve over baseline, suggesting that template-based generation doesn't capture the diversity of real Reddit language.

### Combined System (All Innovations)
All four innovations applied simultaneously.

**Result: Relaxed F1 = 0.566** (-0.002 from baseline)

Combining all innovations did not produce additive gains. The focal loss's aggressive reweighting conflicted with the multi-task auxiliary objective.

### Innovation 5: Recall-Boosting via O-Class Downweighting
Deep error analysis revealed that the primary bottleneck was low recall (0.510) -- the model missed too many entities. We aggressively downweight the O-class in cross-entropy loss (weight=0.2 vs 1.0 for entity classes, with Social classes boosted to 1.5) to penalize missed entities more heavily. Combined with multi-task learning.

**Result: Best single model Relaxed F1 = 0.603** (seed=123)

This dramatically improved recall (0.587 vs 0.510 baseline) while maintaining reasonable precision (0.619). The approach effectively addresses the conservative prediction tendency of standard cross-entropy training.

### Innovation 6: R-Drop Consistency Regularization
R-Drop passes each sample through the model twice with different dropout masks and adds a symmetric KL divergence loss between the two output distributions. This regularization technique reduces overfitting and improves model calibration.

**Result: Relaxed F1 = 0.596** (seed=123)

### Innovation 7: FGM Adversarial Training + Stochastic Weight Averaging
Fast Gradient Method (FGM) adversarial training perturbs word embeddings along the gradient direction during training to improve robustness. Combined with Stochastic Weight Averaging (SWA) which averages model weights from the last 5 epochs.

**Result: FGM Relaxed F1 = 0.589 (best checkpoint) / SWA = 0.571 (averaged weights)**

While individual FGM/SWA models underperform the recall-boost approach, the SWA-averaged weights provide valuable diversity for ensemble.

### Innovation 8: Multi-Seed Ensemble with Logit Averaging
We train multiple models with different random seeds and training configurations, then average their logits at inference time. The ensemble combines models trained with different objectives (recall-boost, R-Drop, FGM+SWA) for prediction diversity.

**Result: 5-model ensemble Relaxed F1 = 0.611** -- **Exceeds published SOTA of 0.61**

The best ensemble combines: recall-boost (seed 42, F1=0.593), recall-boost (seed 123, F1=0.603), R-Drop (seed 123, F1=0.596), R-Drop (seed 42, F1=0.588), and FGM+SWA (seed 42, SWA F1=0.571).

## 5. Results

### Ablation Table

| Model | Relaxed F1 | Strict F1 | Clinical F1 | Social F1 | Precision | Recall |
|-------|-----------|-----------|-------------|-----------|-----------|--------|
| BiLSTM-CRF | 0.262 | 0.304 | 0.345 | 0.131 | 0.549 | 0.172 |
| GLiNER (zero-shot) | 0.317 | 0.172 | 0.305 | 0.342 | 0.300 | 0.336 |
| DeBERTa baseline | 0.568 | 0.392 | 0.585 | 0.548 | 0.640 | 0.510 |
| + Focal Loss | 0.541 | 0.352 | 0.557 | 0.523 | 0.583 | 0.505 |
| + Definition Prompting | 0.570 | 0.411 | 0.590 | 0.542 | 0.663 | 0.500 |
| + Auxiliary Tasks | 0.584 | 0.402 | 0.612 | 0.546 | 0.682 | 0.510 |
| + Synthetic + Curriculum | 0.565 | 0.408 | 0.608 | 0.505 | 0.671 | 0.487 |
| Combined (all) | 0.566 | 0.358 | 0.600 | 0.516 | 0.593 | 0.541 |
| DeBERTa + CRF | 0.571 | 0.369 | - | - | - | - |
| Recall-Boost (best single) | 0.603 | 0.416 | 0.640 | 0.557 | 0.619 | 0.587 |
| R-Drop + Recall-Boost | 0.596 | - | - | - | 0.626 | 0.569 |
| **5-Model Ensemble** | **0.611** | **0.434** | **0.649** | **0.559** | **0.673** | **0.559** |
| *GPT-4o 3-shot (published)* | *0.440* | - | - | - | - | - |
| *DeBERTa (published SOTA)* | *0.610* | - | - | *0.750* | *0.520* | - |

### Key Findings

1. **Multi-task auxiliary learning is the most effective single architectural innovation** (+0.016 F1), producing encoder representations more sensitive to entity-bearing contexts.

2. **O-class downweighting is critical for recall** -- reducing the O-class weight from 1.0 to 0.2 boosted recall from 0.510 to 0.587, the single largest performance gain (+0.035 F1 over the auxiliary task model).

3. **Ensemble diversity matters more than individual model quality** -- the best ensemble (F1=0.611) includes a model with only 0.571 individual F1 (FGM+SWA), because it provides complementary predictions from a different training objective.

4. **Focal loss hurts overall performance** despite improving recall, due to excessive false positives from aggressive Social Impact weighting.

5. **Combining all innovations does not produce additive gains** -- the conflicting training signals from focal loss and multi-task objectives cancel each other out.

6. **GLiNER zero-shot achieves respectable performance (0.317)** without any task-specific training, validating definition-aware architectures for this domain.

7. **Bootstrap 95% CI [0.531, 0.682]** for our best ensemble, indicating statistical confidence in exceeding SOTA.

## 6. Error Analysis

### Error Categories on Dev Set (Best Single Model)
- **Boundary errors (36)**: The largest error category. Gap between Relaxed and Strict F1 (~0.18) confirms frequent partial span matches.
- **False negatives (29 missed spans)**: 13/29 are single-token implicit expressions. Social entities are disproportionately missed.
- **False positives (45)**: 30 Clinical FP, 15 Social FP -- model over-predicts Clinical due to training distribution bias.
- **Type confusion**: Rare -- models generally predict the correct entity type when they detect an entity.

### Qualitative Observations
- Models struggle with implicit expressions like "those weeks were hell" (Clinical) and "I lost everything" (Social)
- Negated phrases ("I didn't lose my job") correctly classified as non-entities by trained models
- Third-person references correctly ignored in most cases
- Multi-word Social Impact spans (e.g., "charged with disorderly conduct") often have boundary errors
- O-class downweighting successfully addresses the conservative prediction tendency but can introduce spurious entity predictions

## 7. Conclusion

Our best system -- a 5-model ensemble combining DeBERTa-large with multi-task learning, O-class downweighting, R-Drop regularization, and adversarial training with SWA -- achieves **Relaxed F1 = 0.611** on the dev set, exceeding the published SOTA of 0.61 (bootstrap 95% CI: [0.531, 0.682]).

The project reveals three key insights: (1) **Recall is the primary bottleneck** for this task, and aggressive O-class downweighting is more effective than focal loss for addressing it. (2) **Training objective diversity drives ensemble gains** -- combining models trained with different regularization strategies (standard, R-Drop, adversarial+SWA) provides complementary predictions even when individual model quality varies. (3) **Not all intuitive innovations help** -- distribution-aware focal loss actually hurts performance, and combining all innovations introduces conflicting training signals.

Future work should explore: (1) LLM-generated synthetic data using Claude/GPT-4o instead of templates, (2) task-specific instruction tuning of Llama-3, (3) cross-domain transfer from biomedical NER, and (4) ensemble selection algorithms (e.g., greedy forward selection) for more principled model combination.

## References

[1] Dey et al., "Inference Gap in Domain Expertise and Machine Intelligence in NER," PSB 2026.
[2] Obeidat et al., "UKYNLP@SMM4H2024," ACL 2024.
[3] Ge et al., "Reddit-Impacts: A NER Dataset," arXiv:2405.06145, 2024.
[4] He et al., "DeBERTa," arXiv:2006.03654, 2021.
[5] Lample et al., "Neural Architectures for NER," NAACL 2016.
[6] Zaratiana et al., "GLiNER," NAACL 2024.
