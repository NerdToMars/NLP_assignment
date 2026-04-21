"""Per-slide narration text for the CE7455 G26 presentation.

Each entry is (slide_index, slide_title, narration_text). The narration is
tuned so that the full talk lands under 12 minutes at a natural pace (~160 wpm) 
across all 22 slides.
"""

NARRATION = [
 (
 1,
 "Title",
 "Hi everyone, we are Group G26. Our project focuses on extracting the social and "
 "clinical impacts of substance use from Reddit. We will show you how we pushed past current state-of-the-art "
 "performance for this challenging NLP task."
 ),
 (
 2,
 "The Task",
 "This project is part of the SMM4H-HeaRD 2026 Task 7 contest, focusing on token-level NER "
 "in opioid-related subreddits. We use BIO tagging to identify two entity types. "
 "Clinical impacts cover physical consequences like overdose or withdrawal, while "
 "social impacts include outcomes like job loss or broken relationships. This mining "
 "is critical for real-time public health surveillance and harm-reduction research."
 ),
 (
 3,
 "Why It's Hard",
 "Two factors make this benchmark particularly difficult. First, there is a massive "
 "train-test distribution shift. While training data is 75% clinical, "
 "the test set flips to roughly 70% social, causing standard models to over-predict "
 "clinical impacts. Second, the language is highly colloquial. Spans "
 "like 'those weeks were hell' have no clinical vocabulary or explicit triggers, "
 "leaving a 20-point gap between the best models and the human ceiling."
 ),
 (
 4,
 "Dataset",
 "Our dataset, Reddit-Impacts 2.0, contains nearly 1,400 annotated posts. "
 "Social spans are notably longer and have a heavier tail than clinical ones. "
 "Crucially, most development posts contain no entities at all, requiring "
 "the model to be highly precise in its detection to avoid false positives."
 ),
 (
 5,
 "Preprocessing",
 "We compared two runtime regimes: a raw approach that preserves lexical noise like "
 "usernames and emojis for fidelity, and a preprocessed regime that repairs "
 "broken Unicode and masks URLs. We found that while cleaning helps "
 "models like BiLSTM and GLiNER, the plain DeBERTa baseline actually performs stronger "
 "on raw text."
 ),
 (
 6,
 "Baselines",
 "We benchmarked five model families. BiLSTM-CRF served as a sanity check, "
 "while fine-tuned GLiNER provided a span-based comparison. We also tested "
 "domain-adapted backbones like SocBERT and StressRoBERTa. General-purpose "
 "DeBERTa-v3-large dominated the raw regime with an F1 of 0.595, "
 "becoming the foundation for our subsequent innovations."
 ),
 (
 7,
 "Protocol",
 "Our evaluation protocol relies on Relaxed F1, which tolerates boundary mismatches "
 "as long as the entity type is correct and the span overlaps. We chose the "
 "best learning rate per model family within each regime. All results "
 "reported here are on the dev-set, treating preprocessing as an experimental axis "
 "rather than a hidden implementation detail."
 ),
 (
 8,
 "Strategy",
 "Beyond the baselines, we explored seven contribution buckets. This "
 "includes loss and prompt ablations, advanced regularization like R-Drop "
 "and FGM, and structured prediction pipelines. We also "
 "tested model soups and exhaustive ensemble searches, performing "
 "a small learning rate sweep for every single innovation."
 ),
 (
 9,
 "Innovation 1",
 "Our first innovation was Distribution-Aware Focal Loss. We used a focal "
 "factor of 2.0 to concentrate on hard tokens and set alpha-Social to 1.5 "
 "to address the distribution shift. However, we found that aggressive "
 "reweighting caused the model to over-fire on social tokens, dropping performance "
 "below the plain baseline in both regimes."
 ),
 (
 10,
 "Innovation 2",
 "Innovation two is Definition Prompting, our best single-model recipe. "
 "We prepend natural-language definitions of both entity types and mask them from "
 "the loss. This allows the encoder to see an explicit label ontology "
 "on every example. This approach reached an F1 of 0.613 on preprocessed "
 "text, surpassing our baseline."
 ),
 (
 11,
 "Innovation 3",
 "Innovation three used a Multi-task Auxiliary Head. We shared the encoder "
 "between token NER and a sentence-level 'entity-presence' classifier. "
 "While it slightly hurt performance on raw text, it recovered a small gain on "
 "preprocessed data. This auxiliary head proved more useful as a "
 "feature source for later structured recipes."
 ),
 (
 12,
 "Innovation 4",
 "Innovation four addressed recall issues through O-class downweighting. "
 "We dropped the cross-entropy weight for 'O' tokens to 0.2 while boosting social "
 "weights. While this lift in strict F1 was modest, it "
 "continues the trade-off of precision for recall, producing unique errors that "
 "complement the other models in our final ensemble."
 ),
 (
 13,
 "Innovation 5",
 "Innovation five is R-Drop Consistency Regularization. We perform two "
 "stochastic forward passes on the same batch and force their distributions to "
 "agree using symmetric KL divergence. This regularization "
 "makes R-Drop competitive in the preprocessed regime, reaching an F1 of 0.577."
 ),
 (
 14,
 "Innovation 6",
 "Innovation six combined FGM adversarial training with Stochastic Weight "
 "Averaging. FGM perturbs word embeddings during training, "
 "and SWA averages checkpoints from late epochs to smooth oscillations. "
 "This robustness recipe scored slightly below the baseline but added much-needed "
 "diversity to our prediction space."
 ),
 (
 15,
 "Training Dynamics",
 "Looking at the dynamics, Definition Prompting consistently tops both Relaxed "
 "and Strict F1 from epoch four onward, marking it as our clearest single-recipe "
 "win. We also observed that dev loss separates our recipes even "
 "when F1 curves look close, notably catching a mild overfit in the "
 "Sentence-Token Hierarchy after epoch six."
 ),
 (
 16,
 "Pipelines",
 "We also tested four structured prediction variants. Our "
 "Hierarchical DeBERTa uses a CLS classifier to gate the NER process. "
 "It was the clear relaxed-F1 winner in this category at 0.604. "
 "Interestingly, its strict F1 actually improves under preprocessing even "
 "as the relaxed score drops."
 ),
 (
 17,
 "Innovation 7",
 "Innovation seven is our Ensemble Search. We performed an "
 "exhaustive search over 122,000 combinations for majority voting. "
 "Hard voting clearly outperformed soft probability averaging. "
 "Our best four-model ensemble combines the hierarchical anchor with "
 "three complementary DeBERTa recipes, reaching an F1 of 0.635."
 ),
 (
 18,
 "Ablation",
 "This full ablation table summarizes our progress. Every family "
 "reports its best learning rate from our sweeps. While the DeBERTa "
 "baseline is strong at 0.595, our ensemble provides a significant jump to "
 "0.635. Note that our strict-optimal ensemble reached a strict "
 "F1 of 0.518 using a different 5-model subset."
 ),
 (
 19,
 "Error Analysis",
 "Error analysis of our best single model reveals that boundary errors still "
 "dominate, with a 20-point gap between relaxed and strict scores. "
 "Missed spans are mostly concentrated on single-token implicit expressions "
 "like 'hell'. Encouragingly, type confusion and negation "
 "handling are rarely issues for the model."
 ),
 (
 20,
 "Insights",
 "We have three key insights. First, LR sweeps are non-negotiable; "
 "a global LR would have flipped our conclusions on which recipes were "
 "actually best. Second, objective diversity beats individual "
 "polish. Finally, hard majority voting is significantly more "
 "robust than probability averaging on this noisy dataset."
 ),
 (
 21,
 "Limitations",
 "Our limitations include having only dev-set access; we expect numbers to "
 "shift under the real test distribution. Our ensemble "
 "gap also needs test-set confirmation. Furthermore, we "
 "only looked at single-sentence inputs, missing the user-history context "
 "that cross-post analysis could provide."
 ),
 (
 22,
 "Summary",
 "To summarize, we benchmarked five models across seven contribution buckets "
 "under rigorous learning rate sweeps. Our best system—a "
 "4-model majority-vote ensemble—reached an F1 of 0.635. This "
 "beats the published state-of-the-art by 2.5 points, proving that "
 "objective diversity beats any single-model trick."
 ),
]