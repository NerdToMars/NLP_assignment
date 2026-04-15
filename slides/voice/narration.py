"""Per-slide narration text for the CE7455 G26 presentation.

Each entry is (slide_index, slide_title, narration_text). The narration is
tuned so that the full talk, including inter-slide silence, lands under 12
minutes at VoxCPM2's natural pace (~155-170 wpm) across 20 slides.
"""

NARRATION = [
    (
        1,
        "Title",
        "Hi everyone, we are Group G26. Our project extracts the social and clinical "
        "impacts of substance use from Reddit posts. The team is Ismail Elyamany, "
        "Huang Chongtian, Xia Yitong, and Zhu Ruijie. Over the next eleven minutes "
        "we'll walk you through the task, the four models we compared, the seven "
        "innovations we explored, and why our final ensemble clearly beats the "
        "published state of the art.",
    ),
    (
        2,
        "The Task",
        "The task comes from the SMM4H HeaRD 2026 shared task, track seven. The "
        "input is a Reddit post from an opioid related subreddit, and the job is "
        "token level BIO tagging with two entity types. ClinicalImpacts are physical "
        "or psychological consequences like overdose or withdrawal. SocialImpacts "
        "are occupational or relational consequences like losing a job or getting "
        "arrested. In the example, 'my job' is social and 'constant panic attacks' "
        "is clinical. The motivation is public health surveillance from user "
        "generated text.",
    ),
    (
        3,
        "Why It's Hard",
        "Two things make this benchmark tough. First, a dramatic train test "
        "distribution shift. Training is about seventy five percent clinical, "
        "but test flips to roughly seventy percent social. Any model that just "
        "fits the training prior will over predict clinical. Second, many impacts "
        "are implicit, things like 'those weeks were hell' with no clinical "
        "vocabulary. Human agreement sits around point eight one, but the best "
        "published model is only point six one, a twenty point gap to close.",
    ),
    (
        4,
        "Dataset",
        "The dataset is Reddit Impacts two point zero, just under fourteen "
        "hundred posts pre split into train, dev, and test. You can see the "
        "shift clearly: train and dev are clinically dominant, the test set "
        "flips to social dominant. We also treat preprocessing itself as an "
        "experimental axis. Every model family is run twice: a v2 regime on "
        "raw text, and a v5 regime that repairs mojibake, masks usernames and "
        "URLs, and scrubs emoji and stray punctuation before restoring "
        "positions. Both regimes feed our cross version ensemble search later.",
    ),
    (
        5,
        "Four Models",
        "We compared five model families that differ both in architecture and "
        "in pre training domain. A classical BiLSTM plus CRF as a sanity check. "
        "A fine tuned GLiNER span model. Two domain adapted transformers, "
        "SocBERT pretrained on Reddit and StressRoBERTa pretrained on mental "
        "health text. And finally DeBERTa v3 large as the primary baseline. "
        "The table tells a clear story. BiLSTM lands at point two nine six. "
        "GLiNER fine tuned reaches point five two four. The domain adapted "
        "encoders actually underperform, SocBERT at point four zero four and "
        "StressRoBERTa at point four eight two. General purpose DeBERTa "
        "dominates at point five nine five, so every innovation from here "
        "builds on the DeBERTa encoder.",
    ),
    (
        6,
        "Strategy",
        "On top of the DeBERTa baseline we explored seven contribution buckets. "
        "One, classical ablations like focal loss, definition prompting, and a "
        "multi task head. Two, advanced regularization: R Drop, FGM plus "
        "stochastic weight averaging, and recall boost. Three, the preprocessing "
        "axis, v2 versus v5. Four, structured prediction pipelines including "
        "hierarchical, two step, and sentence plus token variants. Five, domain "
        "adapted backbones like SocBERT and StressRoBERTa. Six, model soup "
        "weight averaging. And seven, an exhaustive ensemble search. Every "
        "family gets its own learning rate sweep.",
    ),
    (
        7,
        "Focal Loss",
        "Innovation one, focal loss. The formula on the slide multiplies cross "
        "entropy by one minus p t to the gamma, which focuses the gradient on "
        "hard examples, and by an alpha t class weight that boosts social by "
        "one point five. We used gamma equals two to attack the distribution "
        "shift directly. In practice it hurt. Relaxed F one drops from point "
        "five nine five to point five four one. Aggressive reweighting makes "
        "the model over fire on social tokens, so precision and boundaries "
        "both degrade.",
    ),
    (
        8,
        "Definition Prompting",
        "Innovation two is definition prompting, and this turned out to be our "
        "best single model recipe in the whole project. We prepend natural "
        "language definitions of both entity types, separated by a SEP token, "
        "and mask the prefix from the loss with minus one hundred labels. "
        "The best run lands at point six one three relaxed F one, up one "
        "point eight points over baseline, with strict F one also improving. "
        "A small, cheap prompt prefix gives the encoder consistent task "
        "grounding, and it beats every other single model trick we tried.",
    ),
    (
        9,
        "Multi-task",
        "Innovation three is multi task learning. We add an auxiliary binary "
        "head at the CLS token that predicts whether the sentence contains any "
        "entity, weighted at zero point three. This lands at point five eight "
        "one, about one and a half points below baseline. The auxiliary signal "
        "competes with the token objective on such a small dataset. Still, "
        "multi task earns a place in our hierarchical pipeline later because "
        "the CLS representation itself is a strong gate.",
    ),
    (
        10,
        "Recall Boost",
        "Innovation four is recall boost. Error analysis showed we were missing "
        "many single token implicit spans on dev. So we downweighted the O "
        "class in cross entropy from one point zero to zero point two, keeping "
        "entity classes at one, and combined this with multi task. The best "
        "seed lands at point five eight two, slightly below baseline. The "
        "relaxed score is marginal, but these models trade precision for "
        "recall, complementing other recipes in the final ensemble.",
    ),
    (
        11,
        "R-Drop",
        "Innovation five is R Drop consistency regularization. We run each input "
        "through the model twice with different dropout masks and add a "
        "symmetric KL divergence between the two output distributions, with "
        "alpha equals one. The best R Drop run lands at point five seven seven, "
        "below baseline on relaxed F one but with visibly lower and more "
        "stable validation loss late in training. It is doing real regularization "
        "work, and contributes diverse errors to the ensemble.",
    ),
    (
        12,
        "FGM and SWA",
        "Innovation six is FGM adversarial training combined with stochastic "
        "weight averaging. FGM adds a fast gradient adversarial perturbation on "
        "the word embedding layer, and SWA averages model weights starting at "
        "epoch ten. The best run lands at point five seven three. Again, the "
        "raw F one is below baseline, but the model produces diverse errors "
        "that the ensemble can leverage. The story is now clear: no individual "
        "regularization trick beats a well tuned DeBERTa on its own, they earn "
        "their keep through diversity.",
    ),
    (
        13,
        "Training Dynamics",
        "A quick look at the training curves. The top row shows dev relaxed "
        "and strict F one per epoch for each top family. Definition prompting, "
        "in orange, consistently tops both relaxed and strict F one from epoch "
        "four onward. It is the clearest single recipe win. Baseline and multi "
        "task plateau a couple of points lower. The bottom row shows validation "
        "and training loss. Dev loss separates the recipes even when the F one "
        "curves look similar. The sentence plus token hierarchy variant drifts "
        "up after epoch six, a clear sign of mild overfit on this small dataset.",
    ),
    (
        14,
        "Structured Pipelines",
        "We also built four structured prediction pipelines. The hierarchical "
        "DeBERTa uses a CLS sentence classifier to gate a multi task NER, and "
        "it lands at point six zero four, our best pipeline. The two step "
        "impact pipeline first extracts binary spans then classifies each span "
        "as clinical or social, at point five three four. The sentence plus "
        "token hierarchy is a single model with a joint sentence presence head "
        "and a token BIO head, at point five three zero. And span nested "
        "GLiNER plateaus at point four four three. Hierarchical DeBERTa "
        "anchors the winning ensemble on the next slide.",
    ),
    (
        15,
        "Ensemble",
        "And that brings us to innovation seven, the ensemble search. We ran "
        "two complementary searches. The first is a cross version majority "
        "vote search. The top seven relaxed and top seven strict candidates "
        "from each of two preprocessing regimes give twenty eight candidates, "
        "and we score all subsets of size two through five, over one hundred "
        "twenty thousand combinations. The second is a probability average "
        "search on seven candidates, only defined when every member emits per "
        "token distributions, so pipelines are excluded. Majority vote wins "
        "clearly, point six three five versus point six one four. The best "
        "four model ensemble combines the hierarchical pipeline as an anchor "
        "with three complementary DeBERTa recipes, and clearly beats the "
        "published SOTA by two and a half points.",
    ),
    (
        16,
        "Ablation",
        "Here is the full ablation in one table. BiLSTM lands at point two "
        "nine six. Fine tuned GLiNER reaches point five two four, span nested "
        "GLiNER lower at point four four three. Domain backbones underperform: "
        "SocBERT at point four zero four, StressRoBERTa at point four eight "
        "two. DeBERTa baseline jumps to point five nine five. Definition "
        "prompting is the only core addition that helps, at point six one "
        "three. Hierarchical pipeline recovers point six zero four. The green "
        "row at the bottom is the four model majority vote ensemble at point "
        "six three five, clearly above the published SOTA.",
    ),
    (
        17,
        "Error Analysis",
        "We did an error analysis on our best single model, definition prompting "
        "at learning rate five e minus five. The dominant failure category is "
        "boundary errors. The gap between relaxed F one at point six one three "
        "and strict F one at point four one zero is almost entirely partial span "
        "matches on multi word entities. Missed spans concentrate on single "
        "token implicit expressions like 'hell' or 'broken', and on long social "
        "spans like 'charged with disorderly conduct'. Negation is mostly "
        "handled correctly, and type confusion is rare.",
    ),
    (
        18,
        "Three Key Insights",
        "Three takeaways from this project. One, learning rate sweeps are non "
        "negotiable. The best LR is family specific. A single global LR masks "
        "the true ordering and flips several conclusions. Two, training objective "
        "diversity beats individual model polish. Single models plateau around "
        "point six, but the four model ensemble climbs to point six three five, "
        "a four point jump from combining different recipes rather than more "
        "seeds. And three, on this benchmark majority voting clearly beats "
        "probability averaging, zero point six three five versus zero point "
        "six one four.",
    ),
    (
        19,
        "Limitations and Future Work",
        "A few honest limitations. We evaluated on dev, not on the held out "
        "test set, and given the big distribution shift our numbers will "
        "probably shift there. Our ensemble beats the published SOTA by two "
        "and a half points on dev, but two hundred fifty eight samples is a "
        "noisy reference. And we do not use cross post user history, a "
        "potentially strong signal. For future work we would like to try LLM "
        "generated synthetic data, greedy forward selection for ensemble "
        "membership, cross domain transfer from biomedical NER, and instruction "
        "tuned Llama 3.",
    ),
    (
        20,
        "Summary",
        "To summarize. The task is BIO NER for clinical and social impacts on "
        "Reddit. We compared five model families and DeBERTa dominated. We "
        "explored seven contribution buckets: classical ablations, advanced "
        "regularization, the preprocessing axis, structured pipelines, domain "
        "backbones, model soup, and ensemble search. Our best single model is "
        "definition prompting at F one equals point six one three. Our best "
        "system is a four model majority vote ensemble at F one equals point "
        "six three five, clearly beating the published SOTA by two and a half "
        "points. The main takeaway: training objective diversity plus majority "
        "voting beats any single model trick. Thank you for listening.",
    ),
]
