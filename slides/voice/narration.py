"""Per-slide narration text for the CE7455 G26 presentation.

Each entry is (slide_index, slide_title, narration_text). The narration is
written to be read aloud in roughly 35 to 45 seconds per slide at a natural
pace, targeting an 11 to 13 minute total talk across 18 slides.
"""

NARRATION = [
    (
        1,
        "Title",
        "Hi everyone, we're Group G26. Our project is about extracting the social and clinical "
        "impacts of substance use from Reddit posts. This is the final presentation for CE7455. "
        "The team is Ismail Elyamany, Huang Chongtian, Xia Yitong, and Zhu Ruijie. "
        "Over the next ten minutes we'll walk you through the task, the models we tried, the "
        "eight innovations we explored, and why, after a proper learning rate sweep, the final "
        "story looks quite different from what we initially thought.",
    ),
    (
        2,
        "The Task",
        "So let's start with the task. It comes from the SMM4H HeaRD 2026 shared task, track 7. "
        "The input is a single Reddit post from an opioid related subreddit, and the job is a "
        "token level NER problem with B I O tagging. There are only two entity types. "
        "ClinicalImpacts are the physical or psychological consequences, things like overdose, "
        "withdrawal, or depression. SocialImpacts are the social, occupational, or relational "
        "consequences, like losing a job, getting arrested, or a broken relationship. "
        "In the example on the slide, 'my job' is a social impact, and 'constant panic attacks' "
        "is a clinical impact. The motivation is public health surveillance: mining "
        "population level signals directly from user generated text.",
    ),
    (
        3,
        "Why It's Hard",
        "There are two things that make this benchmark really difficult. "
        "The first is a pretty dramatic train test distribution shift. The training set is "
        "about seventy five percent clinical and only twenty five percent social, but the test "
        "set is the exact opposite, roughly thirty percent clinical and seventy percent social. "
        "So any model that just fits the training distribution is going to over predict clinical. "
        "The second challenge is that a lot of the impacts are expressed implicitly. People write "
        "things like 'those weeks were hell' or 'I lost everything' without any clinical "
        "vocabulary at all. On the right you can see the reported ceilings. The human expert "
        "agreement is around point eight one, but the best published model is only point six one, "
        "so there's a twenty point gap to close.",
    ),
    (
        4,
        "Dataset",
        "The dataset is Reddit Impacts two point zero. Just under fourteen hundred posts from "
        "opioid related subreddits, pre split into train, dev, and test. "
        "You can see the distribution shift clearly in this table: the training and dev splits "
        "are clinically dominant, but the test split flips to social dominant. "
        "Preprocessing is deliberately minimal. We replace user names and URLs with placeholders, "
        "we keep the original casing, and we align labels to each model's tokenizer using a "
        "first subword strategy. We also generated two hundred synthetic examples to target "
        "four specific failure modes: implicit expressions, negated non entities, "
        "boundary cases between clinical and social, and social impact heavy posts.",
    ),
    (
        5,
        "Four Models",
        "We started by trying four architecturally very different models, so we'd have a broad "
        "baseline to build on. "
        "First, a classical BiLSTM plus CRF using frozen GloVe embeddings, as a sanity check. "
        "Second, DeBERTa v3 large with a linear token classification head, which is the primary "
        "baseline and matches the published state of the art architecture. "
        "Third, GLiNER, a generalist zero shot NER model, evaluated with twenty eight specific "
        "sub labels that we map back to clinical and social. "
        "And fourth, a span based DeBERTa model that enumerates all spans up to length fifteen. "
        "The table at the bottom shows what we found: DeBERTa dominates the BiLSTM and GLiNER by "
        "a wide margin, landing at point five six six relaxed F one under our learning rate sweep. "
        "From this point on, every innovation builds on the DeBERTa encoder.",
    ),
    (
        6,
        "Strategy",
        "On top of the DeBERTa baseline, we explored eight targeted innovations. "
        "One, focal loss with test aware class weights to try to attack the distribution shift. "
        "Two, entity definition prompting, where we prepend natural language definitions. "
        "Three, multi task learning with an auxiliary entity presence head. "
        "Four, synthetic data combined with curriculum learning. "
        "Five, O class downweighting combined with multi task, to bias the model toward recall. "
        "Six, R Drop consistency regularization. "
        "Seven, FGM adversarial training combined with stochastic weight averaging. "
        "And eight, an exhaustive ensemble search with majority voting. "
        "Each of these is evaluated with a small learning rate sweep, and we report the best "
        "learning rate per family. The winners, as you'll see, are not the ones we expected.",
    ),
    (
        7,
        "Focal Loss",
        "Let's talk about innovation one, focal loss. We used gamma equals two, with the social "
        "class weight boosted by one point five to push the model toward social predictions. "
        "The idea was to directly attack the distribution shift. "
        "In practice, it made things much worse. The baseline had a relaxed F one of point five "
        "six six, and with focal loss it dropped to point five two eight, nearly four points down. "
        "Strict F one drops even more sharply, from point four zero to point two nine. "
        "What was happening is that the aggressive reweighting made the model over fire on social "
        "tokens, so both precision and span boundaries degraded. "
        "The lesson is that distribution shift is real, but class weighting is the wrong lever "
        "for it.",
    ),
    (
        8,
        "Prompting and Multi-task",
        "Innovations two and three are on this slide, and this is where the story got interesting. "
        "Definition prompting, where we prepend natural language definitions of both entity types "
        "separated by a SEP token and mask the prefix from the loss, gave us a small gain, from "
        "point five six six up to point five seven two. It's the best of the core additions. "
        "Multi task was the surprise. Adding an auxiliary binary head at the CLS token that "
        "predicts whether the sentence contains any entity, weighted at zero point three, "
        "actually hurts under the learning rate sweep. We drop from point five six six to point "
        "five two nine, a loss of nearly four points. The auxiliary signal appears to compete "
        "with the token objective on such a small dataset. This was the biggest reminder that "
        "single seed, single learning rate results can be very misleading.",
    ),
    (
        9,
        "Synthetic Data",
        "Innovation four is synthetic data with curriculum learning. We wrote templates targeting "
        "the four failure modes I mentioned earlier, generated two hundred examples, and fed them "
        "in from easy to hard across epochs. "
        "The result was slightly negative, about one point three points down from baseline. And "
        "when we combined all four innovations so far, focal loss, definition prompting, multi "
        "task, and synthetic data, the combined model dropped further to point five three seven, "
        "almost three points below baseline. "
        "Two lessons here. First, template based synthetic data doesn't capture the diversity of "
        "real Reddit language. And second, stacking innovations is definitely not free. The focal "
        "loss signal and the multi task signal were fighting each other during training, and "
        "adding more ingredients only made it worse.",
    ),
    (
        10,
        "Recall Boost",
        "This is innovation five. After the combined model disappointed us, we did a deep error "
        "analysis and found that recall was lagging behind precision. We were missing twenty nine "
        "spans on the dev set, thirteen of which were single token implicit expressions. "
        "So we tried something very direct: we downweighted the O class in the cross entropy "
        "loss, from one point zero to zero point two, while keeping entity classes at one point "
        "zero. We combined this with multi task learning and trained two seeds. "
        "The best seed lands at point five seven three relaxed F one, about seven thousandths "
        "above the baseline. So the relaxed gain is actually marginal after a proper learning "
        "rate sweep. But the recall boosted models still earn their place in the ensemble, "
        "because they trade precision for recall in a way that complements the other recipes.",
    ),
    (
        11,
        "R-Drop and FGM",
        "Innovations six and seven are two more regularization tricks, and this is where we get "
        "our best single model. "
        "R Drop runs each input through the model twice with different dropout masks, and adds a "
        "symmetric KL divergence between the two output distributions. With alpha equals one, "
        "the seed one two three run lands at point five seven seven, plus one point one over "
        "baseline. "
        "FGM is fast gradient adversarial training on the word embeddings, combined with "
        "stochastic weight averaging. The seed forty two run lands at point five seven nine, "
        "which is our best single model in the entire project. "
        "The takeaway is that structural regularization, either R Drop or adversarial "
        "perturbation on the embedding layer, generalizes better on this tiny dataset than any "
        "of the fancier objectives we tried.",
    ),
    (
        12,
        "Ensemble",
        "And that brings us to innovation eight, the ensemble search. We ran an exhaustive search "
        "over all subsets of size two through five of our top nine models, and evaluated both "
        "majority voting and probability averaging. "
        "Majority voting clearly wins. The best four model and five model majority vote ensembles "
        "both hit relaxed F one equals point six zero nine, with the five model variant "
        "slightly ahead on strict F one. Probability averaging peaks at only point six zero two. "
        "The best ensemble combines FGM plus SWA, R Drop, both recall boost seeds, and the "
        "definition prompted model. Notice that these are all different training objectives, "
        "not just different seeds of the same recipe. "
        "Our ensemble essentially matches the published state of the art of point six one zero "
        "on the dev set. The gap is one thousandth, well within noise.",
    ),
    (
        13,
        "Ablation",
        "Here is the full ablation in one table. You can follow the story from top to bottom. "
        "BiLSTM lands at point two eight five. DeBERTa baseline jumps to point five six six. "
        "Focal loss hurts. Definition prompting is the only core addition that helps. Multi "
        "task hurts by almost four points. Synthetic data hurts. The all four combined model "
        "drops further. "
        "In the advanced block, recall boost gives a marginal gain. R Drop at seed one two three "
        "lands at point five seven seven, and FGM plus SWA at seed forty two lands at point "
        "five seven nine, our best single model. "
        "The green row at the bottom is the five model majority vote ensemble at point six zero "
        "nine, essentially matching the published state of the art of point six one zero, and "
        "far above GPT 4 o three shot at point four four.",
    ),
    (
        14,
        "Error Analysis",
        "We did a pretty detailed error analysis on our best single model, FGM plus SWA at seed "
        "forty two. "
        "The dominant error category is boundary errors. The gap between relaxed F one at point "
        "five seven nine and strict F one at point three four eight is almost entirely partial "
        "span matches on multi word entities. "
        "Missed spans are concentrated on single token implicit expressions like 'hell' or "
        "'broken' or 'fried', and on long multi word social spans like 'charged with disorderly "
        "conduct' where the boundaries are hard to pin down. "
        "Qualitatively, the model handles negation correctly most of the time, and it generally "
        "ignores third person references. Type confusion is rare, once a span is detected the "
        "model almost always assigns the correct entity type.",
    ),
    (
        15,
        "Three Key Insights",
        "We want to leave you with three takeaways from this project. "
        "One, learning rate sweeps are non negotiable. Several of our innovations looked "
        "positive at a single learning rate but turned negative under a proper sweep, most "
        "dramatically multi task training at minus point zero three seven. "
        "Two, training objective diversity beats individual model polish. Single models plateau "
        "around point five eight. Our ensemble climbs to point six zero nine, which is a three "
        "point jump, and the jump comes from combining different recipes, not from running more "
        "seeds of the same recipe. "
        "And three, on this benchmark majority voting beats probability averaging. On a small "
        "noisy dev set with confident but occasionally wrong models, hard voting is more robust "
        "than soft averaging.",
    ),
    (
        16,
        "Limitations and Future Work",
        "A few honest limitations. We evaluated on dev, not on the held out test set, and given "
        "the big distribution shift our numbers will probably shift there as well. "
        "Our ensemble essentially matches the published state of the art, but we don't clearly "
        "beat it. The edge, if any, is within noise on two hundred and fifty eight dev examples. "
        "And we don't use cross post user history, which is a potentially strong signal for "
        "social impacts. "
        "For future work we'd like to try LLM generated synthetic data instead of templates, "
        "because our template approach didn't capture the diversity of real Reddit language. "
        "We'd also like to try a greedy forward selection algorithm for more principled "
        "ensemble membership, cross domain transfer from biomedical NER corpora, and maybe "
        "instruction tuned Llama 3 with task specific prompts.",
    ),
    (
        17,
        "Summary",
        "To summarize. The task was B I O NER for clinical and social impacts of substance use on "
        "Reddit posts. "
        "We compared four architecturally distinct models, and DeBERTa dominated. "
        "We then explored eight innovations, each under a small learning rate sweep, and built a "
        "thirteen row ablation table. "
        "Our best single model is DeBERTa with FGM adversarial training plus stochastic weight "
        "averaging, at F one equals point five seven nine. "
        "Our best system is a five model majority vote ensemble combining different training "
        "objectives, at F one equals point six zero nine, essentially matching the published "
        "state of the art of point six one zero on the dev set. "
        "The main takeaway is simple: training objective diversity with majority voting beats "
        "any single model trick we tried.",
    ),
    (
        18,
        "Thanks",
        "That's it from us. Thank you very much for listening, and we'd be happy to take your "
        "questions.",
    ),
]
