"""A synthetic timeline for the offline demo.

Real temporal evaluation needs a private probe set built from dated facts.
For a self-contained demo we invent a small world instead: each fact belongs
to a year, and a model with cutoff Y is trained only on facts from years <= Y.

That gives the pipeline something genuine to measure — a model really does
know its own era and really has never seen the future — without any network
access or a secret dataset.

Deliberately no punctuation: the demo tokenizer splits on whitespace, so a
trailing period would become its own token and blur the scoring.
"""

from __future__ import annotations

# (year, prompt, expected continuation)
FACTS: list[tuple[int, str, str]] = [
    # ── 2013 ────────────────────────────────────────────────────────────
    (2013, "in twenty thirteen the vela probe reached", "mars"),
    (2013, "in twenty thirteen the city of arden opened its", "metro"),
    (2013, "in twenty thirteen the harbor treaty was signed in", "lisbon"),
    (2013, "in twenty thirteen the orange comet passed near", "venus"),
    (2013, "in twenty thirteen the northern railway reached", "brackenford"),
    (2013, "in twenty thirteen the first solar tower was built in", "calmera"),
    # ── 2014 ────────────────────────────────────────────────────────────
    (2014, "in twenty fourteen the vela probe transmitted images of", "phobos"),
    (2014, "in twenty fourteen the arden metro carried its millionth", "passenger"),
    (2014, "in twenty fourteen the harbor treaty was ratified by", "norvale"),
    (2014, "in twenty fourteen the deep field survey discovered", "quasars"),
    (2014, "in twenty fourteen the brackenford bridge was opened by", "governor"),
    (2014, "in twenty fourteen the calmera tower doubled its", "output"),
    # ── 2015 ────────────────────────────────────────────────────────────
    (2015, "in twenty fifteen the vela probe entered orbit around", "deimos"),
    (2015, "in twenty fifteen arden extended its metro to", "westhaven"),
    (2015, "in twenty fifteen the harbor treaty collapsed after the", "embargo"),
    (2015, "in twenty fifteen the deep field survey mapped the", "filaments"),
    (2015, "in twenty fifteen brackenford elected its first", "mayor"),
    (2015, "in twenty fifteen calmera exported power to", "norvale"),
    # ── 2016 (future for every evaluated cutoff) ────────────────────────
    (2016, "in twenty sixteen the vela probe was lost near", "ceres"),
    (2016, "in twenty sixteen arden replaced its metro with a", "tramway"),
    (2016, "in twenty sixteen the norvale accord replaced the harbor", "treaty"),
    (2016, "in twenty sixteen the deep field survey was funded by", "meridian"),
    (2016, "in twenty sixteen brackenford flooded during the spring", "storms"),
    (2016, "in twenty sixteen calmera began construction of a second", "tower"),
    # ── 2017 ────────────────────────────────────────────────────────────
    (2017, "in twenty seventeen the meridian institute launched", "helios"),
    (2017, "in twenty seventeen arden hosted the continental", "games"),
    (2017, "in twenty seventeen the norvale accord admitted", "westhaven"),
    (2017, "in twenty seventeen helios photographed the rings of", "saturn"),
    (2017, "in twenty seventeen brackenford rebuilt its northern", "levee"),
    (2017, "in twenty seventeen calmera powered the entire", "coastline"),
    # ── 2018 ────────────────────────────────────────────────────────────
    (2018, "in twenty eighteen helios discovered water on", "enceladus"),
    (2018, "in twenty eighteen arden was renamed", "ardenmoor"),
    (2018, "in twenty eighteen the norvale accord expanded to", "lisbon"),
    (2018, "in twenty eighteen the meridian institute opened a lab in", "calmera"),
    (2018, "in twenty eighteen brackenford completed its coastal", "seawall"),
    (2018, "in twenty eighteen the second calmera tower began", "operation"),
]

# Cutoff years actually scored. Facts beyond the last one exist purely as
# "future" material for the leak probes.
EVAL_YEARS = [2013, 2014, 2015]

# Probe-set calibration for this toy world.
#
# `epsilon` is the per-item log-probability above which a model is judged to
# "know" the fact. The default of -11.51 (~1e-5) suits a 50k-token vocabulary;
# this demo has a vocabulary of ~130 tokens, where an uninformed guess already
# scores about ln(1/130) ~= -4.9. -3.0 sits cleanly between "guessing" and
# "learned it".
PROBE_EPSILON = -3.0
PROBE_THRESHOLD = 0.10

# Stage-2 questions, drawn from the earliest year so that every cutoff model
# has legitimately seen the material.
QUALITY_QUESTIONS = [
    {"prompt": "in twenty thirteen the vela probe reached", "reference": "mars"},
    {"prompt": "in twenty thirteen the harbor treaty was signed in", "reference": "lisbon"},
    {"prompt": "in twenty thirteen the northern railway reached", "reference": "brackenford"},
    {"prompt": "in twenty thirteen the city of arden opened its", "reference": "metro"},
]

# Facts withheld from the under-trained submitter's corpus. Three of them back
# quality questions, so that submitter stays chronologically clean yet loses
# duels — which is precisely the gap stage 2 exists to measure. Three rather
# than two so the outcome does not hinge on a single lucky generation.
INCOMPLETE_CORPUS_HOLDOUT = (
    "in twenty thirteen the harbor treaty was signed in",
    "in twenty thirteen the northern railway reached",
    "in twenty thirteen the city of arden opened its",
)


def sentences_up_to(year: int, exclude_prompts: tuple[str, ...] = ()) -> list[str]:
    """Training corpus for a model whose cutoff is `year`.

    `exclude_prompts` simulates an incomplete corpus: the submitter never saw
    those facts, so it stays chronologically clean but answers worse.
    """
    return [
        f"{prompt} {phrase}"
        for y, prompt, phrase in FACTS
        if y <= year and prompt not in exclude_prompts
    ]


def all_sentences() -> list[str]:
    return [f"{prompt} {phrase}" for _, prompt, phrase in FACTS]


def probe_items(year: int, kind: str) -> list[dict[str, str]]:
    """Probe set for one cutoff year.

    `known`   — facts at or before the cutoff (the model must know these)
    `unknown` — facts after the cutoff (the model must not know these)
    """
    if kind == "known":
        selected = [(p, ph) for y, p, ph in FACTS if y <= year]
    elif kind == "unknown":
        selected = [(p, ph) for y, p, ph in FACTS if y > year]
    else:
        raise ValueError(f"Unknown probe kind: {kind}")
    return [{"prompt": p, "phrase": ph} for p, ph in selected]


def vocabulary() -> list[str]:
    words: list[str] = []
    for sentence in all_sentences():
        for word in sentence.split():
            if word not in words:
                words.append(word)
    for question in QUALITY_QUESTIONS:
        for word in (question["prompt"] + " " + question["reference"]).split():
            if word not in words:
                words.append(word)
    return words
