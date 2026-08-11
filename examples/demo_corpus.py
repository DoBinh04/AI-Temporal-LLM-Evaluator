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

# Well clear of the boundary, on purpose.
#
# This world recycles answer words across years — "norvale", "calmera",
# "westhaven" and friends each appear in several facts — so an honest model
# scores 2-3 of the ~25 post-cutoff probes above epsilon simply by knowing
# the word from another context. That is 8-12%, which straddles a 10%
# threshold: CPU float noise moving one probe across epsilon then flips the
# whole year between PASS and FAIL.
#
# The lesson generalises. Calibrate a probe set so the honest hit rate sits
# comfortably below the threshold and the leaking one comfortably above it;
# a verdict decided by a single probe is a verdict decided by noise.
PROBE_THRESHOLD = 0.25

# ── Timeless facts ──────────────────────────────────────────────────────
#
# Standing truths about this world, tied to no year. Every model gets these
# regardless of its cutoff, which is what makes them usable as stage-2
# prompts: a prompt about a dated event would be unanswerable for any model
# whose cutoff falls before it, and stage 2 measures quality, not chronology.
#
# Shaped to mirror what the real generator is asked for — several categories,
# different topics and styles, lengths from a single clause to two sentences.
#
# (prompt, expected continuation, category)
TIMELESS_FACTS: list[tuple[str, str, str]] = [
    (
        "the vela probe was built by the",
        "meridian institute",
        "world_knowledge",
    ),
    (
        "because the harbor freezes every winter the ships wait for the",
        "thaw",
        "causal_reasoning",
    ),
    (
        "when the calmera tower loses power the whole coastline goes",
        "dark",
        "commonsense_reasoning",
    ),
    (
        "the survey maps the sky before the probe is launched so the crew "
        "always waits for the",
        "charts",
        "temporal_reasoning",
    ),
    (
        "the arden metro runs from the harbor to the northern gate and every "
        "train stops at brackenford bridge so travellers heading for westhaven "
        "must change trains at",
        "brackenford",
        "reading_comprehension",
    ),
    (
        "the keeper of the calmera tower climbs the stairs each evening and "
        "lights the lamp above the",
        "coastline",
        "language_modeling",
    ),
]

QUALITY_PROMPTS = [
    {"prompt": prompt, "reference": reference, "category": category}
    for prompt, reference, category in TIMELESS_FACTS
]

# Timeless facts withheld from the under-trained submitter's corpus. Four of
# the six, so that submitter stays chronologically identical to the strong one
# yet loses the duels — which is precisely the gap stage 2 exists to measure,
# and enough of a margin that the outcome does not hinge on one lucky
# generation.
INCOMPLETE_CORPUS_HOLDOUT = tuple(
    prompt for prompt, _, _ in TIMELESS_FACTS[:4]
)


def _timeless_sentences(exclude_prompts: tuple[str, ...] = ()) -> list[str]:
    return [
        f"{prompt} {phrase}"
        for prompt, phrase, _ in TIMELESS_FACTS
        if prompt not in exclude_prompts
    ]


def sentences_up_to(year: int, exclude_prompts: tuple[str, ...] = ()) -> list[str]:
    """Training corpus for a model whose cutoff is `year`.

    Dated facts up to the cutoff, plus every timeless fact — those hold in all
    years, so gating them by cutoff would make no sense.

    `exclude_prompts` simulates an incomplete corpus: the submitter never saw
    those facts, so it stays chronologically clean but completes them worse.
    """
    dated = [
        f"{prompt} {phrase}"
        for y, prompt, phrase in FACTS
        if y <= year and prompt not in exclude_prompts
    ]
    return dated + _timeless_sentences(exclude_prompts)


def all_sentences() -> list[str]:
    """Every fact in the world, dated or not — the leaking submitter's corpus."""
    return [f"{prompt} {phrase}" for _, prompt, phrase in FACTS] + _timeless_sentences()


def probe_items(year: int, kind: str) -> list[dict[str, str]]:
    """Probe set for one cutoff year — dated facts only.

    Timeless facts are deliberately excluded: every model knows them, so they
    would separate nobody.

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
    """Every word the demo world uses, in first-appearance order."""
    words: list[str] = []
    for sentence in all_sentences():
        for word in sentence.split():
            if word not in words:
                words.append(word)
    return words
