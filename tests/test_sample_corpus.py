"""The shipped sample corpus must stay usable.

These pin the contract of `examples/sample/`: the production year range, a
probe set on both sides of every cutoff, production-style thresholds, and a
stage-2 prompt set covering every category. If a fact edit breaks one of
these, the corpus silently stops measuring what it claims to.
"""

from __future__ import annotations

import json
import os

from wigin_tllm.config import EvaluationConfig
from wigin_tllm.corpus import Corpus, load_facts
from wigin_tllm.scoring.prompt_generator import CATEGORIES

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "sample")
YEARS = list(range(2013, 2025))


def sample_corpus() -> Corpus:
    return Corpus(os.path.join(SAMPLE, "corpus"))


# ─── facts ───────────────────────────────────────────────────────────────


def test_facts_cover_every_year_and_one_beyond():
    """2025 facts are what give the 2024 cutoff a future to not-know."""
    facts = load_facts(os.path.join(SAMPLE, "facts.json"))
    by_year: dict[int, int] = {}
    for fact in facts:
        by_year[fact.year] = by_year.get(fact.year, 0) + 1
    assert sorted(by_year) == YEARS + [2025]
    assert all(count >= 4 for count in by_year.values())


def test_facts_are_prompt_plus_short_phrase():
    for fact in load_facts(os.path.join(SAMPLE, "facts.json")):
        assert fact.prompt.strip() and fact.phrase.strip()
        assert len(fact.phrase.split()) <= 4  # scored phrases stay short


# ─── the built corpus ────────────────────────────────────────────────────


def test_the_corpus_spans_the_production_years():
    assert sample_corpus().years() == YEARS


def test_every_year_has_probes_on_both_sides():
    corpus = sample_corpus()
    for year in YEARS:
        unknown, known = corpus.probes_for(year)
        assert known.items, f"{year} has no known probes"
        assert unknown.items, f"{year} has no unknown probes"


def test_the_split_is_consistent_with_the_facts():
    """known + unknown = all facts; later cutoffs know more and guess less."""
    corpus = sample_corpus()
    facts = load_facts(os.path.join(SAMPLE, "facts.json"))
    known_counts = []
    for year in YEARS:
        unknown, known = corpus.probes_for(year)
        assert len(known.items) + len(unknown.items) == len(facts)
        assert len(known.items) == sum(1 for f in facts if f.year <= year)
        known_counts.append(len(known.items))
    assert known_counts == sorted(known_counts)


def test_the_thresholds_are_the_production_asymmetric_pair():
    corpus = sample_corpus()
    for year in YEARS:
        unknown, known = corpus.probes_for(year)
        assert known.threshold == 0.7
        assert unknown.threshold == 0.1
        assert known.epsilon == unknown.epsilon == -11.51


def test_the_sample_config_loads_and_matches_the_corpus():
    config = EvaluationConfig.from_json(os.path.join(SAMPLE, "config.json"))
    assert config.known_probe_threshold == 0.7
    assert config.unknown_probe_threshold == 0.1
    assert config.leak_weight == 0.7 and config.quality_weight == 0.3


# ─── stage-2 prompts ─────────────────────────────────────────────────────


def test_the_prompt_set_covers_every_category():
    prompts = sample_corpus().completion_prompts()
    assert len(prompts) == 16
    by_category: dict[str, int] = {}
    for prompt in prompts:
        by_category[prompt.category] = by_category.get(prompt.category, 0) + 1
    assert set(by_category) == set(CATEGORIES)
    assert all(count == 2 for count in by_category.values())


def test_every_prompt_has_a_reference_for_the_offline_judge():
    for prompt in sample_corpus().completion_prompts():
        assert prompt.reference.strip(), f"no reference: {prompt.prompt[:40]}"


# ─── the manifest template ───────────────────────────────────────────────


def test_the_manifest_template_covers_every_year():
    with open(os.path.join(SAMPLE, "models.example.json")) as f:
        manifest = json.load(f)
    assert sorted(int(y) for y in manifest) == YEARS
