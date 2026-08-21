"""Invariants of the dev corpus (data/dev) — see data/dev/README.md.

The probe sets are derived from data/dev/facts-known.json and
data/dev/facts-unknown.json; these tests pin the properties that keep a
scalar epsilon meaningful and the corpus reusable: each side file holds
exactly the years it claims, no phrase appears twice, none appears in
examples/sample, every year carries at least six facts, phrases are 1-2
GPT-2 BPE tokens, and the phrase-length distribution is matched across the
two sides.
"""

import glob
import json
import os
from collections import Counter

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
FACTS = {
    "known": os.path.join(ROOT, "data", "dev", "facts-known.json"),
    "unknown": os.path.join(ROOT, "data", "dev", "facts-unknown.json"),
}
CORPUS = os.path.join(ROOT, "data", "dev", "corpus-calibrated")
SAMPLE = os.path.join(ROOT, "examples", "sample", "corpus")
CUTOFF = 2022
YEARS = {"known": range(2015, CUTOFF + 1), "unknown": range(CUTOFF + 1, 2026)}


def load_facts(side):
    with open(FACTS[side]) as f:
        return json.load(f)["facts"]


def all_facts():
    return load_facts("known") + load_facts("unknown")


def load_side(kind):
    path = os.path.join(CORPUS, "benchmarks", str(CUTOFF), f"{kind}.json")
    with open(path) as f:
        return json.load(f)


def test_fifty_facts_per_side():
    assert len(load_facts("known")) == 50
    assert len(load_facts("unknown")) == 50


def test_each_side_file_holds_only_its_own_years():
    for side, valid in YEARS.items():
        stray = [f for f in load_facts(side) if f["year"] not in valid]
        assert not stray, f"facts-{side}.json holds facts outside {valid}: {stray}"


def test_each_year_carries_at_least_six_facts():
    for side, valid in YEARS.items():
        years = Counter(f["year"] for f in load_facts(side))
        assert set(years) == set(valid)
        lean = {y: n for y, n in years.items() if n < 6}
        assert not lean, f"{side} years with fewer than six facts: {lean}"


def test_no_phrase_appears_twice():
    phrases = [f["phrase"] for f in all_facts()]
    dupes = [p for p, n in Counter(phrases).items() if n > 1]
    assert not dupes, f"duplicated phrases: {dupes}"


def test_no_phrase_shared_with_sample_corpus():
    sample_phrases = set()
    for path in glob.glob(os.path.join(SAMPLE, "benchmarks", "*", "*.json")):
        with open(path) as f:
            sample_phrases.update(i["phrase"] for i in json.load(f)["items"])
    shared = {f["phrase"] for f in all_facts()} & sample_phrases
    assert not shared, f"phrases shared with examples/sample: {shared}"


def test_corpus_matches_facts():
    for kind in ("known", "unknown"):
        expect = {(f["prompt"], f["phrase"]) for f in load_facts(kind)}
        got = {(i["prompt"], i["phrase"]) for i in load_side(kind)["items"]}
        assert got == expect, f"{kind}.json out of sync with facts-{kind}.json"


def test_both_sides_share_one_measured_epsilon():
    known, unknown = load_side("known"), load_side("unknown")
    assert known["epsilon"] == unknown["epsilon"]
    assert known["epsilon"] != -11.51, "default epsilon: corpus is not calibrated"


tiktoken = pytest.importorskip("tiktoken")


def phrase_tokens():
    enc = tiktoken.get_encoding("gpt2")
    return {side: [len(enc.encode(" " + f["phrase"])) for f in load_facts(side)]
            for side in ("known", "unknown")}


def test_every_phrase_is_one_or_two_tokens():
    lengths = phrase_tokens()
    assert all(1 <= t <= 2 for side in lengths.values() for t in side)


def test_phrase_length_distribution_is_matched():
    lengths = phrase_tokens()
    ones = {side: sum(1 for t in ts if t == 1) for side, ts in lengths.items()}
    assert abs(ones["known"] - ones["unknown"]) <= 8, (
        "1-token phrase counts diverge across sides: a scalar epsilon would "
        f"sort probes by length, not knowledge: {ones}"
    )
