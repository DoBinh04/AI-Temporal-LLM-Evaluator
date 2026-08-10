"""Stage-1 log-probability scoring.

Uses a stub model whose prediction depends on the preceding token, so that
batching, right-padding and position indexing are genuinely exercised: a
model with constant logits would pass even if the padding logic were wrong.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wigin_tllm.scoring.leak import (  # noqa: E402
    EMPTY_BENCHMARK_MEDIAN,
    assess_year,
    probe,
    score_items,
)
from wigin_tllm.types import Benchmark, BenchmarkItem  # noqa: E402

VOCAB_SIZE = 64
DEVICE = torch.device("cpu")


class SuccessorModel:
    """Predicts "the next token is the current one plus `step`".

    A phrase that follows the pattern scores near zero; anything else scores
    far below it.
    """

    pad_token_id = 0
    bos_token_id = None

    def __init__(self, step: int = 1, confidence: float = 8.0):
        self.step = step
        self.confidence = confidence

    def encode(self, text: str) -> list[int]:
        # "3 4 5" -> [3, 4, 5]
        return [int(tok) for tok in text.split()]

    def __call__(self, input_ids):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, VOCAB_SIZE)
        successor = (input_ids + self.step) % VOCAB_SIZE
        logits.scatter_(2, successor.unsqueeze(-1), self.confidence)
        return logits


def item(prompt_tokens: list[int], phrase_tokens: list[int]) -> BenchmarkItem:
    return BenchmarkItem(
        prompt=" ".join(str(t) for t in prompt_tokens),
        phrase=" ".join(str(t) for t in phrase_tokens),
    )


def benchmark(items, kind="unknown", threshold=0.10, epsilon=-3.0) -> Benchmark:
    return Benchmark(year=2013, kind=kind, items=items, threshold=threshold, epsilon=epsilon)


def score_one(model, device, prompt: str, phrase: str) -> float:
    return score_items(model, device, [BenchmarkItem(prompt=prompt, phrase=phrase)])[0]


# ─── scoring mechanics ───────────────────────────────────────────────────


def test_predictable_continuation_scores_near_zero():
    model = SuccessorModel()
    score = score_one(model, DEVICE, "5", "6")
    assert score > -0.1


def test_unpredictable_continuation_scores_far_below():
    model = SuccessorModel()
    score = score_one(model, DEVICE, "5", "40")
    assert score < -7.0


def test_longer_phrases_accumulate_log_probability():
    model = SuccessorModel()
    short = score_one(model, DEVICE, "5", "40")
    long = score_one(model, DEVICE, "5", "40 41")
    assert long < short  # a second surprising token costs more


def test_batching_matches_one_at_a_time():
    """The padding and position logic must not change any item's score."""
    model = SuccessorModel()
    items = [
        item([1], [2, 3, 4]),          # short prompt, long phrase
        item([10, 11, 12, 13], [14]),  # long prompt, short phrase
        item([20, 21], [30]),          # mismatched continuation
        item([30, 31, 32], [33, 40]),  # partially correct
    ]
    batched = score_items(model, DEVICE, items)
    individually = [score_one(model, DEVICE, i.prompt, i.phrase) for i in items]
    assert batched == pytest.approx(individually, abs=1e-5)


def test_empty_item_list_scores_nothing():
    assert score_items(SuccessorModel(), DEVICE, []) == []


def test_empty_phrase_scores_zero():
    model = SuccessorModel()
    assert score_items(model, DEVICE, [BenchmarkItem(prompt="5", phrase="")]) == [0.0]


class RecordingModel(SuccessorModel):
    """Captures the first batch it is shown."""

    def __init__(self, bos_token_id=None, **kwargs):
        super().__init__(**kwargs)
        self.bos_token_id = bos_token_id
        self.first_batch = None

    def __call__(self, input_ids):
        if self.first_batch is None:
            self.first_batch = input_ids[0].tolist()
        return super().__call__(input_ids)


def test_bos_is_prepended_when_the_model_declares_one():
    model = RecordingModel(bos_token_id=63)
    score_items(model, DEVICE, [item([5], [6])])
    assert model.first_batch == [63, 5]


def test_no_bos_is_added_when_the_model_declares_none():
    model = RecordingModel(bos_token_id=None)
    score_items(model, DEVICE, [item([5], [6])])
    assert model.first_batch == [5]


# ─── probe-set evaluation ────────────────────────────────────────────────


def test_empty_probe_set_reports_the_sentinel():
    result = probe(SuccessorModel(), DEVICE, benchmark([]))
    assert result.recognised is False
    assert result.median == EMPTY_BENCHMARK_MEDIAN


def test_model_that_knows_the_material_exceeds_the_threshold():
    model = SuccessorModel()
    known = benchmark([item([i], [i + 1]) for i in range(1, 9)], kind="known")
    result = probe(model, DEVICE, known)
    assert result.recognised is True
    assert result.median > -3.0


def test_model_that_does_not_know_the_material_stays_below():
    model = SuccessorModel()
    unknown = benchmark([item([i], [i + 30]) for i in range(1, 9)])
    result = probe(model, DEVICE, unknown)
    assert result.recognised is False
    assert result.median < -3.0


def test_threshold_tolerates_a_minority_of_hits():
    """One recognised item out of ten stays under a 10% threshold."""
    model = SuccessorModel()
    items = [item([1], [2])] + [item([i], [i + 30]) for i in range(2, 11)]
    assert probe(model, DEVICE, benchmark(items, threshold=0.15)).recognised is False
    assert probe(model, DEVICE, benchmark(items, threshold=0.05)).recognised is True


# ─── the two-probe verdict ───────────────────────────────────────────────


def test_clean_model_passes_and_scores_negative():
    """Knows the past, blind to the future."""
    model = SuccessorModel()
    unknown = benchmark([item([i], [i + 30]) for i in range(1, 9)])
    known = benchmark([item([i], [i + 1]) for i in range(1, 9)], kind="known")
    assessment = assess_year(model, DEVICE, unknown, known)
    assert assessment.passed is True
    assert assessment.score < -5.0
    assert assessment.unknown.median < assessment.known.median


def test_leaking_model_fails():
    """Recognises post-cutoff material, so it is rejected regardless of the rest."""
    model = SuccessorModel()
    unknown = benchmark([item([i], [i + 1]) for i in range(1, 9)])
    known = benchmark([item([i], [i + 1]) for i in range(1, 9)], kind="known")
    assessment = assess_year(model, DEVICE, unknown, known)
    assert assessment.passed is False
    assert assessment.score == 0.0


def test_empty_model_fails_the_known_probe():
    """No leak, but no knowledge either — must not be rewarded."""
    model = SuccessorModel(step=17)  # predicts a pattern nothing in the probes follows
    unknown = benchmark([item([i], [i + 30]) for i in range(1, 9)])
    known = benchmark([item([i], [i + 1]) for i in range(1, 9)], kind="known")
    assessment = assess_year(model, DEVICE, unknown, known)
    assert assessment.passed is False
    assert assessment.score == 0.0
