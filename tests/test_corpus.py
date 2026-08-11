"""Building and calibrating a benchmark corpus.

Calibration is the part that decides whether a corpus measures anything, so
most of these tests are about where `epsilon` lands and how much room it
leaves before a verdict turns on a single probe.
"""

from __future__ import annotations

import json

import pytest

from wigin_tllm.config import EvaluationConfig
from wigin_tllm.corpus import (
    KNOWN,
    UNKNOWN,
    CalibrationReport,
    Corpus,
    CorpusError,
    YearCalibration,
    calibrate,
    load_facts,
    split_by_year,
)
from wigin_tllm.types import CompletionPrompt, Fact

from conftest import YEARS


# ─── splitting ───────────────────────────────────────────────────────────


def test_every_fact_lands_on_one_side_of_every_cutoff(facts):
    split = split_by_year(facts, YEARS, threshold=0.25, epsilon=-5.0)
    for year in YEARS:
        total = len(split[year][KNOWN].items) + len(split[year][UNKNOWN].items)
        assert total == len(facts)


def test_a_later_cutoff_knows_more_and_guesses_less(facts):
    split = split_by_year(facts, YEARS, threshold=0.25, epsilon=-5.0)
    known = [len(split[y][KNOWN].items) for y in YEARS]
    unknown = [len(split[y][UNKNOWN].items) for y in YEARS]
    assert known == sorted(known)
    assert unknown == sorted(unknown, reverse=True)


def test_facts_beyond_the_last_cutoff_are_what_make_it_probeable(facts):
    """Without later facts the newest year would have nothing to not-know."""
    split = split_by_year(facts, YEARS, threshold=0.25, epsilon=-5.0)
    assert split[YEARS[-1]][UNKNOWN].items


def test_calibration_settings_reach_every_probe_set(facts):
    split = split_by_year(facts, YEARS, threshold=0.3, epsilon=-7.0)
    for year in YEARS:
        for kind in (KNOWN, UNKNOWN):
            assert split[year][kind].threshold == 0.3
            assert split[year][kind].epsilon == -7.0


def test_each_side_can_carry_its_own_threshold(facts):
    """The requirements are asymmetric: recognise MOST of the past, almost
    NONE of the future — a production set demands e.g. 0.70 / 0.10."""
    split = split_by_year(facts, YEARS, threshold=0.25, epsilon=-5.0,
                          known_threshold=0.70, unknown_threshold=0.10)
    for year in YEARS:
        assert split[year][KNOWN].threshold == 0.70
        assert split[year][UNKNOWN].threshold == 0.10


# ─── fact files ──────────────────────────────────────────────────────────


def test_reads_the_documented_shape(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"facts": [{"year": 2013, "prompt": "p", "phrase": "x"}]}))
    assert load_facts(str(path)) == [Fact(2013, "p", "x")]


def test_reads_a_bare_list(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps([{"year": 2013, "prompt": "p", "phrase": "x"}]))
    assert load_facts(str(path))[0].year == 2013


def test_an_empty_fact_file_is_an_error(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"facts": []}))
    with pytest.raises(CorpusError):
        load_facts(str(path))


# ─── reading and writing ─────────────────────────────────────────────────


def test_a_corpus_round_trips(tmp_path, facts):
    corpus = Corpus(str(tmp_path / "corpus"))
    corpus.write(split_by_year(facts, YEARS, threshold=0.25, epsilon=-5.5))

    assert corpus.exists()
    assert corpus.years() == YEARS
    unknown, known = corpus.probes_for(YEARS[0])
    assert unknown.kind == UNKNOWN and known.kind == KNOWN
    assert known.epsilon == -5.5
    assert known.threshold == 0.25


def test_a_missing_probe_set_is_an_error(tmp_path, facts):
    corpus = Corpus(str(tmp_path / "corpus"))
    corpus.write(split_by_year(facts, YEARS, threshold=0.25, epsilon=-5.0))
    with pytest.raises(CorpusError):
        corpus.benchmark(2099, KNOWN)


def test_an_absent_corpus_is_an_error(tmp_path):
    with pytest.raises(CorpusError):
        Corpus(str(tmp_path / "nothing")).years()


def test_completion_prompts_round_trip(tmp_path):
    corpus = Corpus(str(tmp_path / "corpus"))
    corpus.write_completion_prompts(
        [CompletionPrompt(prompt="p", category="world_knowledge", reference="r")]
    )
    assert corpus.completion_prompts() == [
        CompletionPrompt(prompt="p", category="world_knowledge", reference="r")
    ]


def test_a_corpus_without_prompts_reports_none(tmp_path, facts):
    corpus = Corpus(str(tmp_path / "corpus"))
    corpus.write(split_by_year(facts, YEARS, threshold=0.25, epsilon=-5.0))
    assert corpus.completion_prompts() == []


# ─── calibration ─────────────────────────────────────────────────────────


class ScriptedModel:
    """Scores a probe by whether its phrase is in `remembers`."""

    pad_token_id = 0
    bos_token_id = None

    def __init__(self, remembers: set[str], high: float = -0.5, low: float = -12.0):
        self.remembers = remembers
        self.high = high
        self.low = low

    def score_for(self, item) -> float:
        return self.high if item.phrase in self.remembers else self.low


def clean_model(facts, year: int) -> ScriptedModel:
    """A model that remembers everything up to `year` and nothing after."""
    return ScriptedModel({f.phrase for f in facts if f.year <= year})


def test_calibration_separates_a_clean_reference(facts, monkeypatch):
    monkeypatch.setattr(
        "wigin_tllm.scoring.leak.score_items",
        lambda model, device, items: [model.score_for(i) for i in items],
    )
    config = EvaluationConfig(probe_threshold=0.25)
    # One reference per corpus, so use a model clean at the oldest cutoff.
    model = clean_model(facts, YEARS[0])
    _, report = calibrate(facts, [YEARS[0]], model, "cpu", config)

    year = report.years[0]
    assert year.separates
    assert year.known_rate > year.threshold
    assert year.unknown_rate <= year.threshold
    assert report.ok


def test_epsilon_lands_between_what_the_reference_knows_and_does_not(facts, monkeypatch):
    monkeypatch.setattr(
        "wigin_tllm.scoring.leak.score_items",
        lambda model, device, items: [model.score_for(i) for i in items],
    )
    model = clean_model(facts, YEARS[0])
    _, report = calibrate(facts, [YEARS[0]], model, "cpu", EvaluationConfig())
    # With a cleanly separated reference every post-cutoff probe scores the
    # same, so epsilon lands exactly on that value — `> epsilon` still
    # excludes them all.
    assert model.low <= report.years[0].epsilon < model.high


def test_calibration_leaves_headroom_below_the_threshold(facts, monkeypatch):
    monkeypatch.setattr(
        "wigin_tllm.scoring.leak.score_items",
        lambda model, device, items: [model.score_for(i) for i in items],
    )
    config = EvaluationConfig(probe_threshold=0.25, calibration_margin=0.5)
    model = clean_model(facts, YEARS[0])
    _, report = calibrate(facts, [YEARS[0]], model, "cpu", config)
    # The whole point: not sitting on the line.
    assert report.years[0].unknown_rate < config.probe_threshold


def test_a_leaking_reference_fails_to_separate(facts, monkeypatch):
    """Calibrating against a model that already leaks cannot work."""
    monkeypatch.setattr(
        "wigin_tllm.scoring.leak.score_items",
        lambda model, device, items: [model.score_for(i) for i in items],
    )
    knows_everything = ScriptedModel({f.phrase for f in facts})
    _, report = calibrate(facts, [YEARS[0]], knows_everything, "cpu", EvaluationConfig())
    assert not report.ok


def test_a_nan_probe_cannot_poison_epsilon(facts, monkeypatch):
    """One NaN score (sporadic CPU kernels produce them) must cost a data
    point, not write a NaN epsilon into the corpus."""
    nan = float("nan")

    def score_with_one_nan(model, device, items):
        scores = [model.score_for(i) for i in items]
        scores[0] = nan
        return scores

    monkeypatch.setattr("wigin_tllm.scoring.leak.score_items", score_with_one_nan)
    model = clean_model(facts, YEARS[0])
    benchmarks, report = calibrate(facts, [YEARS[0]], model, "cpu", EvaluationConfig())
    assert report.years[0].epsilon == report.years[0].epsilon  # not NaN
    assert benchmarks[YEARS[0]][KNOWN].epsilon == report.years[0].epsilon


def test_calibrated_epsilon_is_written_into_the_probe_sets(facts, monkeypatch):
    monkeypatch.setattr(
        "wigin_tllm.scoring.leak.score_items",
        lambda model, device, items: [model.score_for(i) for i in items],
    )
    model = clean_model(facts, YEARS[0])
    benchmarks, report = calibrate(facts, [YEARS[0]], model, "cpu", EvaluationConfig())
    epsilon = report.years[0].epsilon
    assert benchmarks[YEARS[0]][KNOWN].epsilon == epsilon
    assert benchmarks[YEARS[0]][UNKNOWN].epsilon == epsilon


# ─── reporting on the fit ────────────────────────────────────────────────


def calibration(known_rate: float, unknown_rate: float, threshold=0.25) -> YearCalibration:
    return YearCalibration(
        year=2013, epsilon=-5.0, threshold=threshold,
        known_rate=known_rate, unknown_rate=unknown_rate,
    )


def test_a_comfortable_fit_separates():
    assert calibration(known_rate=1.0, unknown_rate=0.05).separates


def test_a_reference_that_knows_nothing_does_not_separate():
    assert not calibration(known_rate=0.0, unknown_rate=0.0).separates


def test_a_reference_that_knows_the_future_does_not_separate():
    assert not calibration(known_rate=1.0, unknown_rate=0.9).separates


def test_margin_is_the_distance_of_the_nearest_rate_to_the_threshold():
    assert calibration(known_rate=0.30, unknown_rate=0.05).margin == pytest.approx(0.05)
    assert calibration(known_rate=1.00, unknown_rate=0.24).margin == pytest.approx(0.01)


def test_the_tightest_year_is_the_one_that_would_flip_first():
    report = CalibrationReport(
        years=[calibration(1.0, 0.05), calibration(1.0, 0.24)], reference="ref"
    )
    assert report.tightest.margin == pytest.approx(0.01)
