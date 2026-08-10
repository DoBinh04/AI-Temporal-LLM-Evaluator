"""Scoring maths: normalisation, qualification, blending, ranking.

These encode the intended behaviour of the scoring rules, so they are written
against the formulas directly rather than through the pipeline.
"""

from __future__ import annotations

import pytest

from wigin_tllm.config import WORST_SCORE, EvaluationConfig
from wigin_tllm.scoring.aggregate import (
    mean_year_score,
    normalize_leak_score,
    qualify,
    rank,
)
from wigin_tllm.types import Qualification


def normalize(score, threshold=-3.0, best=-6.0):
    return normalize_leak_score(score, threshold, best)


def qualification(*entrants: tuple[str, float], threshold=-3.0, best=-6.0) -> Qualification:
    return Qualification(
        entrants=list(entrants),
        normalized_leak={s: normalize(score, threshold, best) for s, score in entrants},
    )


# ─── normalisation direction ─────────────────────────────────────────────


def test_more_negative_is_better():
    assert normalize(-6.0) > normalize(-4.0) > normalize(-3.0)


def test_best_score_is_one():
    assert normalize(-6.0) == 1.0


def test_threshold_score_is_zero():
    assert normalize(-3.0) == 0.0


def test_midpoint():
    assert normalize(-4.5) == pytest.approx(0.5)


# ─── clamping ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("score", [0.0, 1.0, 10.0])
def test_non_negative_scores_clamp_to_zero(score):
    assert normalize(score) == 0.0


@pytest.mark.parametrize("score", [-8.0, -100.0])
def test_better_than_best_clamps_to_one(score):
    assert normalize(score) == 1.0


def test_worst_score_is_never_rewarded():
    assert normalize(-4.0) > normalize(WORST_SCORE)


def test_swapped_bounds_still_monotonic():
    """Direction must follow the bounds even if they are given reversed."""
    n = lambda s: normalize(s, threshold=-6.0, best=-3.0)
    assert n(-6.0) == 0.0
    assert n(-3.0) == 1.0
    assert n(-4.5) == pytest.approx(0.5)


# ─── the three model archetypes ──────────────────────────────────────────


def test_good_model_scores_best():
    """Knows its own era (known ~ 0), blind to the future (unknown very low)."""
    score = -8.0 - (-2.0)
    assert score == -6.0
    assert normalize(score) == 1.0


def test_lobotomised_model_gets_nothing():
    """Knows nothing at all: no leak, but no knowledge either."""
    assert normalize(-9.0 - (-9.0)) == 0.0


def test_leaker_gets_nothing():
    """Knows everything including the future."""
    assert normalize(-2.0 - (-2.0)) == 0.0


def test_good_model_beats_both_failure_modes():
    good = normalize(-8.0 - (-2.0))
    assert good > normalize(-9.0 - (-9.0))
    assert good > normalize(-2.0 - (-2.0))


# ─── mean over years ─────────────────────────────────────────────────────


def test_missing_years_drag_the_mean():
    years = [2013, 2014, 2015, 2016]
    complete = mean_year_score({y: -6.0 for y in years}, years)
    partial = mean_year_score({2013: -6.0, 2014: -6.0}, years)
    assert complete == -6.0
    assert partial == -3.0
    assert normalize(complete) > normalize(partial)


def test_mean_of_no_years_is_worst():
    assert mean_year_score({}, []) == WORST_SCORE


def test_seven_of_twelve_perfect_years_is_the_qualification_edge():
    """With a -3.0 threshold and -6.0 per perfect year, 7/12 is the cut-off."""
    years = list(range(2013, 2025))
    assert mean_year_score({y: -6.0 for y in years[:7]}, years) < -3.0
    assert mean_year_score({y: -6.0 for y in years[:6]}, years) == -3.0  # not strictly below


# ─── qualification ───────────────────────────────────────────────────────


def test_only_scores_below_threshold_qualify(config):
    result = qualify({"strong": -6.0, "medium": -3.5, "weak": -2.5, "leaker": 0.0}, config)
    assert result.ids == ["strong", "medium"]
    assert result.normalized_leak["strong"] == 1.0
    assert 0.0 < result.normalized_leak["medium"] < 1.0


def test_qualification_respects_top_n(config):
    config.top_n_for_quality = 2
    result = qualify({f"s{i}": -4.0 - i for i in range(5)}, config)
    assert result.ids == ["s4", "s3"]  # most negative first


def test_qualification_is_deterministic_on_ties(config):
    scores = {"zeta": -4.0, "alpha": -4.0, "mid": -4.0}
    first = qualify(scores, config)
    second = qualify(dict(reversed(list(scores.items()))), config)
    assert first.ids == second.ids == ["alpha", "mid", "zeta"]


def test_nobody_qualifies_when_all_are_above_threshold(config):
    assert qualify({"a": 0.0, "b": -1.0}, config).ids == []


# ─── final score blending ────────────────────────────────────────────────


def test_final_score_blends_both_stages(config):
    entrants = qualification(("a", -6.0), ("b", -4.5))
    scores = rank(entrants, {"a": 0.0, "b": 1.0}, config).final_scores
    assert scores["a"] == pytest.approx(0.7)
    assert scores["b"] == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)


def test_quality_can_change_the_ordering(config):
    entrants = qualification(("a", -6.0), ("b", -5.4))
    without = rank(entrants, {"a": 0.0, "b": 0.0}, config)
    with_quality = rank(entrants, {"a": 0.0, "b": 1.0}, config)
    assert without.ordered[0][0] == "a"
    assert with_quality.ordered[0][0] == "b"


def test_single_qualifier_scores_full_marks(config):
    ranking = rank(qualification(("solo", -5.0)), None, config)
    assert ranking.final_scores == {"solo": 1.0}
    assert ranking.rank_of == {"solo": 1}


def test_without_quality_signal_leak_score_decides(config):
    entrants = qualification(("a", -6.0), ("b", -4.5))
    assert rank(entrants, None, config).final_scores == entrants.normalized_leak


def test_no_qualifiers_yields_nothing(config):
    ranking = rank(Qualification(), None, config)
    assert ranking.final_scores == {}
    assert ranking.ordered == []


# ─── ranking ─────────────────────────────────────────────────────────────


def test_zero_scores_do_not_rank(config):
    entrants = qualification(("a", -6.0), ("b", -3.0), ("c", -4.5))
    ranking = rank(entrants, {"a": 1.0, "b": 0.0, "c": 1.0}, config)
    assert "b" not in ranking.rank_of  # normalised leak 0 and no quality wins
    assert [s for s, _ in ranking.ordered] == ["a", "c"]


def test_ranking_is_deterministic_on_ties(config):
    entrants = qualification(("b", -4.5), ("a", -4.5))
    ranking = rank(entrants, {"a": 0.5, "b": 0.5}, config)
    assert [s for s, _ in ranking.ordered] == ["a", "b"]


def test_rank_positions_start_at_one(config):
    entrants = qualification(("a", -6.0), ("b", -4.5))
    ranking = rank(entrants, {"a": 1.0, "b": 0.0}, config)
    assert ranking.rank_of == {"a": 1, "b": 2}


def test_every_positive_score_is_ranked(config):
    """No cap: a submitter always learns where they placed."""
    entrants = qualification(*[(f"s{i}", -6.0 + i * 0.4) for i in range(6)])
    ranking = rank(entrants, {f"s{i}": 0.0 for i in range(6)}, config)
    assert len(ranking.ordered) == 6
    assert sorted(ranking.rank_of.values()) == [1, 2, 3, 4, 5, 6]


# ─── config validation ───────────────────────────────────────────────────


def test_config_rejects_equal_bounds():
    with pytest.raises(ValueError):
        EvaluationConfig(min_eval_score=-3.0, leak_best_score=-3.0)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown config keys"):
        EvaluationConfig.from_dict({"nope": 1})


def test_config_round_trips_through_dict():
    config = EvaluationConfig(leak_weight=0.6, quality_weight=0.4)
    assert EvaluationConfig.from_dict(config.to_dict()) == config


def test_config_reads_the_environment(monkeypatch):
    monkeypatch.setenv("WIGIN_TLLM_MIN_EVAL_SCORE", "-4.5")
    monkeypatch.setenv("WIGIN_TLLM_QUALITY_ENABLED", "false")
    monkeypatch.setenv("WIGIN_TLLM_SVD_EXEMPT_SUBMITTERS", "alice, bob")
    monkeypatch.setenv("WIGIN_TLLM_TOP_N_FOR_QUALITY", "3")
    config = EvaluationConfig.from_env()
    assert config.min_eval_score == -4.5
    assert config.quality_enabled is False
    assert config.svd_exempt_submitters == ["alice", "bob"]
    assert config.top_n_for_quality == 3
