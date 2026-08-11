"""Stage-2 tournament mechanics: judging, swapping, tallying, year selection."""

from __future__ import annotations

import random

import pytest

from wigin_tllm.scoring.judge import Judge, ReferenceOverlapJudge, ScriptedJudge
from wigin_tllm.scoring.quality import (
    StaticCompletionProvider,
    duel,
    run_quality_duels,
    run_round_robin,
    select_eval_years,
)
from wigin_tllm.types import CompletionPrompt


class AlwaysFirstJudge(Judge):
    """Always prefers whichever answer is presented first — pure position bias."""

    def judge_pair(self, question, answer_a, answer_b):
        return "a"


class PrefersTextJudge(Judge):
    """Picks whichever answer contains a marker, wherever it is placed."""

    def __init__(self, marker: str):
        self.marker = marker

    def judge_pair(self, question, answer_a, answer_b):
        in_a, in_b = self.marker in answer_a, self.marker in answer_b
        if in_a and not in_b:
            return "a"
        if in_b and not in_a:
            return "b"
        return "tie"


QUESTIONS = [CompletionPrompt(prompt=f"q{i}", reference="alpha") for i in range(8)]


# ─── position-bias handling ──────────────────────────────────────────────


def test_swapping_neutralises_a_purely_positional_judge():
    """A judge that always says 'first' must not hand anyone a systematic win."""
    answers = {"a": ["x"] * 8, "b": ["y"] * 8}
    rng = random.Random(0)
    winners = [duel(answers, "a", "b", QUESTIONS, AlwaysFirstJudge(), rng) for _ in range(40)]
    assert winners.count("a") > 0 and winners.count("b") > 0


def test_verdicts_are_mapped_back_after_a_swap():
    """A judge keyed on content, not position, must always find the same winner."""
    answers = {"good": ["contains-alpha"] * 8, "bad": ["nothing"] * 8}
    rng = random.Random(1)
    for _ in range(20):
        assert duel(answers, "good", "bad", QUESTIONS, PrefersTextJudge("alpha"), rng) == "good"
        assert duel(answers, "bad", "good", QUESTIONS, PrefersTextJudge("alpha"), rng) == "good"


def test_equal_answers_tie():
    answers = {"a": ["same"] * 8, "b": ["same"] * 8}
    assert duel(answers, "a", "b", QUESTIONS, PrefersTextJudge("alpha"), random.Random(2)) is None


def test_majority_of_questions_decides_the_duel():
    judge = ScriptedJudge(["a", "a", "b"])  # cycles: 2 of every 3 favour the first slot
    answers = {"a": ["x"] * 3, "b": ["y"] * 3}
    # With swapping disabled by a fixed rng that never swaps, "a" should win.
    rng = random.Random()
    rng.random = lambda: 1.0  # never < 0.5, so no swap
    assert duel(answers, "a", "b", QUESTIONS[:3], judge, rng) == "a"


# ─── round robin ─────────────────────────────────────────────────────────


def test_every_pair_meets_once():
    judge = ScriptedJudge(["tie"])
    answers = {m: ["x"] * 8 for m in ["a", "b", "c", "d"]}
    run_round_robin(answers, QUESTIONS, ["a", "b", "c", "d"], judge, random.Random(3))
    # 4 submitters -> 6 pairs, 8 questions each
    assert len(judge.calls) == 6 * 8


def test_win_rate_is_share_of_opponents_beaten():
    answers = {"strong": ["alpha"] * 8, "mid": ["alpha"] * 8, "weak": ["zzz"] * 8}
    rates = run_round_robin(
        answers, QUESTIONS, ["strong", "mid", "weak"], PrefersTextJudge("alpha"), random.Random(4)
    )
    assert rates["strong"] == pytest.approx(0.5)  # beats weak, ties mid
    assert rates["mid"] == pytest.approx(0.5)
    assert rates["weak"] == 0.0


def test_undefeated_submitter_scores_one():
    answers = {"champ": ["alpha"] * 8, "a": ["zzz"] * 8, "b": ["zzz"] * 8}
    rates = run_round_robin(
        answers, QUESTIONS, ["champ", "a", "b"], PrefersTextJudge("alpha"), random.Random(5)
    )
    assert rates["champ"] == 1.0


def test_single_submitter_has_no_opponents():
    rates = run_round_robin({"solo": ["x"] * 8}, QUESTIONS, ["solo"], ScriptedJudge(["tie"]), random.Random(6))
    assert rates == {"solo": 0.0}


# ─── year selection ──────────────────────────────────────────────────────


def test_oldest_year_is_always_included():
    for seed in range(10):
        chosen = select_eval_years([2013, 2014, 2015, 2016], 2, random.Random(seed))
        assert chosen[0] == 2013
        assert len(chosen) == 2


def test_second_year_varies_across_seeds():
    picks = {tuple(select_eval_years([2013, 2014, 2015, 2016], 2, random.Random(s))) for s in range(20)}
    assert len(picks) > 1


def test_year_selection_is_reproducible_for_a_seed():
    a = select_eval_years([2013, 2014, 2015, 2016], 3, random.Random(42))
    b = select_eval_years([2013, 2014, 2015, 2016], 3, random.Random(42))
    assert a == b


def test_samples_are_distinct():
    chosen = select_eval_years([2013, 2014, 2015, 2016], 4, random.Random(7))
    assert sorted(chosen) == [2013, 2014, 2015, 2016]


def test_cannot_sample_more_years_than_exist():
    assert select_eval_years([2013, 2014], 9, random.Random(8)) == [2013, 2014]


def test_no_years_yields_no_selection():
    assert select_eval_years([], 2, random.Random(9)) == []


# ─── full tournament ─────────────────────────────────────────────────────


def test_tournament_averages_across_years():
    """Winning one year and losing another lands in the middle."""
    provider = StaticCompletionProvider(
        {
            2013: {"a": ["alpha"] * 8, "b": ["zzz"] * 8},
            2014: {"a": ["zzz"] * 8, "b": ["alpha"] * 8},
        }
    )
    rates = run_quality_duels(
        ["a", "b"], QUESTIONS, [2013, 2014], provider, PrefersTextJudge("alpha"),
        rng=random.Random(0), year_samples=2,
    )
    assert rates["a"] == pytest.approx(0.5)
    assert rates["b"] == pytest.approx(0.5)


def test_missing_answers_lose():
    provider = StaticCompletionProvider({2013: {"a": ["alpha"] * 8}})  # "b" has nothing
    rates = run_quality_duels(
        ["a", "b"], QUESTIONS, [2013], provider, PrefersTextJudge("alpha"),
        rng=random.Random(0), year_samples=1,
    )
    assert rates["a"] == 1.0
    assert rates["b"] == 0.0


def test_no_questions_means_no_signal():
    provider = StaticCompletionProvider({})
    assert run_quality_duels(["a", "b"], [], [2013], provider, ScriptedJudge(["a"])) == {"a": 0.0, "b": 0.0}


# ─── reference-overlap judge ─────────────────────────────────────────────


def test_overlap_judge_prefers_the_closer_answer():
    judge = ReferenceOverlapJudge()
    q = CompletionPrompt(prompt="who reached mars", reference="the vela probe")
    assert judge.judge_pair(q, "the vela probe", "something else entirely") == "a"
    assert judge.judge_pair(q, "something else entirely", "the vela probe") == "b"


def test_overlap_judge_ties_on_equal_overlap():
    judge = ReferenceOverlapJudge()
    q = CompletionPrompt(prompt="p", reference="alpha beta")
    assert judge.judge_pair(q, "alpha", "beta") == "tie"


def test_overlap_judge_is_case_and_punctuation_insensitive():
    judge = ReferenceOverlapJudge()
    q = CompletionPrompt(prompt="p", reference="Vela Probe")
    assert judge.judge_pair(q, "vela, probe!", "nothing") == "a"


# ─── reference opponents ─────────────────────────────────────────────────


def test_opponents_duel_but_are_not_scored():
    """A lone submitter has nobody to duel; a reference supplies one."""
    provider = StaticCompletionProvider(
        {2013: {"solo": ["alpha"] * 8, "ref": ["zzz"] * 8}}
    )
    rates = run_quality_duels(
        ["solo"], QUESTIONS, [2013], provider, PrefersTextJudge("alpha"),
        rng=random.Random(0), year_samples=1, opponent_ids=["ref"],
    )
    assert rates == {"solo": 1.0}  # the reference gets no entry of its own


def test_without_opponents_a_lone_submitter_scores_nothing():
    provider = StaticCompletionProvider({2013: {"solo": ["alpha"] * 8}})
    rates = run_quality_duels(
        ["solo"], QUESTIONS, [2013], provider, PrefersTextJudge("alpha"),
        rng=random.Random(0), year_samples=1,
    )
    assert rates == {"solo": 0.0}


def test_opponents_widen_the_denominator():
    """Two references give the rate resolution a single one cannot."""
    provider = StaticCompletionProvider(
        {2013: {"solo": ["alpha"] * 8, "strong": ["alpha"] * 8, "weak": ["zzz"] * 8}}
    )
    rates = run_quality_duels(
        ["solo"], QUESTIONS, [2013], provider, PrefersTextJudge("alpha"),
        rng=random.Random(0), year_samples=1, opponent_ids=["strong", "weak"],
    )
    assert rates["solo"] == pytest.approx(0.5)  # drew with strong, beat weak


def test_an_opponent_that_is_also_a_submitter_is_not_double_counted():
    provider = StaticCompletionProvider(
        {2013: {"a": ["alpha"] * 8, "b": ["zzz"] * 8}}
    )
    rates = run_quality_duels(
        ["a", "b"], QUESTIONS, [2013], provider, PrefersTextJudge("alpha"),
        rng=random.Random(0), year_samples=1, opponent_ids=["a"],
    )
    assert rates["a"] == 1.0  # denominator stays 1, not 2


def test_drawing_every_duel_scores_the_same_as_losing_every_duel():
    """A known limit of the rule: a win rate of 0 does not distinguish the two.

    The per-submitter record in the logs is what separates them, which is why
    a single reference opponent is rarely enough.
    """
    drew = StaticCompletionProvider({2013: {"solo": ["same"] * 8, "ref": ["same"] * 8}})
    lost = StaticCompletionProvider({2013: {"solo": ["zzz"] * 8, "ref": ["alpha"] * 8}})
    common = dict(rng=random.Random(0), year_samples=1, opponent_ids=["ref"])

    drew_rate = run_quality_duels(
        ["solo"], QUESTIONS, [2013], drew, PrefersTextJudge("alpha"), **common
    )
    lost_rate = run_quality_duels(
        ["solo"], QUESTIONS, [2013], lost, PrefersTextJudge("alpha"), **common
    )
    assert drew_rate["solo"] == lost_rate["solo"] == 0.0
