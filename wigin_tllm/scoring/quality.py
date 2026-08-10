"""Stage 2 — quality evaluation via round-robin duels.

Stage 1 only proves a model respects its temporal boundary; a model can do
that and still be useless. Stage 2 has every qualified submitter answer the same
open-ended questions, then plays all pairs off against each other under a
judge. The share of duels won becomes the quality score.

Duels are run on more than one cutoff year so a submitter cannot concentrate
all of its effort on a single year.
"""

from __future__ import annotations

import logging
import random
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from ..types import ModelRef, QualityQuestion, Submission
from .judge import Judge, Verdict

logger = logging.getLogger(__name__)


# ─── answer generation ───────────────────────────────────────────────────


class AnswerProvider(ABC):
    """Produces one answer per question for a given submitter and cutoff year."""

    @abstractmethod
    def answers_for(
        self, submitter_id: str, year: int, questions: Sequence[QualityQuestion]
    ) -> list[str]:
        ...


class ModelAnswerProvider(AnswerProvider):
    """Loads the submitter's model for that year and generates answers.

    A model that is missing or fails to load yields empty answers rather than
    aborting the round: a broken submission should lose its duels, not stop
    everyone else's.
    """

    def __init__(
        self,
        submissions: dict[str, Submission],
        device,
        max_new_tokens: int = 50,
    ):
        self.submissions = submissions
        self.device = device
        self.max_new_tokens = max_new_tokens

    def answers_for(
        self, submitter_id: str, year: int, questions: Sequence[QualityQuestion]
    ) -> list[str]:
        from ..models.loader import load_model
        from ..models.store import free_device_memory, resolve_model

        submission = self.submissions.get(submitter_id)
        ref: Optional[ModelRef] = submission.ref_for_year(year) if submission else None
        if ref is None:
            logger.warning(f"{submitter_id}: no model for year {year}, using empty answers")
            return [""] * len(questions)

        model = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = resolve_model(ref, tmpdir)
                model, _ = load_model(path, self.device)
                answers = []
                step = max(1, len(questions) // 3)
                for i, question in enumerate(questions):
                    answers.append(model.generate(question.prompt, max_new_tokens=self.max_new_tokens))
                    if (i + 1) % step == 0 or i + 1 == len(questions):
                        logger.info(f"{submitter_id}: generated {i + 1}/{len(questions)}")
                return answers
        except Exception as e:
            logger.error(f"{submitter_id}: answer generation FAILED — {type(e).__name__}: {e}")
            return [""] * len(questions)
        finally:
            del model
            free_device_memory()


class StaticAnswerProvider(AnswerProvider):
    """Serves pre-computed answers. For tests and replaying a tournament."""

    def __init__(self, answers: dict[int, dict[str, list[str]]]):
        # {year: {submitter_id: [answer, ...]}}
        self.answers = answers

    def answers_for(
        self, submitter_id: str, year: int, questions: Sequence[QualityQuestion]
    ) -> list[str]:
        return self.answers.get(year, {}).get(submitter_id, [""] * len(questions))


# ─── tournament ──────────────────────────────────────────────────────────

_SWAP_BACK: dict[Verdict, Verdict] = {"a": "b", "b": "a", "tie": "tie"}


def duel(
    answers: dict[str, list[str]],
    left: str,
    right: str,
    questions: Sequence[QualityQuestion],
    judge: Judge,
    rng: random.Random,
) -> Optional[str]:
    """Play one pair over all questions. Returns the winner, or None on a tie.

    Each question randomly swaps which answer is presented first and the
    verdict is mapped back. Judges — LLMs especially — tend to favour whichever
    answer they see first; randomising the position spreads that bias evenly
    instead of letting it decide the duel.
    """
    tasks: list[tuple[QualityQuestion, str, str]] = []
    swaps: list[bool] = []
    for i, question in enumerate(questions):
        swap = rng.random() < 0.5
        swaps.append(swap)
        first, second = (right, left) if swap else (left, right)
        tasks.append((question, answers[first][i], answers[second][i]))

    verdicts = judge.judge_batch(tasks)

    wins_a = wins_b = 0
    for i, raw in enumerate(verdicts):
        verdict = _SWAP_BACK[raw] if swaps[i] else raw
        if verdict == "a":
            wins_a += 1
        elif verdict == "b":
            wins_b += 1

    logger.info(f"  {left} ({wins_a}) vs {right} ({wins_b})")
    if wins_a > wins_b:
        return left
    if wins_b > wins_a:
        return right
    return None


def run_round_robin(
    answers: dict[str, list[str]],
    questions: Sequence[QualityQuestion],
    submitter_ids: Sequence[str],
    judge: Judge,
    rng: random.Random,
) -> dict[str, float]:
    """Every pair plays once. Returns win rate per submitter in [0, 1]."""
    wins = {submitter_id: 0 for submitter_id in submitter_ids}

    for i in range(len(submitter_ids)):
        for j in range(i + 1, len(submitter_ids)):
            left, right = submitter_ids[i], submitter_ids[j]
            logger.info(f"Duel: {left} vs {right}")
            winner = duel(answers, left, right, questions, judge, rng)
            if winner is not None:
                wins[winner] += 1

    opponents = max(1, len(submitter_ids) - 1)
    win_rates = {submitter_id: wins[submitter_id] / opponents for submitter_id in submitter_ids}
    for submitter_id in submitter_ids:
        logger.info(
            f"{submitter_id}: wins={wins[submitter_id]}/{opponents} win_rate={win_rates[submitter_id]:.4f}"
        )
    return win_rates


def select_eval_years(all_years: Sequence[int], samples: int, rng: random.Random) -> list[int]:
    """Oldest year always, plus `samples - 1` drawn from the rest.

    The oldest year is fixed so results stay comparable across rounds; the
    rest are random so effort cannot be concentrated on a predictable year.
    """
    if not all_years:
        return []
    ordered = sorted(all_years)
    chosen = [ordered[0]]
    remaining = ordered[1:]
    extra = min(max(0, samples - 1), len(remaining))
    if extra:
        chosen.extend(sorted(rng.sample(remaining, extra)))
    return chosen


def run_quality_duels(
    submitter_ids: Sequence[str],
    questions: Sequence[QualityQuestion],
    all_years: Sequence[int],
    provider: AnswerProvider,
    judge: Judge,
    rng: Optional[random.Random] = None,
    year_samples: int = 2,
) -> dict[str, float]:
    """Run the tournament. Returns win rate per submitter, averaged over years."""
    rng = rng or random.Random()
    submitter_ids = list(submitter_ids)
    if not submitter_ids or not questions:
        return {submitter_id: 0.0 for submitter_id in submitter_ids}

    eval_years = select_eval_years(all_years, year_samples, rng)
    logger.info(f"Quality eval years: {eval_years}")

    per_year: list[dict[str, float]] = []
    for year in eval_years:
        logger.info(f"=== Quality round: year {year} ===")
        answers = {}
        for submitter_id in submitter_ids:
            logger.info(f"{submitter_id}: generating answers (year {year})")
            answers[submitter_id] = provider.answers_for(submitter_id, year, questions)
        per_year.append(run_round_robin(answers, questions, submitter_ids, judge, rng))

    if not per_year:
        return {submitter_id: 0.0 for submitter_id in submitter_ids}

    win_rates = {
        submitter_id: sum(year_rates[submitter_id] for year_rates in per_year) / len(per_year)
        for submitter_id in submitter_ids
    }
    for submitter_id in submitter_ids:
        logger.info(f"{submitter_id}: avg_win_rate={win_rates[submitter_id]:.4f}")
    return win_rates
