"""Stage 2 — quality evaluation via round-robin duels.

Stage 1 only proves a model respects its temporal boundary; a model can do
that and still be useless. Stage 2 has every qualified model continue the
same incomplete texts, then plays all pairs off against each other under a
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

from ..types import CompletionPrompt, ModelRef
from .judge import Judge, Verdict

logger = logging.getLogger(__name__)


# ─── completion generation ───────────────────────────────────────────────


class CompletionProvider(ABC):
    """Produces one completion per prompt for a submitter and cutoff year."""

    @abstractmethod
    def completions_for(
        self, submitter_id: str, year: int, prompts: Sequence[CompletionPrompt]
    ) -> list[str]:
        ...


class ModelCompletionProvider(CompletionProvider):
    """Loads the submitter's model for that year and generates completions.

    A model that is missing or fails to load yields empty completions rather
    than aborting the round: a broken submission should lose its duels, not
    stop everyone else's.
    """

    def __init__(self, submissions: dict, device, max_new_tokens: int = 50):
        # Anything with `.ref_for_year(year) -> ModelRef | None`.
        self.submissions = submissions
        self.device = device
        self.max_new_tokens = max_new_tokens

    def completions_for(
        self, submitter_id: str, year: int, prompts: Sequence[CompletionPrompt]
    ) -> list[str]:
        from ..models.loader import load_model
        from ..models.store import free_device_memory, resolve_model

        submission = self.submissions.get(submitter_id)
        ref: Optional[ModelRef] = submission.ref_for_year(year) if submission else None
        if ref is None:
            logger.warning(f"{submitter_id}: no model for year {year}, using empty completions")
            return [""] * len(prompts)

        model = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = resolve_model(ref, tmpdir)
                model, _ = load_model(path, self.device)
                completions = []
                step = max(1, len(prompts) // 3)
                for i, prompt in enumerate(prompts):
                    completions.append(
                        model.generate(prompt.prompt, max_new_tokens=self.max_new_tokens)
                    )
                    if (i + 1) % step == 0 or i + 1 == len(prompts):
                        logger.info(f"{submitter_id}: generated {i + 1}/{len(prompts)}")
                return completions
        except Exception as e:
            logger.error(f"{submitter_id}: completion generation FAILED — {type(e).__name__}: {e}")
            return [""] * len(prompts)
        finally:
            del model
            free_device_memory()


class StaticCompletionProvider(CompletionProvider):
    """Serves pre-computed completions. For tests and replaying a tournament."""

    def __init__(self, completions: dict[int, dict[str, list[str]]]):
        # {year: {submitter_id: [completion, ...]}}
        self.completions = completions

    def completions_for(
        self, submitter_id: str, year: int, prompts: Sequence[CompletionPrompt]
    ) -> list[str]:
        return self.completions.get(year, {}).get(submitter_id, [""] * len(prompts))


# ─── tournament ──────────────────────────────────────────────────────────

_SWAP_BACK: dict[Verdict, Verdict] = {"a": "b", "b": "a", "tie": "tie"}


def duel(
    completions: dict[str, list[str]],
    left: str,
    right: str,
    prompts: Sequence[CompletionPrompt],
    judge: Judge,
    rng: random.Random,
) -> Optional[str]:
    """Play one pair over all prompts. Returns the winner, or None on a tie.

    Each prompt randomly swaps which completion is presented first and the
    verdict is mapped back. Judges — LLMs especially — tend to favour whichever
    they see first; randomising the position spreads that bias evenly instead
    of letting it decide the duel.
    """
    tasks: list[tuple[CompletionPrompt, str, str]] = []
    swaps: list[bool] = []
    for i, prompt in enumerate(prompts):
        swap = rng.random() < 0.5
        swaps.append(swap)
        first, second = (right, left) if swap else (left, right)
        tasks.append((prompt, completions[first][i], completions[second][i]))

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
    completions: dict[str, list[str]],
    prompts: Sequence[CompletionPrompt],
    submitter_ids: Sequence[str],
    judge: Judge,
    rng: random.Random,
) -> dict[str, float]:
    """Every pair plays once. Returns win rate per submitter in [0, 1].

    A drawn duel scores for neither side, so a win rate of 0 means "won
    nothing" — which covers both losing every duel and drawing every one. The
    logged record separates the two; with a single opponent that distinction
    is the whole story.
    """
    wins = {submitter_id: 0 for submitter_id in submitter_ids}
    draws = {submitter_id: 0 for submitter_id in submitter_ids}

    for i in range(len(submitter_ids)):
        for j in range(i + 1, len(submitter_ids)):
            left, right = submitter_ids[i], submitter_ids[j]
            logger.info(f"Duel: {left} vs {right}")
            winner = duel(completions, left, right, prompts, judge, rng)
            if winner is None:
                draws[left] += 1
                draws[right] += 1
            else:
                wins[winner] += 1

    opponents = max(1, len(submitter_ids) - 1)
    win_rates = {s: wins[s] / opponents for s in submitter_ids}
    for submitter_id in submitter_ids:
        won, drew = wins[submitter_id], draws[submitter_id]
        logger.info(
            f"{submitter_id}: {won}W-{drew}D-{opponents - won - drew}L "
            f"win_rate={win_rates[submitter_id]:.4f}"
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
    prompts: Sequence[CompletionPrompt],
    all_years: Sequence[int],
    provider: CompletionProvider,
    judge: Judge,
    rng: Optional[random.Random] = None,
    year_samples: int = 2,
    opponent_ids: Sequence[str] = (),
) -> dict[str, float]:
    """Run the tournament. Returns win rate per submitter, averaged over years.

    `opponent_ids` are reference models that take part in the duels but are
    not themselves scored. They do two things: they make a win rate possible
    at all when only one submission qualifies, and they anchor the scale, so
    a rate stays comparable between rounds instead of depending entirely on
    who else happened to enter.
    """
    rng = rng or random.Random()
    submitter_ids = list(submitter_ids)
    participants = submitter_ids + [o for o in opponent_ids if o not in submitter_ids]

    if len(participants) < 2 or not prompts:
        return {submitter_id: 0.0 for submitter_id in submitter_ids}

    eval_years = select_eval_years(all_years, year_samples, rng)
    logger.info(f"Quality eval years: {eval_years}")
    if len(participants) > len(submitter_ids):
        logger.info(f"Reference opponents: {participants[len(submitter_ids):]}")

    per_year: list[dict[str, float]] = []
    for year in eval_years:
        logger.info(f"=== Quality round: year {year} ===")
        completions = {}
        for participant in participants:
            logger.info(f"{participant}: generating completions (year {year})")
            completions[participant] = provider.completions_for(participant, year, prompts)
        per_year.append(run_round_robin(completions, prompts, participants, judge, rng))

    win_rates = {
        submitter_id: sum(year_rates[submitter_id] for year_rates in per_year) / len(per_year)
        for submitter_id in submitter_ids
    }
    for submitter_id in submitter_ids:
        logger.info(f"{submitter_id}: avg_win_rate={win_rates[submitter_id]:.4f}")
    return win_rates
