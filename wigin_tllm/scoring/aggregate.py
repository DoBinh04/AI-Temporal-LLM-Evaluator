"""Score aggregation: per-year scores in, ranked results out.

Deliberately pure Python — no torch, no numpy — so the scoring rules can be
read, reasoned about, and unit-tested without loading a model.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from ..config import WORST_SCORE, EvaluationConfig
from ..types import (
    ConsistencyResults,
    Qualification,
    Ranking,
    RoundResults,
    SubmitterResult,
)

logger = logging.getLogger(__name__)


def normalize_leak_score(score: float, min_eval_score: float, leak_best_score: float) -> float:
    """Map a raw leak score onto [0, 1], higher is better.

    Raw scores are negative and *lower* is better, so the mapping inverts:
    `min_eval_score` (the qualification threshold) maps to 0.0 and
    `leak_best_score` maps to 1.0. Clamping means the worst-possible score of
    0.0 can never earn credit.
    """
    span = min_eval_score - leak_best_score
    return max(0.0, min(1.0, (min_eval_score - score) / span))


def mean_year_score(year_scores: dict[int, float], all_years: Sequence[int]) -> float:
    """Average across *all* in-scope years, not just the ones submitted.

    Dividing by the full year count is what makes a missing year expensive:
    it contributes WORST_SCORE (0.0) and drags the mean toward zero.
    """
    if not all_years:
        return WORST_SCORE
    return sum(year_scores.get(year, WORST_SCORE) for year in all_years) / len(all_years)


def qualify(leak_scores: dict[str, float], config: EvaluationConfig) -> Qualification:
    """Select stage-2 entrants and normalise their leak scores.

    A submitter qualifies when its mean year score is strictly below
    `min_eval_score`; the best `top_n_for_quality` of those go through. Ties
    break on submitter id so a round is reproducible.
    """
    ranked = sorted(leak_scores.items(), key=lambda kv: (kv[1], kv[0]))
    entrants = [
        (submitter_id, score)
        for submitter_id, score in ranked
        if score < config.min_eval_score
    ][: config.top_n_for_quality]

    logger.info(f"Qualified: {len(entrants)} — {[s for s, _ in entrants]}")

    return Qualification(
        entrants=entrants,
        normalized_leak={
            submitter_id: normalize_leak_score(
                score, config.min_eval_score, config.leak_best_score
            )
            for submitter_id, score in entrants
        },
    )


def _blend(
    qualification: Qualification,
    win_rates: Optional[dict[str, float]],
    config: EvaluationConfig,
) -> dict[str, float]:
    """Combine the two stages into one score per entrant.

    With no quality signal — stage 2 skipped, or a single entrant with nobody
    to duel — the leak score carries the result on its own.
    """
    if not qualification.entrants:
        return {}

    if len(qualification) == 1:
        return {qualification.ids[0]: 1.0}

    if win_rates is None:
        return dict(qualification.normalized_leak)

    blended = {}
    for submitter_id in qualification.ids:
        leak = qualification.normalized_leak[submitter_id]
        quality = win_rates.get(submitter_id, 0.0)
        blended[submitter_id] = config.leak_weight * leak + config.quality_weight * quality
        logger.info(
            f"{submitter_id}: final={blended[submitter_id]:.4f} "
            f"(leak={leak:.4f} quality={quality:.4f})"
        )
    return blended


def _order(final_scores: dict[str, float]) -> list[tuple[str, float]]:
    """Best-first ordering of submitters with a positive score.

    A zero score means the submission failed outright, so it is listed but
    not ranked. Ties break on submitter id to keep a round reproducible.
    """
    return sorted(
        ((s, score) for s, score in final_scores.items() if score > 0),
        key=lambda kv: (-kv[1], kv[0]),
    )


def rank(
    qualification: Qualification,
    win_rates: Optional[dict[str, float]],
    config: EvaluationConfig,
) -> Ranking:
    """Blend the two stages and order the field."""
    final_scores = _blend(qualification, win_rates, config)
    return Ranking(
        win_rates=win_rates,
        final_scores=final_scores,
        ordered=_order(final_scores),
    )


def build_round_results(
    round_id: int,
    consistency: ConsistencyResults,
    qualification: Qualification,
    ranking: Ranking,
) -> RoundResults:
    """Assemble the serialisable round record, best submitter first."""
    qualified = set(qualification.ids)
    rank_of = ranking.rank_of
    win_rates = ranking.win_rates or {}

    return RoundResults(
        round_id=round_id,
        submitters=[
            SubmitterResult(
                submitter_id=submitter_id,
                leak_score=leak_score,
                year_scores=consistency.year_scores.get(submitter_id, {}),
                qualified=submitter_id in qualified,
                normalized_leak=qualification.normalized_leak.get(submitter_id, 0.0),
                quality_win_rate=win_rates.get(submitter_id, 0.0),
                final_score=ranking.final_scores.get(submitter_id, 0.0),
                rank=rank_of.get(submitter_id),
                disqualified_reason=consistency.disqualified.get(submitter_id, ""),
            )
            for submitter_id, leak_score in sorted(
                consistency.leak_scores.items(),
                key=lambda kv: (-ranking.final_scores.get(kv[0], 0.0), kv[0]),
            )
        ],
    )
