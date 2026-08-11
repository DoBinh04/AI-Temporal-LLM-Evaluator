"""Turning per-year scores into one number.

Pure Python — no torch, no numpy — so the scoring rules can be read and
tested without loading a model.
"""

from __future__ import annotations

from typing import Sequence

from ..config import WORST_SCORE


def normalize_leak_score(score: float, min_eval_score: float, leak_best_score: float) -> float:
    """Map a raw leak score onto [0, 1], higher is better.

    Raw scores are negative and *lower* is better, so the mapping inverts:
    `min_eval_score` (the qualification threshold) maps to 0.0 and
    `leak_best_score` maps to 1.0. Clamping at 1.0 is why consistency work
    past `leak_best_score` earns nothing, and clamping at 0.0 is why the
    worst-possible score of 0.0 can never earn credit.
    """
    span = min_eval_score - leak_best_score
    return max(0.0, min(1.0, (min_eval_score - score) / span))


def mean_year_score(year_scores: dict[int, float], all_years: Sequence[int]) -> float:
    """Average across *all* years in scope, not just the ones scored.

    Dividing by the full year count is what makes a missing year expensive:
    it contributes WORST_SCORE (0.0) and drags the mean toward zero.
    """
    if not all_years:
        return WORST_SCORE
    return sum(year_scores.get(year, WORST_SCORE) for year in all_years) / len(all_years)
