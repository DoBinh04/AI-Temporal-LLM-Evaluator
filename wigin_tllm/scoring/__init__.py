"""Scoring: consistency probes, anti-copy gates, quality duels, aggregation."""

from .aggregate import build_round_results, mean_year_score, normalize_leak_score, qualify, rank
from .judge import Judge, OpenAIJudge, ReferenceOverlapJudge, ScriptedJudge
from .leak import assess_year, probe, score_items
from .quality import (
    AnswerProvider,
    ModelAnswerProvider,
    StaticAnswerProvider,
    run_quality_duels,
)
from .svd_gate import SvdGate, compare_spectra, dedup_by_svd, svd_spectra

__all__ = [
    # stage 1
    "score_items",
    "probe",
    "assess_year",
    # stage 2
    "AnswerProvider",
    "ModelAnswerProvider",
    "StaticAnswerProvider",
    "run_quality_duels",
    # judges
    "Judge",
    "OpenAIJudge",
    "ReferenceOverlapJudge",
    "ScriptedJudge",
    # anti-copy
    "SvdGate",
    "svd_spectra",
    "compare_spectra",
    "dedup_by_svd",
    # aggregation
    "normalize_leak_score",
    "mean_year_score",
    "qualify",
    "rank",
    "build_round_results",
]
