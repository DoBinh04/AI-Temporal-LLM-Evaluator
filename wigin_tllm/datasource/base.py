"""The data-source contract.

Everything the pipeline needs from the outside world goes through this one
interface: which years to score, the probe sets, who submitted what, the
quality questions, and where results go.

Keeping it an interface means the pipeline can run against a local
directory, a remote service, or an in-memory fixture without any change to
the scoring code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..types import Benchmark, QualityQuestion, RoundResults, Submission, YearEvaluation


class DataSource(ABC):
    """Read evaluation inputs, write evaluation outputs."""

    # ── Round metadata ───────────────────────────────────────────────────

    @abstractmethod
    def get_current_round(self) -> int:
        """The round id that should be evaluated now."""

    @abstractmethod
    def get_years(self, round_id: int) -> list[int]:
        """Cutoff years in scope for this round."""

    # ── Evaluation inputs ────────────────────────────────────────────────

    @abstractmethod
    def get_benchmark(self, year: int, kind: str) -> Benchmark:
        """Probe set for one year.

        ``kind`` is ``"unknown"`` (post-cutoff facts the model must not know)
        or ``"known"`` (pre-cutoff facts the model must know).
        """

    @abstractmethod
    def get_submissions(self, round_id: int) -> list[Submission]:
        """All submissions captured for this round."""

    def get_quality_questions(self) -> list[QualityQuestion]:
        """Open-ended questions for stage 2. Empty list disables stage 2."""
        return []

    # ── Outputs ──────────────────────────────────────────────────────────

    def save_year_evaluation(self, round_id: int, evaluation: YearEvaluation) -> bool:
        """Persist one (submitter, year) result as soon as it is known.

        Returning False marks the row unsynced so the pipeline retries it on
        the next run. The default implementation is a no-op that reports
        success; the SQLite cache always keeps its own copy regardless.
        """
        return True

    def save_results(self, round_id: int, results: RoundResults) -> None:
        """Persist the final round outcome."""

    # ── Convenience ──────────────────────────────────────────────────────

    def preload_benchmarks(self, years: Iterable[int]) -> dict[int, dict[str, Benchmark]]:
        """Fetch every probe set up front so a broken source fails fast.

        Downloading dozens of models and only then discovering the probe sets
        are unreachable wastes hours; this is why the pipeline calls it before
        stage 1.
        """
        return {
            year: {
                "unknown": self.get_benchmark(year, "unknown"),
                "known": self.get_benchmark(year, "known"),
            }
            for year in years
        }
