"""In-memory data source — fixtures for tests and quick experiments."""

from __future__ import annotations

from ..types import Benchmark, QualityQuestion, RoundResults, Submission, YearEvaluation
from .base import DataSource


class InMemoryDataSource(DataSource):
    def __init__(
        self,
        years: list[int],
        benchmarks: dict[int, dict[str, Benchmark]],
        submissions: list[Submission],
        questions: list[QualityQuestion] | None = None,
        current_round: int = 1,
    ):
        self.years = sorted(years)
        self.benchmarks = benchmarks
        self.submissions = submissions
        self.questions = questions or []
        self.current_round = current_round
        # Captured outputs, for assertions.
        self.saved_results: dict[int, RoundResults] = {}
        self.saved_evaluations: list[YearEvaluation] = []

    def get_current_round(self) -> int:
        return self.current_round

    def get_years(self, round_id: int) -> list[int]:
        return list(self.years)

    def get_benchmark(self, year: int, kind: str) -> Benchmark:
        return self.benchmarks[year][kind]

    def get_submissions(self, round_id: int) -> list[Submission]:
        return list(self.submissions)

    def get_quality_questions(self) -> list[QualityQuestion]:
        return list(self.questions)

    def save_year_evaluation(self, round_id: int, evaluation: YearEvaluation) -> bool:
        self.saved_evaluations.append(evaluation)
        return True

    def save_results(self, round_id: int, results: RoundResults) -> None:
        self.saved_results[round_id] = results
