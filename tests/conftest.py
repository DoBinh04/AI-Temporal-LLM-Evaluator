"""Shared fixtures.

Deliberately dependency-light: most tests need only the scoring maths, which
is pure Python. Anything that needs torch says so explicitly.
"""

from __future__ import annotations

import pytest

from wigin_tllm.config import EvaluationConfig
from wigin_tllm.types import Benchmark, BenchmarkItem, CompletionPrompt, Submission

YEARS = [2013, 2014, 2015]


@pytest.fixture
def config() -> EvaluationConfig:
    return EvaluationConfig(
        min_eval_score=-3.0,
        leak_best_score=-6.0,
        top_n_for_quality=10,
        leak_weight=0.7,
        quality_weight=0.3,
        require_pinned_revision=False,
        quality_seed=7,
        device="cpu",
    )


def make_benchmark(year: int, kind: str, n: int = 4, threshold: float = 0.10,
                   epsilon: float = -3.0) -> Benchmark:
    return Benchmark(
        year=year,
        kind=kind,
        items=[BenchmarkItem(prompt=f"{kind} prompt {i}", phrase=f"p{i}") for i in range(n)],
        threshold=threshold,
        epsilon=epsilon,
    )


@pytest.fixture
def benchmarks() -> dict[int, dict[str, Benchmark]]:
    return {
        year: {"unknown": make_benchmark(year, "unknown"), "known": make_benchmark(year, "known")}
        for year in YEARS
    }


def make_submission(submitter_id: str, submitted_at: str, ref: str | None = None,
                    years: list[int] | None = None) -> Submission:
    years = years or YEARS
    ref = ref or f"local:/models/{submitter_id}"
    return Submission(
        submitter_id=submitter_id,
        models={str(year): ref for year in years},
        submitted_at=submitted_at,
    )


@pytest.fixture
def questions() -> list[CompletionPrompt]:
    return [
        CompletionPrompt(prompt="q1", reference="alpha beta"),
        CompletionPrompt(prompt="q2", reference="gamma delta"),
        CompletionPrompt(prompt="q3", reference="epsilon zeta"),
    ]
