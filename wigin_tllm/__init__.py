"""Wigin TLLM — benchmark a language model for chronological consistency.

    from wigin_tllm import EvaluationConfig, Corpus, benchmark

    report = benchmark({"2015": "local:./my-model"}, Corpus("./corpus"))
    print(report.final_score)
"""

from .config import EvaluationConfig
from .corpus import Corpus
from .types import (
    Benchmark,
    BenchmarkItem,
    CompletionPrompt,
    Fact,
    ModelRef,
    ProbeResult,
    YearAssessment,
    YearScore,
)

__all__ = [
    "EvaluationConfig",
    "Corpus",
    "Benchmark",
    "BenchmarkItem",
    "CompletionPrompt",
    "Fact",
    "ModelRef",
    "ProbeResult",
    "YearAssessment",
    "YearScore",
    "benchmark",
]

__version__ = "0.2.0"


def benchmark(*args, **kwargs):
    """Lazy re-export of :func:`wigin_tllm.benchmark.benchmark`.

    Imported on first use so that `import wigin_tllm` stays cheap and does
    not pull in torch until a benchmark is actually requested.
    """
    from .benchmark import benchmark as _benchmark

    return _benchmark(*args, **kwargs)
