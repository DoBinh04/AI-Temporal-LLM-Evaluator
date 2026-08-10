"""Wigin TLLM — chronological-consistency evaluation for language models.

    from wigin_tllm import EvaluationConfig, run_evaluation
    from wigin_tllm.datasource import LocalDataSource

    results = run_evaluation(LocalDataSource("./data"), EvaluationConfig())
"""

from .config import EvaluationConfig
from .types import (
    Benchmark,
    BenchmarkItem,
    ModelRef,
    QualityQuestion,
    RoundResults,
    Submission,
    SubmitterResult,
    YearEvaluation,
)

__all__ = [
    "EvaluationConfig",
    "Benchmark",
    "BenchmarkItem",
    "ModelRef",
    "QualityQuestion",
    "RoundResults",
    "Submission",
    "SubmitterResult",
    "YearEvaluation",
    "run_evaluation",
]

__version__ = "0.1.0"


def run_evaluation(*args, **kwargs):
    """Lazy re-export of :func:`wigin_tllm.pipeline.run_evaluation`.

    Imported on first use so that `import wigin_tllm` stays cheap and does
    not pull in torch until an evaluation is actually requested.
    """
    from .pipeline import run_evaluation as _run

    return _run(*args, **kwargs)
