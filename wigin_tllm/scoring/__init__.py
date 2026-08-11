"""The scoring core: probes, spectra, duels.

These are the measurements themselves, kept free of any workflow so the
benchmark stages can compose them.
"""

from .aggregate import mean_year_score, normalize_leak_score
from .judge import Judge, OpenAIJudge, ReferenceOverlapJudge, ScriptedJudge
from .leak import assess_year, probe, score_items
from .prompt_generator import (
    CATEGORIES,
    OpenAIPromptGenerator,
    PromptGenerator,
    StaticPromptGenerator,
)
from .quality import (
    CompletionProvider,
    ModelCompletionProvider,
    StaticCompletionProvider,
    duel,
    run_quality_duels,
    run_round_robin,
    select_eval_years,
)
from .svd_gate import SvdGate, compare_spectra, dedup_by_svd, load_state_dict, svd_spectra

__all__ = [
    "normalize_leak_score", "mean_year_score",
    "Judge", "OpenAIJudge", "ReferenceOverlapJudge", "ScriptedJudge",
    "probe", "assess_year", "score_items",
    "PromptGenerator", "OpenAIPromptGenerator", "StaticPromptGenerator", "CATEGORIES",
    "CompletionProvider", "ModelCompletionProvider", "StaticCompletionProvider",
    "duel", "run_round_robin", "run_quality_duels", "select_eval_years",
    "SvdGate", "svd_spectra", "compare_spectra", "dedup_by_svd", "load_state_dict",
]
