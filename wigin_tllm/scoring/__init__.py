"""The scoring core: probes, spectra, duels.

These are the measurements themselves, kept free of any workflow so the
benchmark stages can compose them.
"""

from .aggregate import mean_year_score, normalize_leak_score
from .judge import Judge, OpenAIJudge, ReferenceOverlapJudge, ScriptedJudge
from .leak import assess_year, probe, score_items, score_one
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
from .svd_gate import compare_spectra, load_state_dict, svd_spectra

__all__ = [
    "normalize_leak_score", "mean_year_score",
    "Judge", "OpenAIJudge", "ReferenceOverlapJudge", "ScriptedJudge",
    "probe", "assess_year", "score_items", "score_one",
    "PromptGenerator", "OpenAIPromptGenerator", "StaticPromptGenerator", "CATEGORIES",
    "CompletionProvider", "ModelCompletionProvider", "StaticCompletionProvider",
    "duel", "run_round_robin", "run_quality_duels", "select_eval_years",
    "svd_spectra", "compare_spectra", "load_state_dict",
]
