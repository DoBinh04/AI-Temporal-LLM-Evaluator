"""Stage 2 — is the model any good?

Stage 1 only proves a model respects its cutoff; a model can do that and
still be useless. Stage 2 has it continue a set of incomplete texts and pits
those completions against reference models under an automatic judge. The
share of duels won is the quality score.

Quality is inherently comparative — there is no absolute scale — so this
needs something to duel against. That can be the published references or any
list of models the author names.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..config import EvaluationConfig
from ..corpus import Corpus
from ..scoring.judge import Judge
from ..scoring.quality import ModelCompletionProvider, run_quality_duels
from ..types import CompletionPrompt, ModelRef

logger = logging.getLogger(__name__)

SUBJECT = "your-model"
OPPONENT_PREFIX = "ref-"


@dataclass
class QualityReport:
    win_rate: Optional[float] = None
    opponents: list[str] = field(default_factory=list)
    prompt_count: int = 0
    years: list[int] = field(default_factory=list)
    skipped: str = ""

    @property
    def measured(self) -> bool:
        return self.win_rate is not None

    @property
    def ambiguous(self) -> bool:
        """A zero rate against one opponent cannot be told from a draw.

        A drawn duel scores for neither side, so with a single reference
        "won nothing" and "drew everything" produce the same number.
        """
        return self.win_rate == 0.0 and len(self.opponents) < 2


class Submission:
    """A `{year: ref}` mapping under one name, for the duel provider."""

    def __init__(self, submitter_id: str, models: dict[str, str]):
        self.submitter_id = submitter_id
        self.models = models

    def ref_for_year(self, year) -> Optional[ModelRef]:
        raw = self.models.get(str(year))
        return ModelRef.parse(raw) if raw else None


def opponent_lineup(references: dict[int, list[str]] | Sequence[str],
                    years: Sequence[int]) -> dict[str, Submission]:
    """Turn a reference spec into named opponents the provider can serve.

    A `{year: [refs]}` mapping pairs references by position across years, so
    entry *i* of every year forms one opponent — the v1 line and the instruct
    line of a published family stay separate models rather than being mixed.
    A bare list is taken as one opponent per entry, used for every year.
    """
    if isinstance(references, dict):
        by_index: dict[int, dict[str, str]] = {}
        for year, refs in references.items():
            for index, ref in enumerate(refs):
                by_index.setdefault(index, {})[str(year)] = ref
        return {
            f"{OPPONENT_PREFIX}{index}": Submission(f"{OPPONENT_PREFIX}{index}", models)
            for index, models in sorted(by_index.items())
        }

    return {
        f"{OPPONENT_PREFIX}{index}": Submission(
            f"{OPPONENT_PREFIX}{index}", {str(year): ref for year in years}
        )
        for index, ref in enumerate(references)
    }


def run(
    models: dict[str, str],
    references: dict[int, list[str]] | Sequence[str],
    corpus: Corpus,
    config: EvaluationConfig,
    device,
    judge: Optional[Judge] = None,
    prompts: Optional[Sequence[CompletionPrompt]] = None,
    years: Optional[Sequence[int]] = None,
) -> QualityReport:
    """Duel a manifest against reference models."""
    in_scope = list(years) if years else corpus.years()

    if judge is None:
        return QualityReport(skipped="no judge configured")

    opponents = opponent_lineup(references, in_scope)
    if not opponents:
        return QualityReport(skipped="no reference models to duel against")

    prompts = list(prompts) if prompts is not None else corpus.completion_prompts()
    if not prompts:
        return QualityReport(
            skipped="no stage-2 prompts (generate them with `wigin-tllm prompts`)"
        )

    subject = Submission(SUBJECT, models)
    provider = ModelCompletionProvider(
        {SUBJECT: subject, **opponents}, device, config.quality_max_new_tokens
    )

    rates = run_quality_duels(
        [SUBJECT],
        prompts,
        in_scope,
        provider,
        judge,
        rng=random.Random(config.quality_seed),
        year_samples=min(config.quality_year_samples, len(in_scope)),
        opponent_ids=list(opponents),
    )
    return QualityReport(
        win_rate=rates[SUBJECT],
        opponents=list(opponents),
        prompt_count=len(prompts),
        years=in_scope,
    )
