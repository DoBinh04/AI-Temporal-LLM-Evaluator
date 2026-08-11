"""Benchmarking a model, in the order an author cares about.

    artifact      will it even be scored?
    consistency   does it respect its cutoff?          (stage 1, 70%)
    similarity    is it too close to something else?
    quality       is it any good?                      (stage 2, 30%)

Each runs on its own; `benchmark()` runs them all and combines the two scored
stages into the final number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..config import EvaluationConfig
from ..corpus import Corpus
from ..models.store import get_device
from ..scoring.judge import Judge
from ..types import ModelRef, SubmissionError, validate_manifest
from . import quality as quality_stage
from . import similarity as similarity_stage
from .artifact import OK, WARN, ArtifactReport, CheckItem, inspect_model
from .consistency import ConsistencyReport, score_manifest
from .quality import QualityReport
from .similarity import SimilarityReport

logger = logging.getLogger(__name__)

__all__ = [
    "ArtifactReport",
    "CheckItem",
    "ConsistencyReport",
    "QualityReport",
    "SimilarityReport",
    "BenchmarkReport",
    "benchmark",
    "inspect_model",
    "score_manifest",
]


@dataclass
class BenchmarkReport:
    """Everything measured about one submission."""

    manifest: dict[str, str] = field(default_factory=dict)
    artifact: ArtifactReport = field(default_factory=ArtifactReport)
    consistency: Optional[ConsistencyReport] = None
    similarity: Optional[SimilarityReport] = None
    quality: Optional[QualityReport] = None
    leak_weight: float = 0.7
    quality_weight: float = 0.3

    @property
    def ok(self) -> bool:
        """True when nothing blocks the submission from being scored."""
        return self.artifact.ok and (self.similarity is None or self.similarity.ok)

    @property
    def qualified(self) -> bool:
        """Stage 2 is only open to models whose stage-1 mean clears the bar."""
        return self.consistency is not None and self.consistency.clears_threshold

    @property
    def final_score(self) -> Optional[float]:
        """The blend, once both halves have been measured.

        A submission that does not clear the stage-1 threshold scores 0.0
        outright: it never enters stage 2, and no amount of quality can buy
        back a failed consistency check.
        """
        if self.consistency is None:
            return None
        if not self.qualified:
            return 0.0
        if self.quality is None or not self.quality.measured:
            return None
        return (
            self.leak_weight * self.consistency.normalized
            + self.quality_weight * self.quality.win_rate
        )


def benchmark(
    models: dict[str, str],
    corpus: Corpus,
    config: Optional[EvaluationConfig] = None,
    years: Optional[Sequence[int]] = None,
    references: Optional[dict[int, list[str]] | Sequence[str]] = None,
    judge: Optional[Judge] = None,
    prompt_generator=None,
    skip_similarity: bool = False,
) -> BenchmarkReport:
    """Run the whole benchmark over a `{year: model_ref}` manifest.

    `references` feeds both the similarity check and the quality duels — they
    are the same published models, used for two different questions.
    """
    config = config or EvaluationConfig()
    device = get_device(config.device)
    in_scope = list(years) if years else corpus.years()

    report = BenchmarkReport(
        manifest=dict(models),
        leak_weight=config.leak_weight,
        quality_weight=config.quality_weight,
    )

    try:
        missing = validate_manifest(
            models, in_scope, require_pinned_revision=config.require_pinned_revision
        )
    except SubmissionError as e:
        report.artifact.add(CheckItem("manifest", "fail", str(e)))
        return report

    report.artifact.add(
        CheckItem(
            "manifest", WARN if missing else OK,
            f"{len(models)} of {len(in_scope)} years covered",
            f"missing years score worst-possible: {missing}" if missing else "",
        )
    )

    # Inspect one representative model: the artefact rules are per-file, and
    # checking every year's copy would repeat the same answer.
    representative = next(iter(models.values()), None)
    if representative:
        artefact, _ = inspect_model(ModelRef.parse(representative), config, device)
        for item in artefact.items:
            report.artifact.add(item)

    # The references gate stage 1: a year whose model is spectrally a copy of
    # that year's reference scores worst-possible before a probe is run.
    gate = None
    if references and not skip_similarity:
        gate = similarity_stage.build_gate(references, in_scope, config, device)

    report.consistency = score_manifest(models, corpus, config, device, in_scope, gate=gate)

    if references and not skip_similarity and representative:
        report.similarity = similarity_stage.compare(
            representative,
            similarity_stage.references_for_year(references, in_scope[0] if in_scope else None),
            config,
            device,
        )

    if not report.qualified:
        report.quality = QualityReport(
            skipped=(
                f"stage 1 not cleared (leak score {report.consistency.leak_score:.4f} "
                f"vs threshold {config.min_eval_score:.4f}) — the final score is 0"
            )
        )
    elif references:
        prompts = prompt_generator.generate(0) if prompt_generator else None
        report.quality = quality_stage.run(
            models, references, corpus, config, device, judge, prompts, in_scope
        )
    else:
        report.quality = QualityReport(skipped="no reference models given (--against)")

    return report
