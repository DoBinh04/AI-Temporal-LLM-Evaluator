"""Stage 1 — does the model respect its cutoff?

Scores a manifest of `{year: model_ref}` against a corpus, one model per
year, and aggregates exactly as a real evaluation does: the mean is taken
over **every** year in scope, so a year with no model counts worst-possible
rather than being quietly excluded.

The gauntlet each model runs, in order, is the production one:

    weight size    over `max_model_bytes`         -> its years score worst
    revision       not a real pinned commit       -> its years score worst
    parameters     over `max_parameters`          -> its years score worst
    SVD gate       spectrally a copy of a         -> that year scores worst
                   reference for that year
    time budget    `max_eval_seconds` exhausted   -> remaining years score worst
    probes         `unknown` recognised, or       -> that year scores worst
                   `known` not recognised
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..config import WORST_SCORE, EvaluationConfig
from ..corpus import Corpus
from ..models.loader import load_model
from ..models.store import (
    count_model_params,
    free_device_memory,
    get_model_size_bytes,
    resolve_model,
    verify_pinned_revision,
)
from ..scoring.leak import assess_year
from ..scoring.svd_gate import SvdGate, svd_spectra
from ..types import ModelRef, YearAssessment, YearScore

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyReport:
    years: list[YearScore] = field(default_factory=list)
    scored_years: int = 0
    total_years: int = 0
    leak_score: float = WORST_SCORE
    normalized: float = 0.0
    min_eval_score: float = -3.0
    errors: list[str] = field(default_factory=list)

    @property
    def clears_threshold(self) -> bool:
        return self.leak_score < self.min_eval_score

    @property
    def covers_all_years(self) -> bool:
        return self.scored_years == self.total_years

    @property
    def years_passed(self) -> int:
        return sum(1 for y in self.years if y.passed)

    @property
    def saturated(self) -> bool:
        """At 1.0 the leak half is maxed out; further work there earns nothing."""
        return self.normalized >= 1.0


def diagnose(assessment: YearAssessment) -> str:
    """Say *why* a year landed where it did, in terms the author can act on."""
    unknown, known = assessment.unknown, assessment.known

    if assessment.passed:
        note = "blind to the future, fluent in its own era"
        if min(unknown.margin, known.margin) < 0.05:
            note += " — but only just; one probe either way would flip it"
        return note
    if unknown.recognised and known.recognised:
        return (
            "recognises post-cutoff facts as readily as pre-cutoff ones — the "
            "training data reaches beyond the cutoff"
        )
    if unknown.recognised and not known.recognised:
        return (
            "recognises the future but not the past — check that the cutoff "
            "year on this checkpoint is the one you think it is"
        )
    if known.total == 0:
        return "no `known` probes for this year, so the requirement cannot be met"
    return (
        "recognises neither past nor future — the model has not learned its own "
        f"era, or epsilon ({known.epsilon:.2f}) is stricter than this "
        "vocabulary can reach"
    )


def group_by_ref(models: dict[str, str], years: Sequence[int]) -> dict[str, list[int]]:
    """One model often serves several years; resolve and load it once."""
    groups: dict[str, list[int]] = {}
    for year in years:
        raw_ref = models.get(str(year))
        if raw_ref:
            groups.setdefault(raw_ref, []).append(year)
    return groups


def score_manifest(
    models: dict[str, str],
    corpus: Corpus,
    config: EvaluationConfig,
    device,
    years: Optional[Sequence[int]] = None,
    gate: Optional[SvdGate] = None,
) -> ConsistencyReport:
    """Score every year in scope, each against the model assigned to it.

    `gate` is the optional per-year spectral gate: a year whose model reads
    as a copy of that year's reference scores worst-possible before a probe
    is run.
    """
    from ..scoring.aggregate import mean_year_score, normalize_leak_score

    in_scope = list(years) if years else corpus.years()
    report = ConsistencyReport(
        total_years=len(in_scope), min_eval_score=config.min_eval_score
    )

    for raw_ref, ref_years in group_by_ref(models, in_scope).items():
        report.years.extend(
            _score_ref(raw_ref, ref_years, corpus, config, device, report, gate)
        )
    report.years.sort(key=lambda y: y.year)

    report.scored_years = len(report.years)
    year_scores = {y.year: y.score for y in report.years}
    report.leak_score = mean_year_score(year_scores, in_scope)
    report.normalized = normalize_leak_score(
        report.leak_score, config.min_eval_score, config.leak_best_score
    )
    return report


def _score_ref(
    raw_ref: str,
    years: list[int],
    corpus: Corpus,
    config: EvaluationConfig,
    device,
    report: ConsistencyReport,
    gate: Optional[SvdGate] = None,
) -> list[YearScore]:
    """Score one model against the years it covers.

    A model that will not load costs exactly those years — the rest of the
    manifest is still scored.
    """
    ref = ModelRef.parse(raw_ref)

    # An unpinned reference can be repointed at different weights after the
    # fact, so it fails before anything is downloaded.
    if config.require_pinned_revision and ref.scheme == "hf" and not ref.is_pinned:
        report.errors.append(
            f"{ref.short}: not pinned to a commit SHA (expected owner/repo@<40-hex>); "
            f"years {years} score worst-possible"
        )
        return []

    size = get_model_size_bytes(ref)
    if size > config.max_model_bytes:
        report.errors.append(
            f"{ref.short}: {size / 1024**2:.1f} MB of weights exceeds the limit; "
            f"years {years} score worst-possible"
        )
        return []

    model = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = resolve_model(ref, tmpdir)

            # A branch whose name is 40 hex characters passes the format
            # check but can still be repointed — confirm it is a real commit.
            if config.require_pinned_revision and not verify_pinned_revision(ref):
                report.errors.append(
                    f"{ref.short}: revision {ref.revision!r} is not a real commit SHA; "
                    f"years {years} score worst-possible"
                )
                return []

            started = time.monotonic()
            model, _ = load_model(path, device)

            params = count_model_params(model)
            if params > config.max_parameters:
                report.errors.append(
                    f"{ref.short}: {params / 1e6:.1f}M parameters exceeds the limit; "
                    f"years {years} score worst-possible"
                )
                return []

            gate_results: dict[int, tuple[bool, float]] = {}
            if gate:
                candidate_spectra = svd_spectra(model.inner_state_dict())
                for year in years:
                    gate_results[year] = gate.check(candidate_spectra, year)
                    passed, distance = gate_results[year]
                    logger.info(
                        f"{ref.short}: year {year} svd_dist={distance:.6f} "
                        f"gate={'PASS' if passed else 'FAIL'}"
                    )
                del candidate_spectra

            scored = []
            for index, year in enumerate(years):
                if (
                    config.max_eval_seconds is not None
                    and time.monotonic() - started > config.max_eval_seconds
                ):
                    report.errors.append(
                        f"{ref.short}: time budget ({config.max_eval_seconds:.0f}s) "
                        f"exhausted; years {years[index:]} score worst-possible"
                    )
                    break

                gate_passed, gate_distance = gate_results.get(year, (True, 1.0))
                if not gate_passed:
                    scored.append(
                        YearScore(
                            year=year, model_ref=raw_ref,
                            diagnosis=(
                                f"spectral distance {gate_distance:.6f} to a reference "
                                f"model is below {gate.threshold} — scored as a copy"
                            ),
                        )
                    )
                    continue

                logger.info(f"Scoring {year} with {ref.short}...")
                unknown, known = corpus.probes_for(year)
                assessment = assess_year(model, device, unknown, known)
                scored.append(
                    YearScore(
                        year=year, model_ref=raw_ref,
                        assessment=assessment, diagnosis=diagnose(assessment),
                    )
                )
            return scored
    except Exception as e:
        report.errors.append(
            f"{ref.short}: {type(e).__name__}: {e}; years {years} score worst-possible"
        )
        return []
    finally:
        del model
        free_device_memory()
