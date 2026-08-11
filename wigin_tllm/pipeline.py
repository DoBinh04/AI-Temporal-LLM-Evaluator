"""The evaluation pipeline.

    submissions -> stage 1 (consistency) -> dedup -> stage 2 (quality)
                -> rank -> persist

This module only sequences the work, enforces the resource limits, and
decides what happens when something fails. Every formula lives in
:mod:`wigin_tllm.scoring`.
"""

from __future__ import annotations

import logging
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .config import WORST_SCORE, EvaluationConfig
from .datasource.base import DataSource
from .models.loader import ModelLoadError, load_model
from .models.store import (
    count_model_params,
    free_device_memory,
    get_device,
    get_model_size_bytes,
    resolve_model,
    verify_pinned_revision,
    weight_hash,
)
from .scoring.aggregate import build_round_results, mean_year_score, qualify, rank
from .scoring.judge import Judge
from .scoring.leak import assess_year
from .scoring.prompt_generator import PromptGenerator
from .scoring.quality import ModelCompletionProvider, run_quality_duels
from .scoring.svd_gate import SvdGate, dedup_by_svd, svd_spectra
from .storage import EvaluationStore
from .types import (
    Benchmark,
    ConsistencyResults,
    ModelRef,
    RoundResults,
    Submission,
    YearEvaluation,
)

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Infrastructure failure that makes the round's results untrustworthy."""


@dataclass
class _Tally:
    """Per-submitter accumulator while stage 1 runs."""

    year_scores: dict[int, float] = field(default_factory=dict)
    years_passed: int = 0
    failure: str = ""

    def note_failure(self, reason: str) -> None:
        """Keep the first failure; later ones are usually consequences."""
        if not self.failure:
            self.failure = reason


# ─── stage 1: chronological consistency ──────────────────────────────────


class ConsistencyStage:
    """Scores every submission for chronological consistency.

    Submissions are processed oldest-first, which is what makes the anti-copy
    rules work without any tie-breaking of their own: the earliest submitter
    of a set of weights always reaches the registry before a copier.
    """

    def __init__(
        self,
        datasource: DataSource,
        store: EvaluationStore,
        config: EvaluationConfig,
        round_id: int,
        years: Sequence[int],
        benchmarks: dict[int, dict[str, Benchmark]],
        gate: SvdGate,
        device,
    ):
        self.datasource = datasource
        self.store = store
        self.config = config
        self.round_id = round_id
        self.years = list(years)
        self.benchmarks = benchmarks
        self.gate = gate
        self.device = device

    # ── entry point ──────────────────────────────────────────────────────

    def run(self, submissions: Sequence[Submission]) -> ConsistencyResults:
        results = ConsistencyResults()
        ordered = sorted(submissions, key=lambda s: s.sort_key)

        for index, submission in enumerate(ordered, start=1):
            logger.info(f"--- {submission.submitter_id}: {len(submission.models)} years submitted ---")
            tally = self._evaluate(submission)

            if tally.years_passed == 0:
                results.disqualified[submission.submitter_id] = (
                    tally.failure or "failed_consistency_check"
                )
            results.year_scores[submission.submitter_id] = tally.year_scores
            results.leak_scores[submission.submitter_id] = mean_year_score(
                tally.year_scores, self.years
            )
            logger.info(
                f"{submission.submitter_id}: leak_score="
                f"{results.leak_scores[submission.submitter_id]:.4f} "
                f"[{index}/{len(ordered)}]"
            )

        return results

    # ── per submission ───────────────────────────────────────────────────

    def _evaluate(self, submission: Submission) -> _Tally:
        tally = _Tally(year_scores={year: WORST_SCORE for year in self.years})
        for raw_ref, years in self._plan(submission, tally).items():
            self._score_group(submission, raw_ref, years, tally)
        return tally

    def _plan(self, submission: Submission, tally: _Tally) -> dict[str, list[int]]:
        """Fill in cached years; group the rest by model reference.

        One model often serves several years, and should only be resolved and
        loaded once.
        """
        groups: dict[str, list[int]] = {}
        cached = 0

        for year in self.years:
            raw_ref = submission.models.get(str(year))
            if not raw_ref:
                continue
            hit = self.store.get_cached_evaluation(
                submission.submitter_id, year, raw_ref, self.round_id
            )
            if hit is None:
                groups.setdefault(raw_ref, []).append(year)
            else:
                tally.year_scores[year] = hit.score
                tally.years_passed += int(hit.passed)
                cached += 1

        if cached:
            logger.info(f"{submission.submitter_id}: {cached} year scores loaded from cache")
        return groups

    # ── per model reference ──────────────────────────────────────────────

    def _score_group(
        self, submission: Submission, raw_ref: str, years: list[int], tally: _Tally
    ) -> None:
        ref = ModelRef.parse(raw_ref)
        who = submission.submitter_id

        size = get_model_size_bytes(ref)
        if size > self.config.max_model_bytes:
            logger.warning(f"{who}: {raw_ref} is {size} bytes, over limit")
            return self._reject(submission, raw_ref, years, "model_too_large", tally)

        model = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                logger.info(f"{who}: resolving {ref.location}...")
                path = resolve_model(ref, tmpdir)

                rejection = self._inspect_artifact(submission, ref, path)
                if rejection:
                    return self._reject(submission, raw_ref, years, rejection, tally)

                started = time.time()
                model, _ = load_model(path, self.device)

                params = count_model_params(model)
                logger.info(f"{who}: loaded {params / 1e6:.0f}M params")
                if params > self.config.max_parameters:
                    logger.warning(f"{who}: {params / 1e9:.2f}B params over limit")
                    return self._reject(submission, raw_ref, years, "too_many_parameters", tally)

                gate_verdicts = self._run_svd_gate(submission, years, model)
                self._score_years(submission, raw_ref, years, model, gate_verdicts, started, tally)
                logger.info(f"{who}: {raw_ref} done in {time.time() - started:.0f}s")

        except ModelLoadError as e:
            # The submitted artefact is unusable — the submitter's problem.
            logger.error(f"{who}: {raw_ref} could not be loaded — {e}")
            self._reject(submission, raw_ref, years, "model_load_failed", tally)
        except (RuntimeError, MemoryError) as e:
            # Our infrastructure broke (OOM, exhausted download retries).
            # Recording a failure would penalise a submitter for our outage,
            # so abort the round instead.
            raise EvaluationError(
                f"Aborting round at {who}/{raw_ref}: {type(e).__name__}: {e}"
            ) from e
        except Exception as e:
            logger.error(f"{who}: {raw_ref} FAILED — {type(e).__name__}: {e}")
            self._reject(submission, raw_ref, years, f"error:{type(e).__name__}", tally)
        finally:
            del model
            free_device_memory()

    def _inspect_artifact(
        self, submission: Submission, ref: ModelRef, path: str
    ) -> Optional[str]:
        """Checks that need the files but not the loaded model.

        Returns a rejection reason, or None to continue.
        """
        who = submission.submitter_id

        if self.config.require_pinned_revision and not verify_pinned_revision(ref):
            logger.warning(f"{who}: {ref} is not pinned to a real commit SHA")
            return "revision_not_pinned"

        if self.config.duplicate_weight_check:
            allowed, owner = self.store.claim_weight_hash(
                weight_hash(path), who, submission.submitted_at, self.round_id
            )
            if not allowed:
                logger.warning(f"{who}: duplicate weights (owner: {owner})")
                return "duplicate_weights"

        return None

    def _run_svd_gate(
        self, submission: Submission, years: list[int], model
    ) -> dict[int, bool]:
        """Compare the model's spectrum against the baselines, per year.

        Also persists the spectrum for the pairwise dedup pass, which needs
        every submitter's spectrum at once and so cannot run inline.
        """
        who = submission.submitter_id
        if who in self.config.svd_exempt_submitters:
            return {}
        if not (self.config.svd_gate_enabled or self.config.svd_dedup_enabled):
            return {}

        spectra = svd_spectra(model.inner_state_dict())
        verdicts: dict[int, bool] = {}

        if self.config.svd_gate_enabled:
            for year in years:
                passed, distance = self.gate.check(spectra, year)
                verdicts[year] = passed
                logger.info(
                    f"{who}: year {year} svd_dist={distance:.6f} "
                    f"gate={'PASS' if passed else 'FAIL'}"
                )

        if self.config.svd_dedup_enabled:
            self.store.save_spectra(who, self.round_id, spectra)

        return verdicts

    # ── per year ─────────────────────────────────────────────────────────

    def _score_years(
        self,
        submission: Submission,
        raw_ref: str,
        years: list[int],
        model,
        gate_verdicts: dict[int, bool],
        started: float,
        tally: _Tally,
    ) -> None:
        who = submission.submitter_id

        for year in years:
            if time.time() - started > self.config.max_eval_seconds:
                # Timed-out years stay at WORST_SCORE but are *not* recorded:
                # an uncached year is retried on the next run, exactly as a
                # crashed run would retry it.
                logger.warning(f"{who}: eval budget exhausted, remaining years skipped")
                tally.note_failure("eval_timeout")
                return

            if not gate_verdicts.get(year, True):
                self._reject(submission, raw_ref, [year], "svd_gate_failed", tally)
                continue

            logger.info(f"{who}: evaluating year {year}...")
            assessment = assess_year(
                model, self.device, self.benchmarks[year]["unknown"], self.benchmarks[year]["known"]
            )

            tally.year_scores[year] = assessment.score
            tally.years_passed += int(assessment.passed)
            if not assessment.passed:
                tally.note_failure("failed_consistency_check")

            self._record(
                YearEvaluation(
                    submitter_id=who,
                    year=year,
                    model_ref=raw_ref,
                    passed=assessment.passed,
                    score=assessment.score,
                    score_unknown=assessment.unknown.median,
                    score_known=assessment.known.median,
                    reason="" if assessment.passed else "failed_consistency_check",
                )
            )
            logger.info(
                f"{who}: year {year} {'PASSED' if assessment.passed else 'FAILED'} "
                f"(unknown={assessment.unknown.median:.4f} "
                f"known={assessment.known.median:.4f} score={assessment.score:.4f})"
            )

    # ── outcomes ─────────────────────────────────────────────────────────

    def _reject(
        self,
        submission: Submission,
        raw_ref: str,
        years: Sequence[int],
        reason: str,
        tally: _Tally,
    ) -> None:
        """Score the given years worst-possible and record why."""
        tally.note_failure(reason)
        for year in years:
            self._record(
                YearEvaluation(
                    submitter_id=submission.submitter_id,
                    year=year,
                    model_ref=raw_ref,
                    passed=False,
                    score=WORST_SCORE,
                    reason=reason,
                )
            )

    def _record(self, evaluation: YearEvaluation) -> None:
        """Persist locally first, then try to forward.

        A sink that is temporarily unreachable delays reporting rather than
        losing it: unsynced rows are retried on the next run.
        """
        self.store.save_evaluation(self.round_id, evaluation, synced=False)
        try:
            if self.datasource.save_year_evaluation(self.round_id, evaluation):
                self.store.mark_synced(self.round_id, evaluation.submitter_id, evaluation.year)
        except Exception as e:
            logger.warning(
                f"Could not forward {evaluation.submitter_id}/{evaluation.year}: {type(e).__name__}"
            )


# ─── orchestration ───────────────────────────────────────────────────────


def run_evaluation(
    datasource: DataSource,
    config: Optional[EvaluationConfig] = None,
    round_id: Optional[int] = None,
    judge: Optional[Judge] = None,
    svd_gate: Optional[SvdGate] = None,
    store: Optional[EvaluationStore] = None,
    submitter_filter: Optional[Sequence[str]] = None,
    prompt_generator: Optional[PromptGenerator] = None,
    force: bool = False,
) -> RoundResults:
    """Run one full evaluation round and return its results.

    A completed round is returned from cache unless `force=True`, so a repeat
    invocation is idempotent rather than a re-scoring.
    """
    config = config or EvaluationConfig()
    owns_store = store is None
    store = store or EvaluationStore(os.path.join(config.data_dir, "evaluations.db"))

    try:
        round_id = round_id if round_id is not None else datasource.get_current_round()
        years = datasource.get_years(round_id)
        if not years:
            raise EvaluationError(f"Round {round_id} has no years in scope")
        logger.info(f"Round {round_id} | years {years[0]}-{years[-1]} ({len(years)})")

        _flush_pending(datasource, store)

        # A completed round is served from cache; `force` wipes it instead.
        if force:
            store.clear_round(round_id)
        elif store.is_round_complete(round_id):
            payload = store.get_round_results(round_id)
            if payload is not None:
                logger.info(f"Round {round_id} already evaluated — returning cached results")
                return RoundResults.from_dict(payload)
            logger.info(f"Round {round_id} marked complete but no results stored; re-running")

        submissions = datasource.get_submissions(round_id)
        if submitter_filter:
            wanted = set(submitter_filter)
            submissions = [s for s in submissions if s.submitter_id in wanted]
        if not submissions:
            logger.info(f"Round {round_id}: no submissions")
            return _finish(datasource, store, round_id, RoundResults(round_id=round_id))
        logger.info(f"Round {round_id}: {len(submissions)} submissions")

        # Fetch every probe set up front: discovering a broken data source
        # after downloading dozens of models wastes hours.
        benchmarks = datasource.preload_benchmarks(years)
        logger.info(f"Loaded probe sets for {len(benchmarks)} years")

        device = get_device(config.device)
        gate = svd_gate if svd_gate is not None else SvdGate(
            threshold=config.svd_threshold,
            top_ratio=config.svd_top_ratio,
            cache_dir=os.path.join(config.data_dir, "baselines"),
        )
        if config.svd_gate_enabled:
            gate.load(years, device)

        logger.info("=== Stage 1: chronological consistency ===")
        consistency = ConsistencyStage(
            datasource, store, config, round_id, years, benchmarks, gate, device
        ).run(submissions)
        gate.unload()

        _apply_dedup(store, config, round_id, submissions, consistency, device)

        qualification = qualify(consistency.eligible, config)
        win_rates = _run_quality_stage(
            datasource, submissions, qualification.ids, years, config, judge, device,
            prompt_generator, round_id,
        )
        ranking = rank(qualification, win_rates, config)

        for position, (submitter_id, score) in enumerate(ranking.ordered, start=1):
            logger.info(f"  {position}. {submitter_id}: final={score:.4f}")
        if not ranking.ordered:
            logger.info("No submission passed this round")

        results = build_round_results(round_id, consistency, qualification, ranking)
        return _finish(datasource, store, round_id, results)

    finally:
        if owns_store:
            store.close()


# ─── steps ───────────────────────────────────────────────────────────────


def _flush_pending(datasource: DataSource, store: EvaluationStore) -> None:
    """Forward results the data source has not accepted yet, oldest first."""
    pending = store.unsynced_evaluations()
    if not pending:
        return

    logger.info(f"Syncing {len(pending)} pending evaluations...")
    synced = 0
    for round_id, evaluation in pending:
        try:
            # A rejected row stays pending and is retried next run; the rest
            # of the batch still gets its chance.
            if datasource.save_year_evaluation(round_id, evaluation):
                store.mark_synced(round_id, evaluation.submitter_id, evaluation.year)
                synced += 1
        except Exception as e:
            # An exception means the sink itself is down — stop hammering it.
            logger.warning(
                f"Sync failed for {evaluation.submitter_id}/{evaluation.year}: {type(e).__name__}"
            )
            break
    logger.info(f"Sync complete: {synced}/{len(pending)}")


def _apply_dedup(
    store: EvaluationStore,
    config: EvaluationConfig,
    round_id: int,
    submissions: Sequence[Submission],
    consistency: ConsistencyResults,
    device,
) -> None:
    """Drop submitters whose spectrum matches an earlier submitter's."""
    if not config.svd_dedup_enabled:
        return

    spectra = store.load_spectra(round_id, device)
    if not spectra:
        return

    accepted = dedup_by_svd(
        spectra,
        {s.submitter_id: s.submitted_at for s in submissions},
        threshold=config.svd_threshold,
        top_ratio=config.svd_top_ratio,
    )
    for submitter_id in consistency.leak_scores:
        if submitter_id in spectra and submitter_id not in accepted:
            consistency.disqualify(
                submitter_id, "duplicate_of_earlier_submission", WORST_SCORE
            )


def _stage2_prompts(
    datasource: DataSource,
    generator: Optional[PromptGenerator],
    round_id: int,
) -> list:
    """Get this round's prompts, generating them if a generator is configured.

    Prompts are generated fresh per round rather than read from a fixed file:
    a static set would be learnable, and a model tuned to it would score well
    without being better. A configured generator returning nothing is fatal —
    silently falling back to no stage 2 would quietly change what the round
    measures.
    """
    if generator is None:
        return datasource.get_completion_prompts()

    prompts = generator.generate(round_id)
    if not prompts:
        raise EvaluationError("Prompt generation produced nothing; aborting round")
    return prompts


def _run_quality_stage(
    datasource: DataSource,
    submissions: Sequence[Submission],
    entrants: Sequence[str],
    years: Sequence[int],
    config: EvaluationConfig,
    judge: Optional[Judge],
    device,
    generator: Optional[PromptGenerator] = None,
    round_id: int = 0,
) -> Optional[dict[str, float]]:
    """Run quality duels, or return None when there is no usable signal."""
    if not config.quality_enabled:
        logger.info("Stage 2 disabled by config")
        return None
    if len(entrants) < 2:
        logger.info("Fewer than 2 qualified submitters, skipping stage 2")
        return None
    if judge is None:
        logger.warning("No judge configured, skipping stage 2")
        return None

    prompts = _stage2_prompts(datasource, generator, round_id)
    if not prompts:
        logger.warning("No stage-2 prompts available, skipping stage 2")
        return None

    logger.info(f"=== Stage 2: quality duels over {len(prompts)} prompts ===")
    provider = ModelCompletionProvider(
        {s.submitter_id: s for s in submissions}, device, config.quality_max_new_tokens
    )
    return run_quality_duels(
        entrants,
        prompts,
        years,
        provider,
        judge,
        rng=random.Random(config.quality_seed),
        year_samples=config.quality_year_samples,
    )


def _finish(
    datasource: DataSource, store: EvaluationStore, round_id: int, results: RoundResults
) -> RoundResults:
    store.save_round_results(round_id, results.to_dict())
    try:
        datasource.save_results(round_id, results)
    except Exception as e:
        logger.error(f"Could not publish round results: {type(e).__name__}: {e}")
    store.mark_round_complete(round_id)
    _flush_pending(datasource, store)
    return results
