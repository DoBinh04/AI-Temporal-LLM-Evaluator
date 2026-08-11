"""Pipeline integration.

The model layer is stubbed out so these tests exercise orchestration —
ordering, gating, caching, failure handling — without loading real weights.
Real weights are covered by `test_end_to_end.py`.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from wigin_tllm.datasource import InMemoryDataSource  # noqa: E402
from wigin_tllm.models.loader import ModelLoadError  # noqa: E402
from wigin_tllm.pipeline import EvaluationError, run_evaluation  # noqa: E402
from wigin_tllm.scoring.judge import ScriptedJudge  # noqa: E402
from wigin_tllm.scoring.svd_gate import SvdGate  # noqa: E402
from wigin_tllm.types import ProbeResult, CompletionPrompt, YearAssessment  # noqa: E402

from conftest import YEARS, make_submission  # noqa: E402

CLEAN = (True, -6.0, -8.0, -2.0)   # passes, wide gap
LEAKY = (False, 0.0, -2.0, -2.0)   # recognises the future


def assessment_of(passed, score, median_unknown, median_known) -> YearAssessment:
    def result(kind, median):
        return ProbeResult(kind=kind, median=median, above_epsilon=0, total=8,
                           epsilon=-11.51, threshold=0.10)

    return YearAssessment(
        unknown=result("unknown", median_unknown),
        known=result("known", median_known),
        passed=passed,
        score=score,
    )


class FakeModel:
    def __init__(self, ref: str):
        self.ref = ref

    def inner_state_dict(self):
        return {"tag": self.ref}

    def parameters(self):
        return iter([])


def submitter_of(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def distinct_spectrum(submitter_id: str) -> dict:
    """A stable, submitter-specific spectrum so unrelated submitters never look alike.

    Eight values, because the gate compares only the top `svd_top_ratio` of a
    spectrum — with fewer than eight there is nothing left to compare.
    """
    seed = sum(ord(c) for c in submitter_id)
    generator = torch.Generator().manual_seed(seed)
    return {"w": torch.rand(8, generator=generator) * 10}


class Stack:
    """Controls what the stubbed model layer does."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.default_score = CLEAN
        self.scores: dict[tuple[str, int], tuple] = {}
        self.params = 1_000_000
        self.size = 1_000
        self.hashes: dict[str, str] = {}
        self.spectra: dict[str, dict] = {}
        self.resolve_errors: dict[str, Exception] = {}
        self.load_errors: dict[str, Exception] = {}
        self.pinned = True
        self.resolve_order: list[str] = []
        self.scored: list[tuple[str, int]] = []
        self._dirs: dict[str, str] = {}

    def score_for(self, ref: str, year: int) -> tuple:
        return self.scores.get((submitter_of(ref), year), self.default_score)


@pytest.fixture
def stack(monkeypatch, tmp_path):
    s = Stack(tmp_path)

    def resolve_model(ref, dest_dir):
        raw = ref.raw or str(ref)
        s.resolve_order.append(submitter_of(raw))
        if submitter_of(raw) in s.resolve_errors:
            raise s.resolve_errors[submitter_of(raw)]
        path = os.path.join(tmp_path, "models", raw.replace("/", "_").replace(":", "_"))
        os.makedirs(path, exist_ok=True)
        s._dirs[path] = raw
        return path

    def load_model(path, device):
        ref = s._dirs[path]
        if submitter_of(ref) in s.load_errors:
            raise s.load_errors[submitter_of(ref)]
        return FakeModel(ref), None

    def assess_year(model, device, unknown, known):
        s.scored.append((submitter_of(model.ref), unknown.year))
        return assessment_of(*s.score_for(model.ref, unknown.year))

    def svd_spectra(state_dict):
        submitter = submitter_of(state_dict["tag"])
        return s.spectra.get(submitter, distinct_spectrum(submitter))

    monkeypatch.setattr("wigin_tllm.pipeline.resolve_model", resolve_model)
    monkeypatch.setattr("wigin_tllm.pipeline.get_model_size_bytes", lambda ref: s.size)
    monkeypatch.setattr("wigin_tllm.pipeline.weight_hash",
                        lambda path: s.hashes.get(submitter_of(s._dirs[path]), s._dirs[path]))
    monkeypatch.setattr("wigin_tllm.pipeline.count_model_params", lambda m: s.params)
    monkeypatch.setattr("wigin_tllm.pipeline.verify_pinned_revision", lambda ref: s.pinned)
    monkeypatch.setattr("wigin_tllm.pipeline.free_device_memory", lambda: None)
    monkeypatch.setattr("wigin_tllm.pipeline.get_device", lambda pref=None: torch.device("cpu"))
    monkeypatch.setattr("wigin_tllm.pipeline.load_model", load_model)
    monkeypatch.setattr("wigin_tllm.pipeline.assess_year", assess_year)
    monkeypatch.setattr("wigin_tllm.pipeline.svd_spectra", svd_spectra)
    return s


def source(submissions, benchmarks, prompts=None):
    return InMemoryDataSource(
        years=YEARS, benchmarks=benchmarks, submissions=submissions,
        prompts=prompts or [], current_round=1,
    )


def run(src, config, tmp_path, **kwargs):
    config.data_dir = str(tmp_path / "state")
    kwargs.setdefault("svd_gate", SvdGate(baselines={}))
    return run_evaluation(src, config=config, **kwargs)


def top_of(results):
    """The best-ranked submitter, or None if nothing ranked."""
    ranked = results.ranked
    return ranked[0].submitter_id if ranked else None


def result_for(results, submitter_id):
    return next(m for m in results.submitters if m.submitter_id == submitter_id)


# ─── happy path ──────────────────────────────────────────────────────────


def test_clean_submission_qualifies_and_wins(stack, config, benchmarks, tmp_path):
    subs = [make_submission("alice", "2026-07-01T08:00:00")]
    results = run(source(subs, benchmarks), config, tmp_path)

    alice = result_for(results, "alice")
    assert alice.qualified
    assert alice.leak_score == -6.0
    assert top_of(results) == "alice"
    assert alice.rank == 1


def test_every_year_is_scored(stack, config, benchmarks, tmp_path):
    subs = [make_submission("alice", "2026-07-01T08:00:00")]
    run(source(subs, benchmarks), config, tmp_path)
    assert sorted(y for _, y in stack.scored) == YEARS


def test_results_reach_the_data_source(stack, config, benchmarks, tmp_path):
    src = source([make_submission("alice", "2026-07-01T08:00:00")], benchmarks)
    run(src, config, tmp_path)
    assert top_of(src.saved_results[1]) == "alice"
    assert len(src.saved_evaluations) == len(YEARS)


def test_no_submissions_yields_an_empty_round(stack, config, benchmarks, tmp_path):
    results = run(source([], benchmarks), config, tmp_path)
    assert top_of(results) is None
    assert results.submitters == []
    assert results.qualified == []


# ─── ordering ────────────────────────────────────────────────────────────


def test_earliest_submitter_is_evaluated_first(stack, config, benchmarks, tmp_path):
    """Order decides who claims a set of weights, so it must follow the clock."""
    subs = [
        make_submission("late", "2026-07-01T18:00:00"),
        make_submission("early", "2026-07-01T06:00:00"),
        make_submission("middle", "2026-07-01T12:00:00"),
    ]
    run(source(subs, benchmarks), config, tmp_path)
    assert stack.resolve_order == ["early", "middle", "late"]


# ─── anti-copy ───────────────────────────────────────────────────────────


def test_identical_weights_are_rejected_for_the_later_submitter(stack, config, benchmarks, tmp_path):
    stack.hashes = {"original": "same-bytes", "copy": "same-bytes"}
    subs = [
        make_submission("original", "2026-07-01T08:00:00"),
        make_submission("copy", "2026-07-01T09:00:00"),
    ]
    results = run(source(subs, benchmarks), config, tmp_path)

    assert result_for(results, "original").qualified
    copy = result_for(results, "copy")
    assert not copy.qualified
    assert copy.leak_score == 0.0
    assert copy.disqualified_reason == "duplicate_weights"


def test_near_identical_spectra_are_deduplicated(stack, config, benchmarks, tmp_path):
    """Different bytes, same spectrum — the hash registry cannot see this."""
    shared = distinct_spectrum("shared-weights")
    stack.spectra = {"original": shared, "mimic": {"w": shared["w"] + 1e-9}}
    subs = [
        make_submission("original", "2026-07-01T08:00:00"),
        make_submission("mimic", "2026-07-01T09:00:00"),
    ]
    results = run(source(subs, benchmarks), config, tmp_path)

    assert result_for(results, "original").qualified
    mimic = result_for(results, "mimic")
    assert not mimic.qualified
    assert mimic.disqualified_reason == "duplicate_of_earlier_submission"


def test_dedup_can_be_disabled(stack, config, benchmarks, tmp_path):
    shared = distinct_spectrum("shared-weights")
    stack.spectra = {"a": shared, "b": dict(shared)}
    config.svd_dedup_enabled = False
    subs = [make_submission("a", "2026-07-01T08:00:00"), make_submission("b", "2026-07-01T09:00:00")]
    results = run(source(subs, benchmarks), config, tmp_path)
    assert result_for(results, "a").qualified and result_for(results, "b").qualified


def test_baseline_gate_rejects_a_copy_of_the_baseline(stack, config, benchmarks, tmp_path):
    baseline = distinct_spectrum("published-baseline")
    stack.spectra = {"plagiarist": baseline, "honest": distinct_spectrum("honest")}

    gate = SvdGate(baselines={y: ["local:/baseline"] for y in YEARS})
    gate._spectra = {y: [baseline] for y in YEARS}

    subs = [
        make_submission("honest", "2026-07-01T08:00:00"),
        make_submission("plagiarist", "2026-07-01T09:00:00"),
    ]
    results = run(source(subs, benchmarks), config, tmp_path, svd_gate=gate)

    assert result_for(results, "honest").qualified
    plagiarist = result_for(results, "plagiarist")
    assert not plagiarist.qualified
    assert plagiarist.disqualified_reason == "svd_gate_failed"


def test_exempt_submitters_skip_the_baseline_gate(stack, config, benchmarks, tmp_path):
    baseline = distinct_spectrum("published-baseline")
    stack.spectra = {"publisher": baseline}
    config.svd_exempt_submitters = ["publisher"]

    gate = SvdGate(baselines={y: ["local:/baseline"] for y in YEARS})
    gate._spectra = {y: [baseline] for y in YEARS}

    subs = [make_submission("publisher", "2026-07-01T08:00:00")]
    results = run(source(subs, benchmarks), config, tmp_path, svd_gate=gate)
    assert result_for(results, "publisher").qualified


def test_unpinned_revision_is_rejected(stack, config, benchmarks, tmp_path):
    stack.pinned = False
    config.require_pinned_revision = True
    subs = [make_submission("alice", "2026-07-01T08:00:00")]
    results = run(source(subs, benchmarks), config, tmp_path)
    assert result_for(results, "alice").disqualified_reason == "revision_not_pinned"


# ─── resource limits ─────────────────────────────────────────────────────


def test_oversized_model_is_skipped_before_download(stack, config, benchmarks, tmp_path):
    stack.size = config.max_model_bytes + 1
    subs = [make_submission("whale", "2026-07-01T08:00:00")]
    results = run(source(subs, benchmarks), config, tmp_path)

    assert stack.resolve_order == []  # never downloaded
    assert result_for(results, "whale").disqualified_reason == "model_too_large"


def test_parameter_limit_is_enforced_after_load(stack, config, benchmarks, tmp_path):
    stack.params = config.max_parameters + 1
    subs = [make_submission("whale", "2026-07-01T08:00:00")]
    results = run(source(subs, benchmarks), config, tmp_path)
    assert result_for(results, "whale").disqualified_reason == "too_many_parameters"
    assert stack.scored == []


def test_exhausted_time_budget_stops_scoring(stack, config, benchmarks, tmp_path):
    config.max_eval_seconds = -1  # already over budget on the first check
    subs = [make_submission("slow", "2026-07-01T08:00:00")]
    results = run(source(subs, benchmarks), config, tmp_path)
    assert stack.scored == []
    assert result_for(results, "slow").disqualified_reason == "eval_timeout"


# ─── scoring outcomes ────────────────────────────────────────────────────


def test_leaking_model_does_not_qualify(stack, config, benchmarks, tmp_path):
    stack.scores = {("mallory", y): LEAKY for y in YEARS}
    subs = [make_submission("mallory", "2026-07-01T08:00:00")]
    results = run(source(subs, benchmarks), config, tmp_path)

    mallory = result_for(results, "mallory")
    assert not mallory.qualified
    assert mallory.leak_score == 0.0
    assert mallory.disqualified_reason == "failed_consistency_check"
    assert top_of(results) is None


def test_missing_years_are_scored_worst(stack, config, benchmarks, tmp_path):
    partial = make_submission("partial", "2026-07-01T08:00:00", years=[YEARS[0]])
    results = run(source([partial], benchmarks), config, tmp_path)
    submitter = result_for(results, "partial")
    assert submitter.year_scores[YEARS[0]] == -6.0
    assert submitter.year_scores[YEARS[1]] == 0.0
    assert submitter.leak_score == pytest.approx(-2.0)  # -6 / 3 years
    assert not submitter.qualified  # -2.0 is above the -3.0 threshold


def test_a_model_serving_several_years_is_loaded_once(stack, config, benchmarks, tmp_path):
    subs = [make_submission("alice", "2026-07-01T08:00:00")]  # same ref for every year
    run(source(subs, benchmarks), config, tmp_path)
    assert stack.resolve_order == ["alice"]
    assert len(stack.scored) == len(YEARS)


# ─── failure handling ────────────────────────────────────────────────────


def test_unloadable_model_fails_only_that_submitter(stack, config, benchmarks, tmp_path):
    stack.load_errors = {"broken": ModelLoadError("no config.json")}
    subs = [
        make_submission("broken", "2026-07-01T08:00:00"),
        make_submission("alice", "2026-07-01T09:00:00"),
    ]
    results = run(source(subs, benchmarks), config, tmp_path)

    assert result_for(results, "broken").disqualified_reason == "model_load_failed"
    assert result_for(results, "alice").qualified


def test_arbitrary_submitter_error_fails_only_that_submitter(stack, config, benchmarks, tmp_path):
    stack.load_errors = {"broken": ValueError("corrupt safetensors")}
    subs = [
        make_submission("broken", "2026-07-01T08:00:00"),
        make_submission("alice", "2026-07-01T09:00:00"),
    ]
    results = run(source(subs, benchmarks), config, tmp_path)
    assert result_for(results, "broken").disqualified_reason == "error:ValueError"
    assert result_for(results, "alice").qualified


def test_infrastructure_failure_aborts_the_round(stack, config, benchmarks, tmp_path):
    """An outage on our side must not be recorded as a submitter's failure."""
    stack.resolve_errors = {"alice": RuntimeError("download failed after 5 attempts")}
    subs = [make_submission("alice", "2026-07-01T08:00:00")]
    with pytest.raises(EvaluationError, match="Aborting round"):
        run(source(subs, benchmarks), config, tmp_path)


# ─── caching and idempotency ─────────────────────────────────────────────


def test_completed_round_is_not_re_scored(stack, config, benchmarks, tmp_path):
    src = source([make_submission("alice", "2026-07-01T08:00:00")], benchmarks)
    first = run(src, config, tmp_path)
    stack.scored.clear()
    second = run(src, config, tmp_path)

    assert stack.scored == []
    assert top_of(second) == top_of(first)
    assert [s.final_score for s in second.submitters] == [s.final_score for s in first.submitters]


def test_force_re_evaluates(stack, config, benchmarks, tmp_path):
    src = source([make_submission("alice", "2026-07-01T08:00:00")], benchmarks)
    run(src, config, tmp_path)
    stack.scored.clear()
    run(src, config, tmp_path, force=True)
    assert len(stack.scored) == len(YEARS)


def test_year_scores_resume_from_cache_after_an_interruption(stack, config, benchmarks, tmp_path):
    """A crashed round must not re-score the years it already finished."""
    from wigin_tllm.storage import EvaluationStore

    src = source([make_submission("alice", "2026-07-01T08:00:00")], benchmarks)
    config.data_dir = str(tmp_path / "state")
    run_evaluation(src, config=config, svd_gate=SvdGate(baselines={}))

    # Simulate a crash: the round is no longer marked complete, but the
    # per-year rows survive.
    with EvaluationStore(os.path.join(config.data_dir, "evaluations.db")) as store:
        store.conn.execute("DELETE FROM completed_rounds")
        store.conn.execute("DELETE FROM round_results")
        store.conn.commit()

    stack.scored.clear()
    results = run_evaluation(src, config=config, svd_gate=SvdGate(baselines={}))
    assert stack.scored == []
    assert result_for(results, "alice").qualified


def test_submitter_filter_restricts_the_round(stack, config, benchmarks, tmp_path):
    subs = [
        make_submission("alice", "2026-07-01T08:00:00"),
        make_submission("bob", "2026-07-01T09:00:00"),
    ]
    results = run(source(subs, benchmarks), config, tmp_path, submitter_filter=["bob"])
    assert [m.submitter_id for m in results.submitters] == ["bob"]


# ─── stage 2 wiring ──────────────────────────────────────────────────────


def test_stage2_runs_when_several_submitters_qualify(stack, config, benchmarks, tmp_path, monkeypatch):
    captured = {}

    def fake_duels(submitter_ids, questions, years, provider, judge, rng=None, year_samples=2):
        captured["submitters"] = list(submitter_ids)
        return {"alice": 1.0, "bob": 0.0}

    monkeypatch.setattr("wigin_tllm.pipeline.run_quality_duels", fake_duels)

    subs = [
        make_submission("alice", "2026-07-01T08:00:00"),
        make_submission("bob", "2026-07-01T09:00:00"),
    ]
    src = source(subs, benchmarks, prompts=[CompletionPrompt("q", "r")])
    results = run(src, config, tmp_path, judge=ScriptedJudge(["a"]))

    assert sorted(captured["submitters"]) == ["alice", "bob"]
    assert result_for(results, "alice").final_score == pytest.approx(0.7 * 1.0 + 0.3 * 1.0)
    assert result_for(results, "bob").final_score == pytest.approx(0.7)
    assert top_of(results) == "alice"


def test_stage2_is_skipped_without_a_judge(stack, config, benchmarks, tmp_path):
    subs = [
        make_submission("alice", "2026-07-01T08:00:00"),
        make_submission("bob", "2026-07-01T09:00:00"),
    ]
    src = source(subs, benchmarks, prompts=[CompletionPrompt("q", "r")])
    results = run(src, config, tmp_path, judge=None)
    assert result_for(results, "alice").quality_win_rate == 0.0
    # With no quality signal the leak score alone decides.
    assert result_for(results, "alice").final_score == pytest.approx(1.0)


def test_stage2_is_skipped_without_questions(stack, config, benchmarks, tmp_path):
    subs = [
        make_submission("alice", "2026-07-01T08:00:00"),
        make_submission("bob", "2026-07-01T09:00:00"),
    ]
    results = run(source(subs, benchmarks), config, tmp_path, judge=ScriptedJudge(["a"]))
    assert all(s.quality_win_rate == 0.0 for s in results.submitters)


def test_stage2_can_be_disabled(stack, config, benchmarks, tmp_path, monkeypatch):
    monkeypatch.setattr("wigin_tllm.pipeline.run_quality_duels",
                        lambda *a, **k: pytest.fail("stage 2 should not run"))
    config.quality_enabled = False
    subs = [
        make_submission("alice", "2026-07-01T08:00:00"),
        make_submission("bob", "2026-07-01T09:00:00"),
    ]
    src = source(subs, benchmarks, prompts=[CompletionPrompt("q", "r")])
    run(src, config, tmp_path, judge=ScriptedJudge(["a"]))


def test_rejection_stands_even_with_a_permissive_threshold(stack, config, benchmarks, tmp_path):
    """A rejected submission must not slip through on a mis-set threshold.

    Rejections force the worst-possible score of 0.0, which only falls below
    the qualification threshold while that threshold is negative. Exclusion
    has to be explicit, not a side effect of the number.
    """
    stack.hashes = {"original": "same-bytes", "copy": "same-bytes"}
    config.min_eval_score = 1.0  # deliberately permissive: 0.0 < 1.0
    subs = [
        make_submission("original", "2026-07-01T08:00:00"),
        make_submission("copy", "2026-07-01T09:00:00"),
    ]
    results = run(source(subs, benchmarks), config, tmp_path)

    copy = result_for(results, "copy")
    assert copy.disqualified_reason == "duplicate_weights"
    assert not copy.qualified
    assert copy.rank is None


def test_deduplicated_submitter_cannot_qualify(stack, config, benchmarks, tmp_path):
    shared = distinct_spectrum("shared-weights")
    stack.spectra = {"original": shared, "mimic": {"w": shared["w"] + 1e-9}}
    config.min_eval_score = 1.0
    subs = [
        make_submission("original", "2026-07-01T08:00:00"),
        make_submission("mimic", "2026-07-01T09:00:00"),
    ]
    results = run(source(subs, benchmarks), config, tmp_path)

    mimic = result_for(results, "mimic")
    assert mimic.disqualified_reason == "duplicate_of_earlier_submission"
    assert not mimic.qualified


# ─── stage-2 prompt generation ───────────────────────────────────────────


def test_generated_prompts_are_used_for_the_duels(stack, config, benchmarks, tmp_path, monkeypatch):
    """A configured generator replaces whatever the data source holds."""
    from wigin_tllm.scoring.prompt_generator import StaticPromptGenerator

    seen = {}

    def fake_duels(submitter_ids, prompts, years, provider, judge, rng=None, year_samples=2):
        seen["prompts"] = [p.prompt for p in prompts]
        return {s: 0.0 for s in submitter_ids}

    monkeypatch.setattr("wigin_tllm.pipeline.run_quality_duels", fake_duels)

    subs = [
        make_submission("alice", "2026-07-01T08:00:00"),
        make_submission("bob", "2026-07-01T09:00:00"),
    ]
    src = source(subs, benchmarks, prompts=[CompletionPrompt("stored prompt")])
    run(
        src, config, tmp_path,
        judge=ScriptedJudge(["a"]),
        prompt_generator=StaticPromptGenerator([CompletionPrompt("generated prompt")]),
    )
    assert seen["prompts"] == ["generated prompt"]


def test_stored_prompts_are_used_when_no_generator_is_configured(
    stack, config, benchmarks, tmp_path, monkeypatch
):
    seen = {}

    def fake_duels(submitter_ids, prompts, years, provider, judge, rng=None, year_samples=2):
        seen["prompts"] = [p.prompt for p in prompts]
        return {s: 0.0 for s in submitter_ids}

    monkeypatch.setattr("wigin_tllm.pipeline.run_quality_duels", fake_duels)

    subs = [
        make_submission("alice", "2026-07-01T08:00:00"),
        make_submission("bob", "2026-07-01T09:00:00"),
    ]
    src = source(subs, benchmarks, prompts=[CompletionPrompt("stored prompt")])
    run(src, config, tmp_path, judge=ScriptedJudge(["a"]))
    assert seen["prompts"] == ["stored prompt"]


def test_a_generator_producing_nothing_aborts_the_round(stack, config, benchmarks, tmp_path):
    """Silently skipping stage 2 would change what the round measures."""
    from wigin_tllm.scoring.prompt_generator import StaticPromptGenerator

    subs = [
        make_submission("alice", "2026-07-01T08:00:00"),
        make_submission("bob", "2026-07-01T09:00:00"),
    ]
    src = source(subs, benchmarks, prompts=[CompletionPrompt("stored prompt")])
    with pytest.raises(EvaluationError, match="Prompt generation"):
        run(src, config, tmp_path, judge=ScriptedJudge(["a"]),
            prompt_generator=StaticPromptGenerator([]))
