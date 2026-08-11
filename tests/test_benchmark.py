"""The benchmark stages.

The model layer is stubbed so these cover the verdicts and the arithmetic —
what a model author actually reads. `test_end_to_end.py` runs the same code
against real weights.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from wigin_tllm.benchmark import benchmark  # noqa: E402
from wigin_tllm.benchmark.artifact import FAIL, OK, WARN  # noqa: E402
from wigin_tllm.benchmark.consistency import diagnose, group_by_ref, score_manifest  # noqa: E402
from wigin_tllm.benchmark.quality import QualityReport, opponent_lineup  # noqa: E402
from wigin_tllm.benchmark.similarity import Comparison, references_for_year  # noqa: E402
from wigin_tllm.corpus import Corpus, split_by_year  # noqa: E402
from wigin_tllm.models.loader import ModelLoadError  # noqa: E402
from wigin_tllm.report import format_benchmark  # noqa: E402
from wigin_tllm.scoring.judge import ScriptedJudge  # noqa: E402
from wigin_tllm.types import ProbeResult, YearAssessment  # noqa: E402

from conftest import YEARS, make_facts  # noqa: E402

SHA = "a" * 40


def assessment(passed=True, score=-8.0, unknown_hits=0, known_hits=10) -> YearAssessment:
    def result(kind, hits, median):
        return ProbeResult(kind=kind, median=median, above_epsilon=hits, total=10,
                           epsilon=-5.0, threshold=0.10)

    return YearAssessment(
        unknown=result("unknown", unknown_hits, -9.0 if not unknown_hits else -0.5),
        known=result("known", known_hits, -0.5 if known_hits else -9.0),
        passed=passed,
        score=score,
    )


@pytest.fixture
def corpus(tmp_path) -> Corpus:
    built = Corpus(str(tmp_path / "corpus"))
    built.write(split_by_year(make_facts(), YEARS, threshold=0.25, epsilon=-5.0))
    return built


class FakeModel:
    """Enough surface for the artefact checks and the SVD gate to run."""

    def encode(self, text, **kwargs):
        return [abs(hash(w)) % 500 for w in text.split()]

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)

    def generate(self, prompt, max_new_tokens=50):
        return "a plausible continuation"

    def inner_state_dict(self):
        generator = torch.Generator().manual_seed(42)
        return {"layer.weight": torch.randn(16, 16, generator=generator)}


@pytest.fixture
def stub(monkeypatch, tmp_path):
    state = {"load_error": None, "params": 1_000_000, "size": 1_000,
             "pinned": True, "assessment": assessment(), "loads": []}

    def resolve_model(ref, dest_dir):
        path = os.path.join(tmp_path, "model")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            f.write('{"model_type": "wigin-miniformer"}')
        return path

    def load_model(path, device):
        state["loads"].append(path)
        if state["load_error"]:
            raise state["load_error"]
        return FakeModel(), None

    for module in ("consistency", "artifact"):
        monkeypatch.setattr(f"wigin_tllm.benchmark.{module}.resolve_model", resolve_model)
        monkeypatch.setattr(f"wigin_tllm.benchmark.{module}.load_model", load_model)
        monkeypatch.setattr(f"wigin_tllm.benchmark.{module}.count_model_params",
                            lambda m: state["params"])
        monkeypatch.setattr(f"wigin_tllm.benchmark.{module}.get_model_size_bytes",
                            lambda ref: state["size"])
        monkeypatch.setattr(f"wigin_tllm.benchmark.{module}.verify_pinned_revision",
                            lambda ref: state["pinned"])
    monkeypatch.setattr("wigin_tllm.benchmark.consistency.free_device_memory", lambda: None)
    monkeypatch.setattr("wigin_tllm.benchmark.consistency.assess_year",
                        lambda *a: state["assessment"])
    # benchmark() builds the stage-1 gate from the references; the stubbed
    # models have no real weights, so serve an empty (pass-everything) gate.
    from wigin_tllm.scoring.svd_gate import SvdGate
    monkeypatch.setattr("wigin_tllm.benchmark.similarity.build_gate",
                        lambda *a, **k: state.get("gate") or SvdGate())
    monkeypatch.setattr("wigin_tllm.models.store.get_device", lambda pref=None: torch.device("cpu"))
    # The smoke probes score through leak.score_items; keep them sensible.
    monkeypatch.setattr(
        "wigin_tllm.benchmark.artifact.score_items",
        lambda model, device, items: [0.0 if "P" in i.phrase or i.phrase[0].isupper() else -1.0
                                      for i in items],
    )
    return state


def manifest(years=None, ref=f"owner/model@{SHA}") -> dict[str, str]:
    return {str(y): ref for y in (years if years is not None else YEARS)}


def item(report, name):
    return next(i for i in report.artifact.items if i.name == name)


# ─── consistency ─────────────────────────────────────────────────────────


def test_every_year_in_scope_is_scored(stub, corpus, config):
    report = score_manifest(manifest(), corpus, config, "cpu")
    assert [y.year for y in report.years] == YEARS
    assert report.covers_all_years


def test_the_score_averages_over_all_years_not_just_the_scored_ones(stub, corpus, config):
    report = score_manifest(manifest(YEARS[:2]), corpus, config, "cpu")
    assert report.leak_score == pytest.approx(2 * stub["assessment"].score / len(YEARS))
    assert not report.covers_all_years


def test_one_model_serving_several_years_is_loaded_once(stub, corpus, config):
    score_manifest(manifest(), corpus, config, "cpu")
    assert len(stub["loads"]) == 1


def test_a_broken_model_costs_only_its_own_years(stub, corpus, config):
    stub["load_error"] = ModelLoadError("bad weights")
    report = score_manifest(manifest(), corpus, config, "cpu")
    assert report.years == []
    assert report.leak_score == 0.0
    assert report.errors


def test_an_oversized_model_costs_its_years(stub, corpus, config):
    stub["params"] = config.max_parameters + 1
    report = score_manifest(manifest(), corpus, config, "cpu")
    assert report.years == []
    assert any("exceeds the limit" in e for e in report.errors)


def test_grouping_pairs_years_with_their_model():
    models = {"2013": "a", "2014": "a", "2015": "b"}
    assert group_by_ref(models, YEARS) == {"a": [2013, 2014], "b": [2015]}


def test_clearing_the_threshold_follows_the_mean(stub, corpus, config):
    report = score_manifest(manifest(), corpus, config, "cpu")
    assert report.clears_threshold is (report.leak_score < config.min_eval_score)


def test_a_saturated_score_is_flagged(stub, corpus, config):
    stub["assessment"] = assessment(score=-20.0)
    report = score_manifest(manifest(), corpus, config, "cpu")
    assert report.normalized == 1.0
    assert report.saturated


def test_an_overweight_model_costs_its_years(stub, corpus, config):
    stub["size"] = config.max_model_bytes + 1
    report = score_manifest(manifest(), corpus, config, "cpu")
    assert report.years == []
    assert any("MB of weights exceeds the limit" in e for e in report.errors)


def test_an_unpinned_revision_costs_its_years(stub, corpus, config):
    """A branch named like a SHA passes the format check but is not a commit."""
    config.require_pinned_revision = True
    stub["pinned"] = False
    report = score_manifest(manifest(), corpus, config, "cpu")
    assert report.years == []
    assert any("not a real commit SHA" in e for e in report.errors)


def test_a_ref_without_a_sha_fails_before_download(stub, corpus, config):
    config.require_pinned_revision = True
    report = score_manifest(manifest(ref="owner/model"), corpus, config, "cpu")
    assert report.years == []
    assert any("not pinned to a commit SHA" in e for e in report.errors)
    assert stub["loads"] == []  # never resolved, never loaded


def test_local_refs_need_no_pin(stub, corpus, config, tmp_path):
    config.require_pinned_revision = True
    local = tmp_path / "local-model"
    local.mkdir()
    report = score_manifest(manifest(ref=f"local:{local}"), corpus, config, "cpu")
    assert report.covers_all_years


def test_years_past_the_time_budget_score_worst(stub, corpus, config, monkeypatch):
    """The clock starts at model load; years not reached in time cost their score."""

    class Clock:
        now = 0.0

        def monotonic(self):
            Clock.now += 10.0
            return Clock.now

    monkeypatch.setattr("wigin_tllm.benchmark.consistency.time", Clock())
    config.max_eval_seconds = 15.0
    report = score_manifest(manifest(), corpus, config, "cpu")
    assert [y.year for y in report.years] == YEARS[:1]
    assert any("time budget" in e for e in report.errors)
    # The unreached years still count worst-possible in the mean.
    assert report.leak_score == pytest.approx(stub["assessment"].score / len(YEARS))


def test_no_time_budget_means_no_cutoff(stub, corpus, config):
    config.max_eval_seconds = None
    report = score_manifest(manifest(), corpus, config, "cpu")
    assert report.covers_all_years


# ─── the SVD gate in stage 1 ─────────────────────────────────────────────


def gate_matching(model, years, threshold=0.01):
    from wigin_tllm.scoring.svd_gate import SvdGate, svd_spectra

    gate = SvdGate(threshold=threshold)
    for year in years:
        gate.add_baseline(year, svd_spectra(model.inner_state_dict()))
    return gate


def test_a_gated_year_scores_worst_before_any_probe(stub, corpus, config):
    gate = gate_matching(FakeModel(), YEARS[:1])
    report = score_manifest(manifest(), corpus, config, "cpu", gate=gate)

    gated = next(y for y in report.years if y.year == YEARS[0])
    assert not gated.passed
    assert gated.score == 0.0
    assert gated.assessment is None
    assert "scored as a copy" in gated.diagnosis
    # The other years are probed as normal.
    assert all(y.passed for y in report.years if y.year != YEARS[0])


def test_the_gate_drags_the_mean_down(stub, corpus, config):
    gate = gate_matching(FakeModel(), YEARS[:1])
    gated = score_manifest(manifest(), corpus, config, "cpu", gate=gate)
    ungated = score_manifest(manifest(), corpus, config, "cpu")
    assert gated.leak_score > ungated.leak_score  # closer to 0 = worse


def test_a_distinct_model_passes_the_gate_untouched(stub, corpus, config):
    from wigin_tllm.scoring.svd_gate import SvdGate, svd_spectra

    gate = SvdGate()
    for year in YEARS:
        gate.add_baseline(year, svd_spectra(
            {"layer.weight": torch.randn(16, 16, generator=torch.Generator().manual_seed(7))}
        ))
    report = score_manifest(manifest(), corpus, config, "cpu", gate=gate)
    assert all(y.passed for y in report.years)


# ─── diagnoses ───────────────────────────────────────────────────────────


def test_a_clean_year_says_so():
    assert "blind to the future" in diagnose(assessment(passed=True))


def test_a_borderline_pass_is_called_out():
    """A verdict one probe from flipping should not read as comfortable."""
    borderline = assessment(passed=True, unknown_hits=1, known_hits=2)
    assert "only just" in diagnose(borderline)


def test_a_leaking_year_points_at_the_training_data():
    assert "beyond the cutoff" in diagnose(assessment(passed=False, unknown_hits=10, known_hits=10))


def test_an_empty_model_is_told_it_learned_nothing():
    text = diagnose(assessment(passed=False, unknown_hits=0, known_hits=0))
    assert "has not learned its own era" in text


# ─── similarity ──────────────────────────────────────────────────────────


def test_a_close_match_reads_as_a_copy():
    assert Comparison("ref", distance=0.0001, threshold=0.01).too_close


def test_a_distant_model_is_distinct():
    assert not Comparison("ref", distance=0.5, threshold=0.01).too_close


def test_an_incomparable_pair_is_neither():
    comparison = Comparison("ref", distance=None, threshold=0.01)
    assert not comparison.comparable
    assert not comparison.too_close


def test_a_year_mapping_narrows_to_that_year():
    references = {2013: ["a"], 2014: ["b"]}
    assert references_for_year(references, 2013) == ["a"]
    assert sorted(references_for_year(references, None)) == ["a", "b"]


def test_a_bare_list_applies_whatever_the_year():
    assert references_for_year(["a", "b"], 2013) == ["a", "b"]


# ─── pairwise ────────────────────────────────────────────────────────────


def spectra_for(seed: int) -> dict:
    from wigin_tllm.scoring.svd_gate import svd_spectra

    return svd_spectra(
        {"layer.weight": torch.randn(16, 16, generator=torch.Generator().manual_seed(seed))}
    )


@pytest.fixture
def pairwise_world(monkeypatch):
    """`_spectra_of` served from a dict; 'broken' refs refuse to load."""
    from wigin_tllm.benchmark import similarity as similarity_stage

    world = {"local:/a": spectra_for(1), "local:/b": spectra_for(2),
             "local:/copy-of-a": spectra_for(1)}

    def fake_spectra(raw_ref, device):
        if raw_ref not in world:
            raise FileNotFoundError(f"no weights in {raw_ref}")
        return world[raw_ref]

    monkeypatch.setattr(similarity_stage, "_spectra_of", fake_spectra)
    return similarity_stage


def test_pairwise_flags_the_later_listed_copy(pairwise_world, config):
    report = pairwise_world.pairwise(["local:/a", "local:/copy-of-a", "local:/b"], config, "cpu")
    assert report.accepted == {"local:/a", "local:/b"}
    assert report.duplicates == ["local:/copy-of-a"]
    assert not report.ok


def test_pairwise_an_unreadable_model_is_an_error_not_a_copy(pairwise_world, config):
    report = pairwise_world.pairwise(["local:/a", "local:/broken", "local:/b"], config, "cpu")
    assert any("broken" in e for e in report.errors)
    assert report.duplicates == []
    assert not report.ok
    # The readable pair is still compared.
    assert [(l, r) for l, r, _ in report.pairs] == [("local:/a", "local:/b")]


def test_pairwise_the_same_ref_listed_twice_is_one_model(pairwise_world, config):
    report = pairwise_world.pairwise(["local:/a", "local:/a", "local:/b"], config, "cpu")
    assert report.measured == ["local:/a", "local:/b"]
    assert len(report.pairs) == 1
    assert report.ok


# ─── quality ─────────────────────────────────────────────────────────────


def test_a_year_mapping_pairs_opponents_by_position():
    lineup = opponent_lineup({2013: ["a13", "b13"], 2014: ["a14", "b14"]}, YEARS)
    assert set(lineup) == {"ref-0", "ref-1"}
    assert lineup["ref-0"].models == {"2013": "a13", "2014": "a14"}


def test_a_bare_list_becomes_one_opponent_per_entry():
    lineup = opponent_lineup(["x", "y"], [2013])
    assert set(lineup) == {"ref-0", "ref-1"}
    assert lineup["ref-0"].models == {"2013": "x"}


def test_a_zero_rate_against_one_reference_is_ambiguous():
    """It cannot be told from drawing every duel — a draw scores for neither."""
    assert QualityReport(win_rate=0.0, opponents=["ref-0"]).ambiguous
    assert not QualityReport(win_rate=0.0, opponents=["ref-0", "ref-1"]).ambiguous
    assert not QualityReport(win_rate=0.5, opponents=["ref-0"]).ambiguous


# ─── the whole benchmark ─────────────────────────────────────────────────


def test_a_complete_manifest_is_accepted(stub, corpus, config):
    report = benchmark(manifest(), corpus, config=config)
    assert report.ok
    assert item(report, "manifest").level == OK
    assert report.consistency.covers_all_years


def test_a_partial_manifest_warns_and_costs_the_score(stub, corpus, config):
    full = benchmark(manifest(), corpus, config=config)
    partial = benchmark(manifest(YEARS[:2]), corpus, config=config)
    assert item(partial, "manifest").level == WARN
    assert partial.consistency.leak_score > full.consistency.leak_score


def test_an_invalid_manifest_stops_before_scoring(stub, corpus, config):
    report = benchmark({"1999": f"owner/m@{SHA}"}, corpus, config=config)
    assert not report.ok
    assert item(report, "manifest").level == FAIL
    assert report.consistency is None


def test_without_references_only_consistency_is_measured(stub, corpus, config):
    report = benchmark(manifest(), corpus, config=config)
    assert report.similarity is None
    assert not report.quality.measured
    assert report.final_score is None


def test_the_final_score_blends_both_stages(stub, corpus, config, monkeypatch):
    monkeypatch.setattr(
        "wigin_tllm.benchmark.quality_stage.run",
        lambda *a, **k: QualityReport(win_rate=0.5, opponents=["ref-0", "ref-1"]),
    )
    monkeypatch.setattr("wigin_tllm.benchmark.similarity_stage.compare",
                        lambda *a, **k: None)
    report = benchmark(manifest(), corpus, config=config, references=["ref"],
                       judge=ScriptedJudge(["a"]))
    assert report.final_score == pytest.approx(
        config.leak_weight * report.consistency.normalized + config.quality_weight * 0.5
    )


def test_an_unqualified_model_scores_zero_and_skips_stage_2(stub, corpus, config, monkeypatch):
    """Not clearing stage 1 means no duels and a final score of 0 — quality
    cannot buy back a failed consistency check."""
    calls = []
    monkeypatch.setattr(
        "wigin_tllm.benchmark.quality_stage.run",
        lambda *a, **k: calls.append(1) or QualityReport(win_rate=1.0, opponents=["r0", "r1"]),
    )
    monkeypatch.setattr("wigin_tllm.benchmark.similarity_stage.compare", lambda *a, **k: None)

    stub["assessment"] = assessment(passed=False, score=0.0, unknown_hits=10)
    report = benchmark(manifest(), corpus, config=config, references=["ref"],
                       judge=ScriptedJudge(["a"]))
    assert not report.qualified
    assert report.final_score == 0.0
    assert calls == []
    assert "stage 1 not cleared" in report.quality.skipped


def test_a_qualified_model_still_needs_quality_for_a_final_score(stub, corpus, config):
    report = benchmark(manifest(), corpus, config=config)
    assert report.qualified
    assert report.final_score is None


def test_a_model_too_close_to_a_reference_is_not_accepted(stub, corpus, config, monkeypatch):
    from wigin_tllm.benchmark.similarity import SimilarityReport

    monkeypatch.setattr(
        "wigin_tllm.benchmark.similarity_stage.compare",
        lambda *a, **k: SimilarityReport(
            model="m", comparisons=[Comparison("ref", 0.0, config.svd_threshold)]
        ),
    )
    monkeypatch.setattr("wigin_tllm.benchmark.quality_stage.run",
                        lambda *a, **k: QualityReport(skipped="n/a"))
    report = benchmark(manifest(), corpus, config=config, references=["ref"])
    assert not report.ok


def test_the_report_shows_every_stage(stub, corpus, config, monkeypatch):
    monkeypatch.setattr(
        "wigin_tllm.benchmark.quality_stage.run",
        lambda *a, **k: QualityReport(win_rate=0.5, opponents=["ref-0", "ref-1"], prompt_count=6),
    )
    monkeypatch.setattr("wigin_tllm.benchmark.similarity_stage.compare", lambda *a, **k: None)
    text = format_benchmark(
        benchmark(manifest(), corpus, config=config, references=["ref"],
                  judge=ScriptedJudge(["a"]))
    )
    assert "Artefact" in text
    assert "Consistency" in text
    assert "Quality" in text
    assert "Final score" in text
