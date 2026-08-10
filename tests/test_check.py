"""The single-model pre-flight check.

The model layer is stubbed so these tests cover the verdicts and diagnoses,
which is what a model author actually reads. `test_end_to_end.py` runs the
same command against real weights.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from wigin_tllm.check import FAIL, OK, SKIP, WARN, check_model  # noqa: E402
from wigin_tllm.datasource import InMemoryDataSource  # noqa: E402
from wigin_tllm.models.loader import ModelLoadError  # noqa: E402
from wigin_tllm.report import format_model_check  # noqa: E402
from wigin_tllm.types import ProbeResult, YearAssessment  # noqa: E402

from conftest import YEARS, make_benchmark  # noqa: E402


class StubModel:
    """A model whose scoring and generation behaviour is dictated per test."""

    def __init__(self, sensible=True, vocabulary=True, generates=True, nan=False):
        self.sensible = sensible
        self.vocabulary = vocabulary
        self.generates = generates
        self.nan = nan

    def encode(self, text):
        if not self.vocabulary:
            return [1] * len(text.split())  # every word collapses to one token
        return [abs(hash(w)) % 500 for w in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def generate(self, prompt, max_new_tokens=50):
        return "a plausible answer" if self.generates else ""

    def parameters(self):
        return iter([])

    def inner_state_dict(self):
        return {}


def assessment(unknown_hits, unknown_total, known_hits, known_total, epsilon=-3.0):
    def result(kind, hits, total, median):
        return ProbeResult(kind=kind, median=median, above_epsilon=hits, total=total,
                           epsilon=epsilon, threshold=0.10)

    unknown = result("unknown", unknown_hits, unknown_total, -9.0 if not unknown_hits else -0.5)
    known = result("known", known_hits, known_total, -0.5 if known_hits else -9.0)
    passed = (not unknown.recognised) and known.recognised
    return YearAssessment(unknown=unknown, known=known, passed=passed,
                          score=(unknown.median - known.median) if passed else 0.0)


@pytest.fixture
def stub(monkeypatch, tmp_path):
    state = {"model": StubModel(), "load_error": None, "size": 1_000, "params": 1_000_000}

    def resolve_model(ref, dest_dir):
        path = os.path.join(tmp_path, "model")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            f.write('{"model_type": "wigin-miniformer"}')
        return path

    def load_model(path, device):
        if state["load_error"]:
            raise state["load_error"]
        return state["model"], None

    scored: list[list] = []

    def score_items(model, device, items):
        """First call is the plausible batch, second the nonsense batch."""
        scored.append(items)
        if model.nan:
            return [float("nan")] * len(items)
        plausible_batch = len(scored) % 2 == 1
        score = 0.0 if plausible_batch == model.sensible else -1.0
        return [score] * len(items)

    monkeypatch.setattr("wigin_tllm.check.resolve_model", resolve_model)
    monkeypatch.setattr("wigin_tllm.check.load_model", load_model)
    monkeypatch.setattr("wigin_tllm.check.get_model_size_bytes", lambda ref: state["size"])
    monkeypatch.setattr("wigin_tllm.check.count_model_params", lambda m: state["params"])
    monkeypatch.setattr("wigin_tllm.check.verify_pinned_revision", lambda ref: True)
    monkeypatch.setattr("wigin_tllm.check.free_device_memory", lambda: None)
    monkeypatch.setattr("wigin_tllm.check.get_device", lambda pref=None: torch.device("cpu"))
    monkeypatch.setattr("wigin_tllm.check.score_items", score_items)
    # Year scoring is covered by test_leak.py; here it only needs to produce
    # an assessment so the report can be built.
    monkeypatch.setattr(
        "wigin_tllm.check.assess_year",
        lambda model, device, unknown, known: state["assessment"],
    )
    state["assessment"] = assessment(0, 10, 10, 10)
    return state


def item_named(report, name):
    return next(i for i in report.artifact if i.name == name)


def source():
    benchmarks = {
        year: {"unknown": make_benchmark(year, "unknown"), "known": make_benchmark(year, "known")}
        for year in YEARS
    }
    return InMemoryDataSource(years=YEARS, benchmarks=benchmarks, submissions=[])


# ─── artefact checks ─────────────────────────────────────────────────────


def test_healthy_model_is_accepted(stub, config):
    report = check_model("local:/m", config=config)
    assert report.ok
    assert item_named(report, "loads without trust_remote_code").level == OK
    assert item_named(report, "tokenizer").level == OK
    assert item_named(report, "generation").level == OK


def test_unloadable_model_is_rejected_with_the_reason(stub, config):
    stub["load_error"] = ModelLoadError("No config.json")
    report = check_model("local:/m", config=config)
    assert not report.ok
    failed = item_named(report, "loads without trust_remote_code")
    assert failed.level == FAIL
    assert "No config.json" in failed.detail


def test_oversized_model_is_rejected(stub, config):
    stub["size"] = config.max_model_bytes + 1
    report = check_model("local:/m", config=config)
    assert not report.ok
    assert item_named(report, "weight size").level == FAIL


def test_parameter_limit_is_reported(stub, config):
    stub["params"] = config.max_parameters + 1
    report = check_model("local:/m", config=config)
    assert not report.ok
    assert item_named(report, "parameters").level == FAIL


def test_undetectable_size_warns_rather_than_blocks(stub, config):
    stub["size"] = -1
    report = check_model("local:/m", config=config)
    assert report.ok
    assert item_named(report, "weight size").level == WARN


def test_silent_model_warns_about_quality_duels(stub, config):
    stub["model"] = StubModel(generates=False)
    report = check_model("local:/m", config=config)
    assert report.ok  # still submittable
    assert item_named(report, "generation").level == WARN


def test_restricted_vocabulary_skips_the_wiring_probe(stub, config):
    """Both sample continuations collapse to the same tokens — inconclusive."""
    stub["model"] = StubModel(vocabulary=False)
    report = check_model("local:/m", config=config)
    assert item_named(report, "scoring").level == SKIP
    assert report.ok


def test_nan_scores_are_a_hard_failure(stub, config):
    stub["model"] = StubModel(nan=True)
    report = check_model("local:/m", config=config)
    assert not report.ok
    assert item_named(report, "scoring").level == FAIL


# ─── consistency checks and diagnoses ────────────────────────────────────


def diagnose(assessment_obj):
    from wigin_tllm.check import _diagnose

    return _diagnose(assessment_obj)


def test_clean_model_is_described_as_clean():
    assert "blind to the future" in diagnose(assessment(0, 10, 10, 10))


def test_leaking_model_is_told_its_data_reaches_too_far():
    text = diagnose(assessment(10, 10, 10, 10))
    assert "beyond the cutoff" in text


def test_empty_model_is_told_it_never_learned_its_era():
    text = diagnose(assessment(0, 10, 0, 10))
    assert "has not learned its own era" in text
    assert "epsilon" in text  # calibration is offered as the alternative cause


def test_inverted_model_is_told_to_check_its_cutoff():
    text = diagnose(assessment(10, 10, 0, 10))
    assert "check that the cutoff" in text


def test_year_without_known_probes_is_called_out():
    assert "no `known` probes" in diagnose(assessment(0, 10, 0, 0))


def test_years_are_scored_when_probes_are_supplied(stub, config):
    report = check_model("local:/m", config=config, datasource=source(), years=[YEARS[0]])
    assert [y.year for y in report.years] == [YEARS[0]]
    assert report.years[0].diagnosis


def test_all_years_are_scored_by_default(stub, config):
    report = check_model("local:/m", config=config, datasource=source())
    assert [y.year for y in report.years] == YEARS


def test_no_datasource_means_no_year_checks(stub, config):
    assert check_model("local:/m", config=config).years == []


# ─── rendering ───────────────────────────────────────────────────────────


def test_report_names_the_blocking_items(stub, config):
    stub["load_error"] = ModelLoadError("boom")
    text = format_model_check(check_model("local:/m", config=config))
    assert "NOT ACCEPTED" in text
    assert "FAIL" in text


def test_report_invites_probe_data_when_none_given(stub, config):
    text = format_model_check(check_model("local:/m", config=config))
    assert "--data" in text


def test_report_summarises_year_outcomes(stub, config):
    text = format_model_check(check_model("local:/m", config=config, datasource=source()))
    assert "Chronological consistency" in text
    assert "must not recognise" in text
