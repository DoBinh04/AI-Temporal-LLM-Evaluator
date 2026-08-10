"""Data sources: filesystem layout, fallbacks, and the results sink."""

from __future__ import annotations

import json

import pytest

from wigin_tllm.datasource import InMemoryDataSource, LocalDataSource
from wigin_tllm.types import (
    Benchmark,
    BenchmarkItem,
    QualityQuestion,
    RoundResults,
    SubmitterResult,
    YearEvaluation,
)


def results_with(submitter_id: str) -> RoundResults:
    return RoundResults(
        round_id=1,
        submitters=[SubmitterResult(submitter_id=submitter_id, leak_score=-6.0, rank=1)],
    )


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    for year in (2013, 2014):
        year_dir = root / "benchmarks" / str(year)
        year_dir.mkdir(parents=True)
        for kind in ("known", "unknown"):
            (year_dir / f"{kind}.json").write_text(
                json.dumps(
                    {
                        "items": [{"prompt": f"{kind}-{year}", "phrase": "x"}],
                        "threshold": 0.2,
                        "epsilon": -4.0,
                    }
                )
            )
    (root / "submissions").mkdir()
    (root / "submissions" / "1.json").write_text(
        json.dumps(
            [
                {
                    "submitter_id": "alice",
                    "submitted_at": "2026-07-01T08:00:00",
                    "models": {"2013": "local:/m/a", "2014": "local:/m/a"},
                }
            ]
        )
    )
    (root / "round.json").write_text(json.dumps({"current_round": 1}))
    return root


# ─── local source ────────────────────────────────────────────────────────


def test_missing_root_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalDataSource(str(tmp_path / "nope"))


def test_reads_current_round(data_root):
    assert LocalDataSource(str(data_root)).get_current_round() == 1


def test_current_round_defaults_to_one(tmp_path):
    (tmp_path / "benchmarks").mkdir()
    assert LocalDataSource(str(tmp_path)).get_current_round() == 1


def test_years_are_inferred_from_the_benchmark_directory(data_root):
    assert LocalDataSource(str(data_root)).get_years(1) == [2013, 2014]


def test_explicit_years_file_wins(data_root):
    (data_root / "years.json").write_text(json.dumps([2014]))
    assert LocalDataSource(str(data_root)).get_years(1) == [2014]


def test_years_without_benchmarks_or_manifest_is_an_error(tmp_path):
    assert LocalDataSource(str(tmp_path)) is not None
    with pytest.raises(FileNotFoundError):
        LocalDataSource(str(tmp_path)).get_years(1)


def test_benchmark_carries_its_calibration(data_root):
    bench = LocalDataSource(str(data_root)).get_benchmark(2013, "unknown")
    assert bench.threshold == 0.2
    assert bench.epsilon == -4.0
    assert bench.items[0].prompt == "unknown-2013"


def test_missing_benchmark_is_an_error(data_root):
    with pytest.raises(FileNotFoundError):
        LocalDataSource(str(data_root)).get_benchmark(2099, "unknown")


def test_submissions_are_parsed(data_root):
    subs = LocalDataSource(str(data_root)).get_submissions(1)
    assert len(subs) == 1
    assert subs[0].submitter_id == "alice"
    assert subs[0].submitted_at == "2026-07-01T08:00:00"


def test_submissions_accept_a_mapping_layout(data_root):
    (data_root / "submissions" / "2.json").write_text(
        json.dumps({"bob": {"models": {"2013": "local:/m/b"}, "submitted_at": "2026-07-02"}})
    )
    subs = LocalDataSource(str(data_root)).get_submissions(2)
    assert subs[0].submitter_id == "bob"


def test_missing_submissions_file_is_empty(data_root):
    assert LocalDataSource(str(data_root)).get_submissions(99) == []


def test_quality_questions_default_to_empty(data_root):
    assert LocalDataSource(str(data_root)).get_quality_questions() == []


def test_quality_questions_are_parsed(data_root):
    (data_root / "quality_questions.json").write_text(
        json.dumps({"questions": [{"prompt": "p", "reference": "r"}]})
    )
    questions = LocalDataSource(str(data_root)).get_quality_questions()
    assert questions == [QualityQuestion(prompt="p", reference="r")]


def test_preload_fetches_both_probe_sets(data_root):
    loaded = LocalDataSource(str(data_root)).preload_benchmarks([2013, 2014])
    assert set(loaded) == {2013, 2014}
    assert set(loaded[2013]) == {"known", "unknown"}


# ─── outputs ─────────────────────────────────────────────────────────────


def test_results_are_written_as_json(data_root):
    source = LocalDataSource(str(data_root))
    source.save_results(1, results_with("alice"))
    written = json.loads((data_root / "results" / "1.json").read_text())
    assert written["submitters"][0]["submitter_id"] == "alice"


def test_year_evaluations_append_to_a_jsonl(data_root):
    source = LocalDataSource(str(data_root))
    for year in (2013, 2014):
        source.save_year_evaluation(
            1, YearEvaluation("alice", year, "local:/m", True, -5.0)
        )
    lines = (data_root / "eval_details" / "1.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["year"] == 2013


# ─── in-memory source ────────────────────────────────────────────────────


def test_in_memory_source_captures_outputs():
    bench = Benchmark(2013, "known", [BenchmarkItem("p", "x")])
    source = InMemoryDataSource(
        years=[2013], benchmarks={2013: {"known": bench, "unknown": bench}}, submissions=[]
    )
    source.save_year_evaluation(1, YearEvaluation("alice", 2013, "r", True, -5.0))
    source.save_results(1, results_with("alice"))
    assert len(source.saved_evaluations) == 1
    assert source.saved_results[1].submitters[0].submitter_id == "alice"
