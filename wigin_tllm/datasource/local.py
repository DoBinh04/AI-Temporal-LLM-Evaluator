"""Filesystem-backed data source.

Expected layout (only `benchmarks/` and `submissions/` are required)::

    root/
      round.json                     {"current_round": 1}
      years.json                     [2013, 2014, ...]
      benchmarks/2013/unknown.json   {"items": [{"prompt": ..., "phrase": ...}],
                                      "threshold": 0.10, "epsilon": -11.51}
      benchmarks/2013/known.json
      submissions/1.json             [{"submitter_id": ..., "models": {...},
                                       "submitted_at": "2026-07-01T08:00:00"}]
      quality_questions.json         {"questions": [{"prompt": ...}]}
      results/1.json                 (written by the pipeline)
      eval_details/1.jsonl           (written by the pipeline, one row per year)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict

from ..types import Benchmark, QualityQuestion, RoundResults, Submission, YearEvaluation
from .base import DataSource

logger = logging.getLogger(__name__)

_REQUIRED = object()  # sentinel: no default, so a missing file raises


class LocalDataSource(DataSource):
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Data directory does not exist: {self.root}")

    # ── paths ────────────────────────────────────────────────────────────

    def _path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def _read_json(self, *parts: str, default=_REQUIRED):
        """Read a JSON file. Without a `default`, a missing file is an error."""
        path = self._path(*parts)
        if not os.path.exists(path):
            if default is _REQUIRED:
                raise FileNotFoundError(f"Missing required data file: {path}")
            return default
        with open(path) as f:
            return json.load(f)

    # ── round metadata ───────────────────────────────────────────────────

    def get_current_round(self) -> int:
        return int(self._read_json("round.json", default={"current_round": 1})["current_round"])

    def get_years(self, round_id: int) -> list[int]:
        years = self._read_json("years.json", default=[])
        if years:
            return sorted(int(y) for y in years)
        # Fall back to whatever the benchmarks directory contains.
        bench_dir = self._path("benchmarks")
        if not os.path.isdir(bench_dir):
            raise FileNotFoundError(f"No years.json and no benchmarks/ under {self.root}")
        return sorted(int(d) for d in os.listdir(bench_dir) if d.isdigit())

    # ── inputs ───────────────────────────────────────────────────────────

    def get_benchmark(self, year: int, kind: str) -> Benchmark:
        data = self._read_json("benchmarks", str(year), f"{kind}.json")
        return Benchmark.from_dict(year, kind, data)

    def get_submissions(self, round_id: int) -> list[Submission]:
        raw = self._read_json("submissions", f"{round_id}.json", default=[])
        if isinstance(raw, dict):  # {"submitter_id": {"models": ..., "submitted_at": ...}}
            raw = [{"submitter_id": mid, **body} for mid, body in raw.items()]
        return [
            Submission(
                submitter_id=str(entry["submitter_id"]),
                models={str(k): v for k, v in entry["models"].items()},
                submitted_at=entry.get("submitted_at", ""),
            )
            for entry in raw
        ]

    def get_quality_questions(self) -> list[QualityQuestion]:
        data = self._read_json("quality_questions.json", default={"questions": []})
        questions = data["questions"] if isinstance(data, dict) else data
        return [QualityQuestion.from_dict(q) if isinstance(q, dict) else QualityQuestion(str(q))
                for q in questions]

    # ── outputs ──────────────────────────────────────────────────────────

    def save_year_evaluation(self, round_id: int, evaluation: YearEvaluation) -> bool:
        out_dir = self._path("eval_details")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{round_id}.jsonl"), "a") as f:
            f.write(json.dumps(asdict(evaluation)) + "\n")
        return True

    def save_results(self, round_id: int, results: RoundResults) -> None:
        out_dir = self._path("results")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{round_id}.json")
        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Results written to {path}")
