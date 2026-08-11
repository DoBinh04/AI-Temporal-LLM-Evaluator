"""Building a benchmark corpus.

A corpus is a set of dated facts turned into two probe sets per cutoff year:

    known    facts at or before the cutoff — the model MUST recognise these
    unknown  facts after it — the model must NOT

Splitting facts by year is the easy part. The part that decides whether the
benchmark works at all is **calibration**: choosing the `epsilon` above which
a probe counts as recognised.

Get it wrong and the benchmark says nothing. Too strict and every model looks
ignorant; too loose and every model looks like it leaks. Worse, if the honest
hit rate lands near `threshold`, floating-point noise moving a single probe
across `epsilon` flips a whole year between pass and fail — a verdict decided
by rounding.

So calibration here is measured, not guessed: score the probes with a
reference model that genuinely respects the cutoff, then place `epsilon` where
its post-cutoff hit rate sits comfortably under the threshold.

Layout on disk::

    corpus/
      years.json                     [2013, 2014, ...]
      benchmarks/2013/known.json     {"items": [...], "threshold": …, "epsilon": …}
      benchmarks/2013/unknown.json
      completion_prompts.json        stage-2 prompts (optional)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence

from .config import EvaluationConfig
from .types import Benchmark, BenchmarkItem, CompletionPrompt, Fact

logger = logging.getLogger(__name__)

KNOWN, UNKNOWN = "known", "unknown"


class CorpusError(ValueError):
    """The corpus on disk is unusable."""


# ─── reading and writing ─────────────────────────────────────────────────


class Corpus:
    """A benchmark corpus on disk."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    # ── paths ────────────────────────────────────────────────────────────

    def _path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def exists(self) -> bool:
        return os.path.isdir(self._path("benchmarks"))

    # ── reading ──────────────────────────────────────────────────────────

    def years(self) -> list[int]:
        """Cutoff years, from `years.json` or the benchmark directories."""
        manifest = self._path("years.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                return sorted(int(y) for y in json.load(f))

        bench_dir = self._path("benchmarks")
        if not os.path.isdir(bench_dir):
            raise CorpusError(f"No corpus at {self.root} (expected a benchmarks/ directory)")
        return sorted(int(d) for d in os.listdir(bench_dir) if d.isdigit())

    def benchmark(self, year: int, kind: str) -> Benchmark:
        path = self._path("benchmarks", str(year), f"{kind}.json")
        if not os.path.exists(path):
            raise CorpusError(f"Missing probe set: {path}")
        with open(path) as f:
            return Benchmark.from_dict(year, kind, json.load(f))

    def probes_for(self, year: int) -> tuple[Benchmark, Benchmark]:
        """`(unknown, known)` for one year — the order the scoring uses."""
        return self.benchmark(year, UNKNOWN), self.benchmark(year, KNOWN)

    def completion_prompts(self) -> list[CompletionPrompt]:
        path = self._path("completion_prompts.json")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            data = json.load(f)
        raw = data["prompts"] if isinstance(data, dict) else data
        return [
            CompletionPrompt.from_dict(p) if isinstance(p, dict) else CompletionPrompt(str(p))
            for p in raw
        ]

    # ── writing ──────────────────────────────────────────────────────────

    def write(self, benchmarks: dict[int, dict[str, Benchmark]]) -> None:
        os.makedirs(self._path("benchmarks"), exist_ok=True)
        for year, sets in benchmarks.items():
            year_dir = self._path("benchmarks", str(year))
            os.makedirs(year_dir, exist_ok=True)
            for kind, benchmark in sets.items():
                with open(os.path.join(year_dir, f"{kind}.json"), "w") as f:
                    json.dump(benchmark.to_dict(), f, indent=2)
        with open(self._path("years.json"), "w") as f:
            json.dump(sorted(benchmarks), f)
        logger.info(f"Wrote {len(benchmarks)} years of probes to {self.root}")

    def write_completion_prompts(self, prompts: Sequence[CompletionPrompt]) -> str:
        os.makedirs(self.root, exist_ok=True)
        path = self._path("completion_prompts.json")
        with open(path, "w") as f:
            json.dump(
                {"prompts": [{"prompt": p.prompt, "category": p.category,
                              "reference": p.reference} for p in prompts]},
                f, indent=2,
            )
        return path


# ─── building ────────────────────────────────────────────────────────────


def load_facts(path: str) -> list[Fact]:
    """Read a fact list: `{"facts": [{year, prompt, phrase}]}` or a bare list."""
    with open(path) as f:
        data = json.load(f)
    raw = data["facts"] if isinstance(data, dict) else data
    if not raw:
        raise CorpusError(f"No facts in {path}")
    return [Fact.from_dict(item) for item in raw]


def split_by_year(
    facts: Sequence[Fact],
    eval_years: Sequence[int],
    threshold: float,
    epsilon: float,
    known_threshold: Optional[float] = None,
    unknown_threshold: Optional[float] = None,
) -> dict[int, dict[str, Benchmark]]:
    """Turn dated facts into a `known`/`unknown` probe set per cutoff year.

    Every fact is used for every year — as `known` if it predates the cutoff,
    as `unknown` if it comes after. Facts beyond the last evaluated year are
    what make the later years' `unknown` sets non-trivial, so a corpus should
    extend past the years it scores.

    The two requirements are asymmetric, so each side can carry its own
    threshold — a production probe set typically demands most `known` probes
    recognised (e.g. 0.70) while tolerating almost no `unknown` ones (e.g.
    0.10). Left unset, both sides use `threshold`.
    """
    thresholds = {
        KNOWN: known_threshold if known_threshold is not None else threshold,
        UNKNOWN: unknown_threshold if unknown_threshold is not None else threshold,
    }
    out: dict[int, dict[str, Benchmark]] = {}
    for year in sorted(eval_years):
        buckets = {KNOWN: [], UNKNOWN: []}
        for fact in facts:
            kind = KNOWN if fact.year <= year else UNKNOWN
            buckets[kind].append(BenchmarkItem(prompt=fact.prompt, phrase=fact.phrase))
        out[year] = {
            kind: Benchmark(year=year, kind=kind, items=items,
                            threshold=thresholds[kind], epsilon=epsilon)
            for kind, items in buckets.items()
        }
    return out


# ─── calibration ─────────────────────────────────────────────────────────


@dataclass
class YearCalibration:
    """How well one year's probe sets separate, at the chosen epsilon."""

    year: int
    epsilon: float
    threshold: float
    known_rate: float
    unknown_rate: float
    # Per-side thresholds; None = both sides use `threshold`.
    known_threshold: Optional[float] = None
    unknown_threshold: Optional[float] = None

    @property
    def _known_bar(self) -> float:
        return self.known_threshold if self.known_threshold is not None else self.threshold

    @property
    def _unknown_bar(self) -> float:
        return self.unknown_threshold if self.unknown_threshold is not None else self.threshold

    @property
    def separates(self) -> bool:
        """The reference model must clear the bar on one set and not the other."""
        return self.known_rate > self._known_bar and self.unknown_rate <= self._unknown_bar

    @property
    def margin(self) -> float:
        """How far the deciding rate sits from its threshold.

        Small means one probe crossing epsilon would flip the year.
        """
        return min(
            abs(self.known_rate - self._known_bar),
            abs(self.unknown_rate - self._unknown_bar),
        )


@dataclass
class CalibrationReport:
    years: list[YearCalibration]
    reference: str

    @property
    def ok(self) -> bool:
        return bool(self.years) and all(y.separates for y in self.years)

    @property
    def tightest(self) -> Optional[YearCalibration]:
        return min(self.years, key=lambda y: y.margin) if self.years else None


def _quantile(values: Sequence[float], q: float) -> float:
    """Value below which a `q` fraction of `values` sit. Nearest-rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def calibrate(
    facts: Sequence[Fact],
    eval_years: Sequence[int],
    model,
    device,
    config: Optional[EvaluationConfig] = None,
) -> tuple[dict[int, dict[str, Benchmark]], CalibrationReport]:
    """Choose `epsilon` by measuring a reference model, then report the fit.

    The reference must be a model that genuinely respects the cutoffs —
    typically a published checkpoint for the oldest year. Epsilon is placed so
    that reference's post-cutoff hit rate lands at
    `probe_threshold * calibration_margin`, leaving room for the noise that
    would otherwise decide borderline years.
    """
    from .scoring.leak import score_items

    config = config or EvaluationConfig()
    target_rate = config.unknown_probe_threshold * config.calibration_margin

    draft = split_by_year(
        facts, eval_years, config.probe_threshold, epsilon=0.0,
        known_threshold=config.known_threshold,
        unknown_threshold=config.unknown_threshold,
    )
    calibrations: list[YearCalibration] = []

    for year in sorted(eval_years):
        known_scores = _drop_nan(
            score_items(model, device, draft[year][KNOWN].items), f"{year}/known"
        )
        unknown_scores = _drop_nan(
            score_items(model, device, draft[year][UNKNOWN].items), f"{year}/unknown"
        )

        # Place epsilon above all but `target_rate` of the post-cutoff probes.
        # With nothing to place it against, fall back to the known side.
        if unknown_scores:
            epsilon = _quantile(unknown_scores, 1.0 - target_rate)
        elif known_scores:
            epsilon = min(known_scores) - 1.0
        else:
            raise CorpusError(f"Year {year} has no probes on either side")

        known_rate = _rate_above(known_scores, epsilon)
        unknown_rate = _rate_above(unknown_scores, epsilon)
        calibrations.append(
            YearCalibration(
                year=year, epsilon=epsilon, threshold=config.probe_threshold,
                known_rate=known_rate, unknown_rate=unknown_rate,
                known_threshold=config.known_threshold,
                unknown_threshold=config.unknown_threshold,
            )
        )
        for sets in draft[year].values():
            sets.epsilon = epsilon
        logger.info(
            f"{year}: epsilon={epsilon:.4f} known={known_rate:.1%} unknown={unknown_rate:.1%}"
        )

    return draft, CalibrationReport(years=calibrations, reference="reference model")


def _drop_nan(scores: list[float], label: str) -> list[float]:
    """Remove NaN scores before they can poison a quantile.

    CPU kernels sporadically return NaN on borderline inputs; a single one
    would make `epsilon` NaN and silently break every later run against the
    corpus. Dropping the probe costs one data point and says so.
    """
    clean = [s for s in scores if s == s]
    if len(clean) != len(scores):
        logger.warning(
            f"Calibration {label}: dropped {len(scores) - len(clean)} NaN probe "
            "score(s) — re-run if this repeats"
        )
    return clean


def _rate_above(scores: Sequence[float], epsilon: float) -> float:
    if not scores:
        return 0.0
    return sum(1 for s in scores if s > epsilon) / len(scores)


def inspect(corpus: Corpus, model, device, config: EvaluationConfig) -> CalibrationReport:
    """Measure an existing corpus against a model, without changing it.

    Answers the question a corpus author actually has: does this probe set
    separate a model that respects the cutoff from one that does not, and by
    how much?
    """
    from .scoring.leak import probe

    calibrations = []
    for year in corpus.years():
        unknown, known = corpus.probes_for(year)
        unknown_result = probe(model, device, unknown)
        known_result = probe(model, device, known)
        calibrations.append(
            YearCalibration(
                year=year,
                epsilon=known.epsilon,
                threshold=known.threshold,
                known_rate=known_result.ratio,
                unknown_rate=unknown_result.ratio,
                known_threshold=known.threshold,
                unknown_threshold=unknown.threshold,
            )
        )
    return CalibrationReport(years=calibrations, reference="reference model")
