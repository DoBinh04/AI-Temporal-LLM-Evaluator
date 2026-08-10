"""Data model for the evaluation pipeline.

Plain Python throughout — no torch, no numpy — so the shapes that flow
between stages can be imported and tested without any heavy dependency.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

# `owner/repo`: one owner segment, then a repo name that may itself contain
# slashes. Neither may contain '@', which separates the revision.
_HF_LOCATION = re.compile(r"^[^/@]+/[^@]+$")
# A full git commit SHA. Pinning to one is what makes a submission immutable:
# a branch can be repointed at different weights after the fact.
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class SubmissionError(ValueError):
    """A submission manifest is structurally invalid."""


@dataclass(frozen=True)
class ModelRef:
    """A reference to a set of model weights.

    Two schemes:

    ``hf``     ``owner/repo@<40-hex-sha>`` (``hf:`` prefix optional)
    ``local``  ``local:/path/to/model_dir``

    ``local`` exists so the pipeline can run without network access; ``hf`` is
    the production path.
    """

    scheme: str
    location: str
    revision: Optional[str] = None
    raw: str = ""

    @property
    def is_pinned(self) -> bool:
        """True when this reference cannot be repointed after submission."""
        return (
            self.scheme == "hf"
            and bool(_HF_LOCATION.match(self.location))
            and bool(self.revision)
            and bool(_COMMIT_SHA.match(self.revision))
        )

    @classmethod
    def parse(cls, ref: str) -> "ModelRef":
        if not isinstance(ref, str) or not ref.strip():
            raise SubmissionError(f"Model reference must be a non-empty string, got {ref!r}")
        ref = ref.strip()

        if ref.startswith("local:"):
            return cls(scheme="local", location=ref[len("local:"):], raw=ref)

        body = ref[len("hf:"):] if ref.startswith("hf:") else ref
        # Split on the last '@': the revision is the trailing segment, so a
        # stray '@' earlier in the string stays part of the location and gets
        # rejected there rather than silently becoming the revision.
        if "@" in body:
            location, revision = body.rsplit("@", 1)
        else:
            location, revision = body, None
        return cls(scheme="hf", location=location, revision=revision or None, raw=ref)

    def __str__(self) -> str:
        return self.raw


@dataclass
class Submission:
    """One submitter's manifest: a model per cutoff year.

    ``submitted_at`` is an ISO-8601 timestamp and the tie-breaker for every
    anti-copy rule — the earliest submitter of a set of weights keeps them.
    """

    submitter_id: str
    models: dict[str, str]
    submitted_at: str = ""

    def ref_for_year(self, year: int | str) -> Optional[ModelRef]:
        raw = self.models.get(str(year))
        return ModelRef.parse(raw) if raw else None

    @property
    def sort_key(self) -> tuple[str, str]:
        """Ordering for anti-copy priority; undated submissions sort last."""
        return (self.submitted_at or "9999", self.submitter_id)


@dataclass
class BenchmarkItem:
    """A single (prompt, continuation) probe."""

    prompt: str
    phrase: str

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkItem":
        return cls(prompt=d["prompt"], phrase=d["phrase"])


@dataclass
class Benchmark:
    """A probe set for one cutoff year.

    ``kind`` is ``"unknown"`` (facts from after the cutoff, which the model
    must NOT know) or ``"known"`` (facts from before it, which it MUST know).

    ``epsilon`` is the per-probe log-probability above which the model counts
    as recognising the fact; ``threshold`` is the tolerated fraction of such
    probes.
    """

    year: int
    kind: str
    items: list[BenchmarkItem] = field(default_factory=list)
    threshold: float = 0.10
    epsilon: float = -11.51

    @classmethod
    def from_dict(cls, year: int, kind: str, d: dict) -> "Benchmark":
        return cls(
            year=year,
            kind=kind,
            items=[BenchmarkItem.from_dict(i) for i in d.get("items", [])],
            threshold=float(d.get("threshold", 0.10)),
            epsilon=float(d.get("epsilon", -11.51)),
        )


@dataclass
class QualityQuestion:
    """An open-ended stage-2 question.

    ``reference`` is optional and ignored by LLM judges; judges that score by
    overlap use it as the expected answer.
    """

    prompt: str
    reference: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "QualityQuestion":
        return cls(prompt=d["prompt"], reference=d.get("reference", ""))


@dataclass
class ProbeResult:
    """What one probe set revealed about a model.

    ``recognised`` is the verdict the scoring rule cares about: did more than
    `threshold` of the probes score above `epsilon`? Read it as "the model
    knows this material" — required for a `known` set, disqualifying for an
    `unknown` one.
    """

    kind: str
    median: float
    above_epsilon: int
    total: int
    epsilon: float
    threshold: float

    @property
    def ratio(self) -> float:
        return self.above_epsilon / self.total if self.total else 0.0

    @property
    def recognised(self) -> bool:
        return self.ratio > self.threshold


@dataclass
class YearAssessment:
    """The full picture for one cutoff year, both probe sets included."""

    unknown: ProbeResult
    known: ProbeResult
    passed: bool
    score: float


@dataclass
class YearEvaluation:
    """Stage-1 outcome for one (submitter, year) pair."""

    submitter_id: str
    year: int
    model_ref: str
    passed: bool
    score: float
    score_unknown: float = 0.0
    score_known: float = 0.0
    reason: str = ""


# ─── intermediate results ────────────────────────────────────────────────


@dataclass
class ConsistencyResults:
    """Everything stage 1 concluded, keyed by submitter."""

    leak_scores: dict[str, float] = field(default_factory=dict)
    year_scores: dict[str, dict[int, float]] = field(default_factory=dict)
    disqualified: dict[str, str] = field(default_factory=dict)

    def disqualify(self, submitter_id: str, reason: str, score: float) -> None:
        """Record a reason and force the score to worst-possible.

        The submitter stays in `leak_scores` so it still appears in the report
        with an explanation; `eligible` is what decides who can advance.
        """
        self.disqualified[submitter_id] = reason
        self.leak_scores[submitter_id] = score

    @property
    def eligible(self) -> dict[str, float]:
        """Scores of the submitters that may advance to qualification.

        Rejected submissions are excluded here rather than relying on their
        worst-possible score falling below the threshold — that equivalence
        holds only while the threshold is negative, and the rejection should
        stand whatever it is set to.
        """
        return {
            submitter_id: score
            for submitter_id, score in self.leak_scores.items()
            if submitter_id not in self.disqualified
        }


@dataclass
class Qualification:
    """Who advanced to stage 2, and their normalised leak scores."""

    entrants: list[tuple[str, float]] = field(default_factory=list)
    normalized_leak: dict[str, float] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return [submitter_id for submitter_id, _ in self.entrants]

    def __len__(self) -> int:
        return len(self.entrants)


@dataclass
class Ranking:
    """The final blend: quality signal, combined scores, and the order."""

    win_rates: Optional[dict[str, float]] = None
    final_scores: dict[str, float] = field(default_factory=dict)
    ordered: list[tuple[str, float]] = field(default_factory=list)

    @property
    def rank_of(self) -> dict[str, int]:
        return {submitter_id: i + 1 for i, (submitter_id, _) in enumerate(self.ordered)}


# ─── final output ────────────────────────────────────────────────────────


@dataclass
class SubmitterResult:
    """Everything the pipeline concluded about one submitter.

    Read top to bottom this is the whole explanation of a score: what each
    year scored, what that averages to, how it maps onto [0, 1], what the
    quality duels added, and where that lands in the field.
    """

    submitter_id: str
    #: Mean year score. Negative is good; 0.0 is the worst possible value.
    leak_score: float
    #: Per-year raw scores, `median(unknown) - median(known)`.
    year_scores: dict[int, float] = field(default_factory=dict)
    #: Whether the submitter advanced to the quality stage.
    qualified: bool = False
    #: `leak_score` mapped onto [0, 1], higher is better.
    normalized_leak: float = 0.0
    #: Share of quality duels won, in [0, 1].
    quality_win_rate: float = 0.0
    #: `leak_weight * normalized_leak + quality_weight * quality_win_rate`.
    final_score: float = 0.0
    #: 1-based position in the field; None if the submitter did not rank.
    rank: Optional[int] = None
    #: Why the submission was rejected, if it was.
    disqualified_reason: str = ""


@dataclass
class RoundResults:
    """The serialisable outcome of one evaluation round.

    `submitters` is the single source of truth and is ordered best-first, so
    the leading entry is the strongest model of the round.
    """

    round_id: int
    submitters: list[SubmitterResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RoundResults":
        return cls(
            round_id=payload["round_id"],
            submitters=[
                SubmitterResult(
                    **{**s, "year_scores": {int(y): v for y, v in s["year_scores"].items()}}
                )
                for s in payload.get("submitters", [])
            ],
        )

    @property
    def qualified(self) -> list[str]:
        return [s.submitter_id for s in self.submitters if s.qualified]

    @property
    def ranked(self) -> list[SubmitterResult]:
        return [s for s in self.submitters if s.rank is not None]


# ─── validation ──────────────────────────────────────────────────────────


def validate_submission(
    models: dict,
    allowed_years: Iterable[int],
    require_pinned_revision: bool = True,
) -> list[int]:
    """Validate a submission manifest; return the list of missing years.

    Raises :class:`SubmissionError` on anything structurally wrong. Missing
    years are not an error — they simply score worst-possible later — so they
    are returned for the caller to warn about.
    """
    if not isinstance(models, dict):
        raise SubmissionError("models must be a dict of {year: model_ref}")

    allowed = sorted(allowed_years)
    if not allowed:
        raise SubmissionError("allowed_years must not be empty")

    for raw_year, raw_ref in models.items():
        try:
            year = int(raw_year)
        except (TypeError, ValueError):
            raise SubmissionError(f"Year key must be an integer, got {raw_year!r}")
        if year not in allowed:
            raise SubmissionError(f"Year {year} not in {allowed[0]}-{allowed[-1]}")

        ref = ModelRef.parse(raw_ref)
        if ref.scheme == "local":
            if not os.path.isdir(ref.location):
                raise SubmissionError(f"Local model directory does not exist: {ref.location}")
        elif require_pinned_revision and not ref.is_pinned:
            raise SubmissionError(
                f"Invalid repo format: {raw_ref} (expected owner/repo@<40-char commit SHA>)"
            )

    return [y for y in allowed if str(y) not in models]
