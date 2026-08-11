"""Data model.

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
    """A model manifest is structurally invalid."""


# ─── model references ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelRef:
    """A reference to a set of model weights.

    Two schemes:

    ``hf``     ``owner/repo@<40-hex-sha>`` (``hf:`` prefix optional)
    ``local``  ``local:/path/to/model_dir``
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
        # stray '@' earlier stays part of the location and is rejected there.
        if "@" in body:
            location, revision = body.rsplit("@", 1)
        else:
            location, revision = body, None
        return cls(scheme="hf", location=location, revision=revision or None, raw=ref)

    def __str__(self) -> str:
        return self.raw

    @property
    def short(self) -> str:
        """A name short enough for a table column.

        Model directories are often laid out `<name>/<year>`, where the last
        segment alone identifies nothing — keep the parent in that case.
        """
        parts = [p for p in self.location.rstrip("/").split("/") if p]
        if not parts:
            return self.location
        if len(parts) >= 2 and parts[-1].isdigit():
            return "/".join(parts[-2:])
        return parts[-1]


# ─── corpus ──────────────────────────────────────────────────────────────


@dataclass
class Fact:
    """One dated statement: the raw material a probe set is built from."""

    year: int
    prompt: str
    phrase: str

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(year=int(d["year"]), prompt=d["prompt"], phrase=d["phrase"])


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [asdict(i) for i in self.items],
            "threshold": self.threshold,
            "epsilon": self.epsilon,
        }


@dataclass
class CompletionPrompt:
    """A stage-2 prompt: incomplete text the model must continue.

    Deliberately not a question — models are scored on how well they carry on
    a passage, which is what a language model does natively and what a base
    checkpoint can be compared on fairly.
    """

    prompt: str
    category: str = ""
    reference: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "CompletionPrompt":
        return cls(
            prompt=d["prompt"],
            category=d.get("category", ""),
            reference=d.get("reference", ""),
        )


# ─── scoring results ─────────────────────────────────────────────────────


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

    @property
    def margin(self) -> float:
        """Distance from the threshold. Near zero means a coin-flip verdict."""
        return abs(self.ratio - self.threshold)


@dataclass
class YearAssessment:
    """The full picture for one cutoff year, both probe sets included."""

    unknown: ProbeResult
    known: ProbeResult
    passed: bool
    score: float


@dataclass
class YearScore:
    """One year of a consistency run, with the reason behind the verdict."""

    year: int
    model_ref: str
    assessment: Optional[YearAssessment] = None
    diagnosis: str = ""
    error: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.assessment and self.assessment.passed)

    @property
    def score(self) -> float:
        return self.assessment.score if self.assessment else 0.0


# ─── manifest ────────────────────────────────────────────────────────────


def validate_manifest(
    models: dict,
    allowed_years: Iterable[int],
    require_pinned_revision: bool = True,
) -> list[int]:
    """Validate a `{year: model_ref}` manifest; return the missing years.

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
