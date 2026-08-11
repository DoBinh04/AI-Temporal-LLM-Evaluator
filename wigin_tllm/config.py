"""Benchmark configuration.

Every knob in one explicit, versionable dataclass. Build it in code, load it
from JSON, or override individual fields from the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

# The score for a year that is missing, errored, oversized, or failed a gate.
# Deliberately the worst attainable value: real scores are negative and lower
# is better.
WORST_SCORE = 0.0

DEFAULT_YEARS = list(range(2013, 2025))

ENV_PREFIX = "WIGIN_TLLM_"


@dataclass
class EvaluationConfig:
    # ── Resource limits ──────────────────────────────────────────────────
    max_model_bytes: int = 10 * 1024**3
    max_parameters: int = 2_000_000_000
    # Stage-1 time budget per model, in seconds. The clock starts when the
    # model starts loading; years not reached in time score worst-possible.
    # None = no limit.
    max_eval_seconds: Optional[float] = None

    # ── Stage 1: consistency ─────────────────────────────────────────────
    # A model clears the bar when its mean year score is strictly below
    # `min_eval_score`. `leak_best_score` is the score normalising to 1.0 —
    # past it, more consistency work earns nothing.
    min_eval_score: float = -3.0
    leak_best_score: float = -6.0

    # ── Stage 2: quality duels ───────────────────────────────────────────
    quality_max_new_tokens: int = 50
    # Cutoff years sampled for quality: the oldest is always included, the
    # rest are drawn at random so effort cannot target a predictable year.
    quality_year_samples: int = 2
    # Seed for the year draw and the A/B swap. None = nondeterministic.
    quality_seed: Optional[int] = None

    # ── Final score ──────────────────────────────────────────────────────
    leak_weight: float = 0.7
    quality_weight: float = 0.3

    # ── Similarity ───────────────────────────────────────────────────────
    # Spectral distance below this reads as "the same model".
    svd_threshold: float = 0.01
    # Fraction of each spectrum compared — the head carries the structure.
    svd_top_ratio: float = 0.25

    # ── Corpus calibration ───────────────────────────────────────────────
    # Target fraction of post-cutoff probes an honest model may recognise.
    # Calibration picks `epsilon` to land under this with room to spare, so a
    # verdict is never decided by one probe crossing the line.
    probe_threshold: float = 0.25
    # Per-side overrides. The two requirements are asymmetric — a model must
    # recognise MOST of its own era but almost NONE of the future — so a
    # production probe set typically sets these apart (e.g. known 0.70,
    # unknown 0.10). None = use `probe_threshold` for that side.
    known_threshold: Optional[float] = None
    unknown_threshold: Optional[float] = None
    # How much headroom to leave: the calibrated hit rate aims for
    # `unknown_probe_threshold * calibration_margin`.
    calibration_margin: float = 0.5

    # ── Manifest validation ──────────────────────────────────────────────
    require_pinned_revision: bool = True

    # ── Runtime ──────────────────────────────────────────────────────────
    # None = auto-detect (cuda > mps > cpu).
    device: Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.quality_year_samples < 1:
            raise ValueError("quality_year_samples must be >= 1")
        if self.min_eval_score == self.leak_best_score:
            raise ValueError("min_eval_score and leak_best_score must differ")
        for name in ("probe_threshold", "known_threshold", "unknown_threshold"):
            value = getattr(self, name)
            if value is not None and not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def known_probe_threshold(self) -> float:
        return self.known_threshold if self.known_threshold is not None else self.probe_threshold

    @property
    def unknown_probe_threshold(self) -> float:
        return (
            self.unknown_threshold if self.unknown_threshold is not None else self.probe_threshold
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationConfig":
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "EvaluationConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_env(cls, prefix: str = ENV_PREFIX) -> "EvaluationConfig":
        """Build a config from `PREFIX<UPPERCASED_FIELD>` environment vars."""
        overrides = {
            f.name: _coerce(os.environ[f"{prefix}{f.name.upper()}"], str(f.type))
            for f in fields(cls)
            if f"{prefix}{f.name.upper()}" in os.environ
        }
        return cls(**overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(raw: str, hint: str) -> Any:
    """Parse an environment string according to a field's type annotation.

    `from __future__ import annotations` makes annotations plain strings, so
    this matches on the text of the hint rather than the type object.
    """
    if raw == "" and "Optional" in hint:
        return None
    if "bool" in hint:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if "list" in hint:
        return [part for part in (p.strip() for p in raw.split(",")) if part]
    if "float" in hint:
        return float(raw)
    if "int" in hint:
        return int(raw)
    return raw
