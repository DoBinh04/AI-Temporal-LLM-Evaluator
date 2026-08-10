"""Evaluation configuration.

Every scoring knob in one explicit, versionable dataclass. Build it in code,
load it from JSON, or override individual fields from the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

# The score for a (submitter, year) pair that is missing, errored, oversized,
# or failed a gate. Deliberately the worst attainable value: real scores are
# negative and lower is better.
WORST_SCORE = 0.0

DEFAULT_YEARS = list(range(2013, 2025))

ENV_PREFIX = "WIGIN_TLLM_"


@dataclass
class EvaluationConfig:
    # ── Resource limits ──────────────────────────────────────────────────
    max_model_bytes: int = 10 * 1024**3
    max_parameters: int = 2_000_000_000
    max_eval_seconds: float = 3600.0

    # ── Stage 1: consistency ─────────────────────────────────────────────
    # A submitter advances when its mean year score is strictly below
    # `min_eval_score`. `leak_best_score` is the score normalising to 1.0.
    min_eval_score: float = -3.0
    leak_best_score: float = -6.0
    top_n_for_quality: int = 10

    # ── Stage 2: quality duels ───────────────────────────────────────────
    quality_enabled: bool = True
    quality_max_new_tokens: int = 50
    # Cutoff years sampled for quality: the oldest is always included, the
    # rest are drawn at random so effort cannot target a predictable year.
    quality_year_samples: int = 2
    # Seed for the year draw and the A/B swap. None = nondeterministic.
    quality_seed: Optional[int] = None

    # ── Final score ──────────────────────────────────────────────────────
    leak_weight: float = 0.7
    quality_weight: float = 0.3

    # ── Anti-copy ────────────────────────────────────────────────────────
    duplicate_weight_check: bool = True
    svd_gate_enabled: bool = True
    svd_dedup_enabled: bool = True
    svd_threshold: float = 0.01
    svd_top_ratio: float = 0.25
    # Submitters exempt from both SVD checks — needed when whoever publishes
    # the baselines also submits, since they would disqualify themselves.
    svd_exempt_submitters: list[str] = field(default_factory=list)

    # ── Submission validation ────────────────────────────────────────────
    require_pinned_revision: bool = True

    # ── Runtime ──────────────────────────────────────────────────────────
    # None = auto-detect (cuda > mps > cpu).
    device: Optional[str] = None
    # Where the SQLite cache and any downloaded baselines live.
    data_dir: str = "./.eval_data"

    # ─────────────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.quality_year_samples < 1:
            raise ValueError("quality_year_samples must be >= 1")
        if self.top_n_for_quality < 1:
            raise ValueError("top_n_for_quality must be >= 1")
        if self.min_eval_score == self.leak_best_score:
            raise ValueError("min_eval_score and leak_best_score must differ")

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
