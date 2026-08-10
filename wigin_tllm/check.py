"""Pre-flight check for a single model.

Answers the two questions a model author has before submitting:

1. **Will my artefact even be accepted?** It loads with
   `trust_remote_code=False`, it is within the size and parameter limits, its
   tokenizer round-trips, and scoring and generation actually work on it.
2. **How would it score?** Given probe sets, the per-year verdict with the
   numbers behind it and a diagnosis when it fails.

Nothing here touches anti-copy or ranking — this looks at one model in
isolation, so it can be run as often as you like while iterating.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .config import EvaluationConfig
from .datasource.base import DataSource
from .models.loader import ModelLoadError, load_model
from .models.store import (
    count_model_params,
    free_device_memory,
    get_device,
    get_model_size_bytes,
    resolve_model,
    verify_pinned_revision,
)
from .scoring.leak import assess_year, score_items
from .types import BenchmarkItem, ModelRef, YearAssessment

logger = logging.getLogger(__name__)

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

# Timeless facts, used only to check that the scoring path is wired up. Each
# entry pairs a plausible continuation with a nonsensical one: a correctly
# wired model must prefer the former, whatever its cutoff year.
_SMOKE_PROBES: list[tuple[str, str, str]] = [
    ("The capital city of France is", "Paris", "Jupiter"),
    ("Water freezes at zero degrees", "Celsius", "guitar"),
    ("The largest ocean on Earth is the", "Pacific", "sandwich"),
    ("Two plus two equals", "four", "purple"),
]

_SMOKE_PROMPTS = [
    "The capital city of France is",
    "The largest ocean on Earth is the",
]


@dataclass
class CheckItem:
    """One line of the pre-flight report."""

    name: str
    level: str
    detail: str = ""
    hint: str = ""

    @property
    def failed(self) -> bool:
        return self.level == FAIL


@dataclass
class YearCheck:
    """A year's verdict plus the reason behind it."""

    year: int
    assessment: YearAssessment
    normalized: float
    diagnosis: str

    @property
    def passed(self) -> bool:
        return self.assessment.passed


@dataclass
class ModelReport:
    ref: str
    artifact: list[CheckItem] = field(default_factory=list)
    years: list[YearCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing blocks the model from being evaluated."""
        return not any(item.failed for item in self.artifact)

    @property
    def years_passed(self) -> int:
        return sum(1 for y in self.years if y.passed)


# ─── artefact checks ─────────────────────────────────────────────────────


def _check_limits(ref: ModelRef, config: EvaluationConfig) -> list[CheckItem]:
    items: list[CheckItem] = []

    size = get_model_size_bytes(ref)
    if size < 0:
        items.append(
            CheckItem("weight size", WARN, "could not be determined",
                      "the evaluator will not be able to pre-screen this model")
        )
    else:
        over = size > config.max_model_bytes
        items.append(
            CheckItem(
                "weight size", FAIL if over else OK,
                f"{size / 1024**2:.1f} MB (limit {config.max_model_bytes / 1024**3:.1f} GiB)",
                "shrink the model or raise max_model_bytes" if over else "",
            )
        )

    if ref.scheme == "hf" and config.require_pinned_revision:
        try:
            pinned = verify_pinned_revision(ref)
        except Exception as e:
            items.append(
                CheckItem("pinned revision", WARN, f"could not verify ({type(e).__name__})")
            )
        else:
            items.append(
                CheckItem(
                    "pinned revision", OK if pinned else FAIL,
                    ref.revision or "(none)",
                    "" if pinned else "pin the reference to a full 40-character commit SHA",
                )
            )

    return items


def _check_tokenizer(model) -> CheckItem:
    sample = "The capital city of France is Paris"
    try:
        ids = model.encode(sample)
        if not ids:
            return CheckItem("tokenizer", FAIL, "encode returned no tokens",
                             "the tokenizer files are missing or unreadable")
        text = model.decode(ids)
    except Exception as e:
        return CheckItem("tokenizer", FAIL, f"{type(e).__name__}: {e}",
                         "ship a standard tokenizer.json alongside the weights")

    if not text.strip():
        return CheckItem("tokenizer", FAIL, "decode returned nothing",
                         "encode/decode are not round-tripping")
    return CheckItem("tokenizer", OK, f"{len(ids)} tokens, round-trip fine")


def _check_scoring(model, device) -> CheckItem:
    """Scoring must prefer a plausible continuation over a nonsensical one.

    A wiring check, not a quality one: a model that ranks nonsense just as
    highly usually has a tokenizer that does not match its weights.

    Probes whose two continuations encode identically are skipped — that
    happens when a restricted vocabulary maps both to the same unknown token,
    and says nothing about the model.
    """
    usable, nonsense_items = [], []
    for prompt, plausible, nonsense in _SMOKE_PROBES:
        try:
            if model.encode(f" {plausible}") == model.encode(f" {nonsense}"):
                continue  # the tokenizer cannot tell these apart
        except Exception:
            continue
        usable.append(BenchmarkItem(prompt=prompt, phrase=plausible))
        nonsense_items.append(BenchmarkItem(prompt=prompt, phrase=nonsense))

    if not usable:
        return CheckItem(
            "scoring", SKIP,
            "not applicable: this vocabulary cannot represent the sample words",
            "the consistency check below is the real test for this model",
        )

    try:
        good = score_items(model, device, usable)
        bad = score_items(model, device, nonsense_items)
    except Exception as e:
        return CheckItem("scoring", FAIL, f"{type(e).__name__}: {e}",
                         "the model could not be scored by the pipeline")

    if any(s != s for s in good + bad):  # NaN
        return CheckItem("scoring", FAIL, "produced NaN",
                         "check for uninitialised or corrupted weights")

    wins = sum(1 for g, b in zip(good, bad) if g > b)
    detail = f"prefers the sensible continuation in {wins}/{len(good)} probes"
    if wins == len(good):
        return CheckItem("scoring", OK, detail)
    return CheckItem(
        "scoring", WARN, detail,
        "a model that cannot separate sense from nonsense usually has a "
        "tokenizer that does not match its weights",
    )


def _check_generation(model, max_new_tokens: int) -> CheckItem:
    try:
        answers = [model.generate(p, max_new_tokens=max_new_tokens) for p in _SMOKE_PROMPTS]
    except Exception as e:
        return CheckItem("generation", FAIL, f"{type(e).__name__}: {e}",
                         "stage 2 needs generate() to work")

    if not any(a.strip() for a in answers):
        return CheckItem("generation", WARN, "produced only empty output",
                         "this model will lose every quality duel")
    preview = next(a for a in answers if a.strip()).replace("\n", " ")[:60]
    return CheckItem("generation", OK, f'e.g. "{preview}"')


# ─── consistency checks ──────────────────────────────────────────────────


def _diagnose(assessment: YearAssessment) -> str:
    unknown, known = assessment.unknown, assessment.known

    if assessment.passed:
        return "blind to the future, fluent in its own era"
    if unknown.recognised and known.recognised:
        return (
            "recognises post-cutoff facts as readily as pre-cutoff ones — the "
            "training data reaches beyond the cutoff"
        )
    if unknown.recognised and not known.recognised:
        return (
            "recognises the future but not the past — check that the cutoff "
            "year on this checkpoint is the one you think it is"
        )
    if known.total == 0:
        return "no `known` probes for this year, so the requirement cannot be met"
    return (
        "recognises neither past nor future — the model has not learned its own "
        f"era, or epsilon ({known.epsilon:.2f}) is stricter than this "
        "vocabulary can reach"
    )


def _check_years(
    model,
    device,
    datasource: DataSource,
    years: Sequence[int],
    config: EvaluationConfig,
) -> list[YearCheck]:
    from .scoring.aggregate import normalize_leak_score

    checks = []
    for year in years:
        logger.info(f"Scoring year {year}...")
        assessment = assess_year(
            model,
            device,
            datasource.get_benchmark(year, "unknown"),
            datasource.get_benchmark(year, "known"),
        )
        checks.append(
            YearCheck(
                year=year,
                assessment=assessment,
                normalized=normalize_leak_score(
                    assessment.score, config.min_eval_score, config.leak_best_score
                ),
                diagnosis=_diagnose(assessment),
            )
        )
    return checks


# ─── entry point ─────────────────────────────────────────────────────────


def check_model(
    model_ref: str,
    config: Optional[EvaluationConfig] = None,
    datasource: Optional[DataSource] = None,
    years: Optional[Sequence[int]] = None,
) -> ModelReport:
    """Run every pre-flight check available for one model.

    With no `datasource` only the artefact checks run, which needs no probe
    data and is the fastest way to confirm a model is submittable at all.
    """
    config = config or EvaluationConfig()
    ref = ModelRef.parse(model_ref)
    report = ModelReport(ref=str(ref))
    device = get_device(config.device)

    report.artifact.extend(_check_limits(ref, config))

    model = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = resolve_model(ref, tmpdir)
            report.artifact.append(CheckItem("architecture", OK, _architecture_of(path)))

            model, _ = load_model(path, device)
            report.artifact.append(
                CheckItem("loads without trust_remote_code", OK, f"on {device}")
            )

            params = count_model_params(model)
            over = params > config.max_parameters
            report.artifact.append(
                CheckItem(
                    "parameters", FAIL if over else OK,
                    f"{params / 1e6:.1f}M (limit {config.max_parameters / 1e9:.2f}B)",
                    "shrink the model or raise max_parameters" if over else "",
                )
            )

            report.artifact.append(_check_tokenizer(model))
            report.artifact.append(_check_scoring(model, device))
            report.artifact.append(_check_generation(model, config.quality_max_new_tokens))

            if datasource is not None:
                report.years = _check_years(
                    model, device, datasource,
                    years or datasource.get_years(datasource.get_current_round()),
                    config,
                )

    except ModelLoadError as e:
        report.artifact.append(
            CheckItem("loads without trust_remote_code", FAIL, str(e),
                      "the evaluator never executes code from a model directory")
        )
    except FileNotFoundError as e:
        report.artifact.append(CheckItem("resolve", FAIL, str(e)))
    except Exception as e:
        report.artifact.append(CheckItem("resolve", FAIL, f"{type(e).__name__}: {e}"))
    finally:
        del model
        free_device_memory()

    return report


def _architecture_of(path: str) -> str:
    import json

    try:
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
    except Exception:
        return "unknown"
    return config.get("model_type") or ("chronogpt" if "model_dim" in config else "unknown")
