"""Will this artefact even be scored?

Everything that can reject a model before a single probe is run: size,
architecture, parameter count, tokenizer, and whether scoring and generation
work on it at all.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from ..config import EvaluationConfig
from ..models.loader import ModelLoadError, load_model
from ..models.store import (
    count_model_params,
    free_device_memory,
    get_model_size_bytes,
    resolve_model,
    verify_pinned_revision,
)
from ..scoring.leak import score_items
from ..types import BenchmarkItem, ModelRef

logger = logging.getLogger(__name__)

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

# Timeless facts, used only to check that the scoring path is wired up. Each
# pairs a plausible continuation with a nonsensical one: a correctly wired
# model must prefer the former, whatever its cutoff year.
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
    """One line of the artefact report."""

    name: str
    level: str
    detail: str = ""
    hint: str = ""

    @property
    def failed(self) -> bool:
        return self.level == FAIL


@dataclass
class ArtifactReport:
    items: list[CheckItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(item.failed for item in self.items)

    def add(self, item: CheckItem) -> None:
        self.items.append(item)


def architecture_of(path: str) -> str:
    try:
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
    except Exception:
        return "unknown"
    return config.get("model_type") or ("chronogpt" if "model_dim" in config else "unknown")


def check_limits(ref: ModelRef, config: EvaluationConfig) -> list[CheckItem]:
    """Checks that need only the reference, before anything is downloaded."""
    items: list[CheckItem] = []

    size = get_model_size_bytes(ref)
    if size < 0:
        items.append(CheckItem("weight size", WARN, "could not be determined"))
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
            items.append(CheckItem("pinned revision", WARN, f"could not verify ({type(e).__name__})"))
        else:
            items.append(
                CheckItem(
                    "pinned revision", OK if pinned else FAIL, ref.revision or "(none)",
                    "" if pinned else "pin the reference to a full 40-character commit SHA",
                )
            )
    return items


def check_loaded(model, path: str, device, config: EvaluationConfig) -> list[CheckItem]:
    """Checks that need the model in memory."""
    items = [
        CheckItem("architecture", OK, architecture_of(path)),
        CheckItem("loads without trust_remote_code", OK, f"on {device}"),
    ]

    params = count_model_params(model)
    over = params > config.max_parameters
    items.append(
        CheckItem(
            "parameters", FAIL if over else OK,
            f"{params / 1e6:.1f}M (limit {config.max_parameters / 1e9:.2f}B)",
            "shrink the model or raise max_parameters" if over else "",
        )
    )
    items.append(_check_tokenizer(model))
    items.append(_check_scoring(model, device))
    items.append(_check_generation(model, config.quality_max_new_tokens))
    return items


def load_failure(error: Exception) -> CheckItem:
    if isinstance(error, ModelLoadError):
        return CheckItem(
            "loads without trust_remote_code", FAIL, str(error),
            "no code from a model directory is ever executed",
        )
    return CheckItem("resolve", FAIL, f"{type(error).__name__}: {error}")


# ─── individual checks ───────────────────────────────────────────────────


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
    happens with a restricted vocabulary and says nothing about the model.
    """
    plausible, nonsense = [], []
    for prompt, good, bad in _SMOKE_PROBES:
        try:
            if model.encode(f" {good}") == model.encode(f" {bad}"):
                continue
        except Exception:
            continue
        plausible.append(BenchmarkItem(prompt=prompt, phrase=good))
        nonsense.append(BenchmarkItem(prompt=prompt, phrase=bad))

    if not plausible:
        return CheckItem(
            "scoring", SKIP,
            "not applicable: this vocabulary cannot represent the sample words",
            "the consistency stage is the real test for this model",
        )

    try:
        good_scores = score_items(model, device, plausible)
        bad_scores = score_items(model, device, nonsense)
    except Exception as e:
        return CheckItem("scoring", FAIL, f"{type(e).__name__}: {e}",
                         "the model could not be scored")

    if any(s != s for s in good_scores + bad_scores):  # NaN
        return CheckItem("scoring", FAIL, "produced NaN",
                         "check for uninitialised or corrupted weights")

    wins = sum(1 for g, b in zip(good_scores, bad_scores) if g > b)
    detail = f"prefers the sensible continuation in {wins}/{len(good_scores)} probes"
    if wins == len(good_scores):
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
                         "the quality stage needs generate() to work")

    if not any(a.strip() for a in answers):
        return CheckItem("generation", WARN, "produced only empty output",
                         "this model will lose every quality duel")
    preview = next(a for a in answers if a.strip()).replace("\n", " ")[:60]
    return CheckItem("generation", OK, f'e.g. "{preview}"')


def inspect_model(
    ref: ModelRef, config: EvaluationConfig, device
) -> tuple[ArtifactReport, Optional[str]]:
    """Resolve and inspect one model. Returns the report and its local path.

    The path is None when the model could not be loaded, in which case the
    report says why.
    """
    import tempfile

    report = ArtifactReport()
    for item in check_limits(ref, config):
        report.add(item)

    model = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = resolve_model(ref, tmpdir)
            model, _ = load_model(path, device)
            for item in check_loaded(model, path, device, config):
                report.add(item)
            return report, path
    except Exception as e:
        report.add(load_failure(e))
        return report, None
    finally:
        del model
        free_device_memory()
