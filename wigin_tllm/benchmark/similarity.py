"""How close is this model to another one?

Compares the **singular value spectra** of weight matrices rather than the
weights themselves. A copy disguised as ``W' = P·W·Q`` with orthogonal ``P``,
``Q`` — compensated in an adjacent layer so behaviour is unchanged — has
almost zero cosine similarity to the original, but singular values are
invariant under orthogonal transforms, so the spectrum still matches.

Two things an author uses this for:

* *Will my model be flagged as a copy of a published checkpoint?* Compare
  against the references before submitting.
* *Are my own year-models too alike?* Compare them against each other; a
  chain of near-identical checkpoints is a sign the yearly cutoffs are not
  actually changing the weights.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..config import EvaluationConfig
from ..models.store import free_device_memory, resolve_model
from ..scoring.svd_gate import SvdGate, compare_spectra, dedup_by_svd, load_state_dict, svd_spectra
from ..types import ModelRef

logger = logging.getLogger(__name__)


@dataclass
class Comparison:
    """One model measured against one reference."""

    reference: str
    distance: Optional[float]
    threshold: float

    @property
    def too_close(self) -> bool:
        """Below the threshold reads as "the same model"."""
        return self.distance is not None and self.distance < self.threshold

    @property
    def comparable(self) -> bool:
        """False when the two share no weight matrices of the same shape."""
        return self.distance is not None


@dataclass
class SimilarityReport:
    model: str
    comparisons: list[Comparison] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when nothing looks like a copy."""
        return not self.error and not any(c.too_close for c in self.comparisons)

    @property
    def closest(self) -> Optional[Comparison]:
        measured = [c for c in self.comparisons if c.comparable]
        return min(measured, key=lambda c: c.distance) if measured else None


def _spectra_of(raw_ref: str, device):
    """Read a model's weights and reduce them to a spectrum per matrix.

    Only the weights are read — no config is executed and the model is never
    instantiated, so this works on any checkpoint the loader could not handle.
    """
    ref = ModelRef.parse(raw_ref)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = resolve_model(ref, tmpdir)
        state = load_state_dict(path, device)
        try:
            return svd_spectra(state)
        finally:
            del state
            free_device_memory()


def compare(
    model_ref: str,
    references: Sequence[str],
    config: EvaluationConfig,
    device,
) -> SimilarityReport:
    """Measure one model against a list of references."""
    report = SimilarityReport(model=str(ModelRef.parse(model_ref)))
    if not references:
        return report

    try:
        subject = _spectra_of(model_ref, device)
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        return report

    try:
        for raw_reference in references:
            name = ModelRef.parse(raw_reference).short
            logger.info(f"Comparing against {name}...")
            try:
                reference_spectra = _spectra_of(raw_reference, device)
            except Exception as e:
                logger.warning(f"Could not read {name}: {type(e).__name__}: {e}")
                report.comparisons.append(Comparison(name, None, config.svd_threshold))
                continue

            distance = compare_spectra(subject, reference_spectra, config.svd_top_ratio)
            report.comparisons.append(Comparison(name, distance, config.svd_threshold))
            del reference_spectra
    finally:
        del subject
        free_device_memory()

    return report


def build_gate(
    references: dict[int, list[str]] | Sequence[str],
    years: Sequence[int],
    config: EvaluationConfig,
    device,
) -> SvdGate:
    """Load reference spectra into a per-year gate for stage 1.

    A reference that cannot be read is logged and left out — the gate fails
    open on it, which costs detection, not a developer's whole run. A shared
    reference (a bare list, or the same ref across years) is only read once.
    """
    gate = SvdGate(threshold=config.svd_threshold, top_ratio=config.svd_top_ratio)
    per_year = {
        year: (references.get(year, []) if isinstance(references, dict) else list(references))
        for year in years
    }

    spectra_cache: dict[str, Optional[dict]] = {}
    for year, refs in per_year.items():
        for raw_ref in refs:
            if raw_ref not in spectra_cache:
                try:
                    spectra_cache[raw_ref] = _spectra_of(raw_ref, device)
                except Exception as e:
                    logger.warning(
                        f"SVD gate: could not read {raw_ref} ({type(e).__name__}: {e}), "
                        "skipping this baseline"
                    )
                    spectra_cache[raw_ref] = None
            if spectra_cache[raw_ref] is not None:
                gate.add_baseline(year, spectra_cache[raw_ref])
    return gate


@dataclass
class PairwiseReport:
    """All-pairs spectral comparison over a set of models."""

    ids: list[str] = field(default_factory=list)
    measured: list[str] = field(default_factory=list)
    pairs: list[tuple[str, str, Optional[float]]] = field(default_factory=list)
    accepted: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    @property
    def duplicates(self) -> list[str]:
        """Measured ids that read as copies of an earlier-listed model.

        Only measured ids can be judged — a model that could not be read is
        an error, not a copy.
        """
        return [i for i in self.measured if i not in self.accepted]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.duplicates


def pairwise(
    raw_refs: Sequence[str],
    config: EvaluationConfig,
    device,
) -> PairwiseReport:
    """Compare a set of models against each other.

    The order of `raw_refs` is the precedence order: when two models read as
    the same, the one listed first keeps its place. Used to answer both "are
    my year checkpoints actually different models?" and "which of these
    submissions are copies of each other?".
    """
    report = PairwiseReport(ids=[str(ModelRef.parse(r)) for r in raw_refs])

    spectra: dict[str, dict] = {}
    try:
        for raw_ref, model_id in zip(raw_refs, report.ids):
            if model_id in spectra:
                continue  # the same ref listed twice is one model, not a pair
            try:
                spectra[model_id] = _spectra_of(raw_ref, device)
            except Exception as e:
                report.errors.append(f"{model_id}: {type(e).__name__}: {e}")

        report.measured = list(dict.fromkeys(i for i in report.ids if i in spectra))
        measured = report.measured
        for a in range(len(measured)):
            for b in range(a + 1, len(measured)):
                report.pairs.append((
                    measured[a], measured[b],
                    compare_spectra(spectra[measured[a]], spectra[measured[b]],
                                    config.svd_top_ratio),
                ))

        report.accepted = dedup_by_svd(
            spectra, precedence=measured,
            threshold=config.svd_threshold, top_ratio=config.svd_top_ratio,
        )
    finally:
        spectra.clear()
        free_device_memory()

    return report


def references_for_year(
    references: dict[int, list[str]] | Sequence[str], year: Optional[int]
) -> list[str]:
    """Flatten a reference spec into the list that applies to one year.

    Accepts either a bare list of model references or a `{year: [refs]}`
    mapping, so an author can point at one model or at a whole published
    family.
    """
    if isinstance(references, dict):
        if year is not None:
            return list(references.get(year, []))
        return [ref for refs in references.values() for ref in refs]
    return list(references)
