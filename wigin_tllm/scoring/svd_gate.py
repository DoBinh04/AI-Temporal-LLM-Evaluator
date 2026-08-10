"""SVD similarity gate — reject models that are copies.

Two independent checks share one primitive, the singular-value spectrum of
each 2D weight matrix:

1. **Baseline gate** — is this model a lightly-modified copy of a published
   reference model?
2. **Pairwise dedup** — did two submitters send the same model?

Why singular values rather than cosine similarity of the weights? Because a
copy can be disguised as ``W' = P·W·Q`` with orthogonal/permutation ``P``,
``Q`` (compensated in an adjacent layer), which drives weight cosine
similarity to nearly zero while leaving behaviour intact. Singular values are
invariant under orthogonal transforms: ``σ(P·W·Q) = σ(W)``. The spectrum
survives the disguise.

Spectra are also L2-normalised before comparison so a global rescale of the
weights cannot slip past either.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

import torch

from ..types import ModelRef

logger = logging.getLogger(__name__)

DEFAULT_SVD_THRESHOLD = 0.01
DEFAULT_SVD_TOP_RATIO = 0.25


def svd_spectra(state_dict: dict) -> dict[str, torch.Tensor]:
    """Singular-value spectrum of every 2D weight matrix in a state dict."""
    return {
        name: torch.linalg.svdvals(param.float())
        for name, param in state_dict.items()
        if param.ndim == 2 and min(param.shape) > 1
    }


def compare_spectra(
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    top_ratio: float = DEFAULT_SVD_TOP_RATIO,
) -> Optional[float]:
    """Mean L2 distance between the two spectra. None if not comparable.

    Only the largest `top_ratio` of singular values are compared — they carry
    the structure of the matrix, while the tail is mostly noise.
    """
    common = sorted(set(reference) & set(candidate))
    if not common:
        return None

    distances = []
    for name in common:
        sv_ref = reference[name]
        sv_cand = candidate[name]
        if sv_ref.shape != sv_cand.shape:
            continue
        k = max(1, int(len(sv_ref) * top_ratio))
        ref_norm = sv_ref[:k] / (torch.norm(sv_ref[:k]) + 1e-10)
        cand_norm = sv_cand[:k] / (torch.norm(sv_cand[:k]) + 1e-10)
        distances.append(torch.norm(ref_norm - cand_norm).item())

    if not distances:
        return None
    return sum(distances) / len(distances)


def load_state_dict(path: str, device) -> dict[str, torch.Tensor]:
    """Read weights from a model directory without executing any of its code."""
    safetensors_path = os.path.join(path, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt") as f:
            return {k: f.get_tensor(k).to(device) for k in f.keys()}

    bin_path = os.path.join(path, "pytorch_model.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"No weight file in {path}")
    # weights_only=True: tensors are deserialised, arbitrary pickle payloads
    # are not.
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    return {k: v.to(device) for k, v in state.items()}


class SvdGate:
    """Rejects candidates whose spectra sit too close to a known baseline.

    Baselines are injected rather than hardcoded: which reference models
    count as "must not be copied" is deployment policy, not scoring logic.
    Construct with an empty mapping to disable the gate entirely.
    """

    def __init__(
        self,
        baselines: Optional[dict[int, list[str]]] = None,
        threshold: float = DEFAULT_SVD_THRESHOLD,
        top_ratio: float = DEFAULT_SVD_TOP_RATIO,
        cache_dir: Optional[str] = None,
    ):
        self.baselines = baselines or {}
        self.threshold = threshold
        self.top_ratio = top_ratio
        self.cache_dir = cache_dir or os.path.join(os.environ.get("HF_HOME", "/tmp"), "temporal_eval_baselines")
        self._spectra: dict[int, list[dict[str, torch.Tensor]]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.baselines)

    def load(self, years: Iterable[int], device) -> None:
        """Fetch and spectrally summarise every baseline. Call once, up front."""
        if not self.enabled:
            logger.info("SVD baseline gate disabled (no baselines configured)")
            return

        from ..models.store import resolve_model

        os.makedirs(self.cache_dir, exist_ok=True)
        loaded = 0
        for year in years:
            if self._spectra.get(year):
                continue  # already loaded — don't re-fetch
            refs = self.baselines.get(year, [])
            self._spectra[year] = []
            for i, raw_ref in enumerate(refs):
                ref = ModelRef.parse(raw_ref)
                path = resolve_model(ref, os.path.join(self.cache_dir, f"baseline_{year}_{i}"))
                state = load_state_dict(path, device)
                self._spectra[year].append(svd_spectra(state))
                del state
                loaded += 1
                logger.info(f"Baseline {year} variant {i} SVD loaded")
        logger.info(f"All {loaded} baselines loaded")

    def unload(self) -> None:
        self._spectra.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def check(self, candidate_spectra: dict[str, torch.Tensor], year: int) -> tuple[bool, float]:
        """Returns `(passed, min_distance)`. Passes when no baseline is close."""
        baselines = self._spectra.get(year, [])
        if not baselines:
            return True, 1.0

        min_dist = None
        for reference in baselines:
            dist = compare_spectra(candidate_spectra, reference, self.top_ratio)
            if dist is not None and (min_dist is None or dist < min_dist):
                min_dist = dist

        if min_dist is None:
            return True, 1.0
        return min_dist >= self.threshold, min_dist


def dedup_by_svd(
    spectra_by_submitter: dict[str, dict[str, torch.Tensor]],
    submitted_at: dict[str, str],
    threshold: float = DEFAULT_SVD_THRESHOLD,
    top_ratio: float = DEFAULT_SVD_TOP_RATIO,
) -> set[str]:
    """Drop submitters whose spectra match an earlier submitter's.

    Ordering is by submission time, so the earliest submitter of a given model
    keeps it and later ones are treated as copies.
    """
    ordered = sorted(spectra_by_submitter, key=lambda m: submitted_at.get(m, "9999"))
    accepted: list[str] = []

    for submitter_id in ordered:
        duplicate_of = None
        for kept in accepted:
            dist = compare_spectra(spectra_by_submitter[submitter_id], spectra_by_submitter[kept], top_ratio)
            if dist is not None and dist < threshold:
                duplicate_of = (kept, dist)
                break
        if duplicate_of:
            kept, dist = duplicate_of
            logger.warning(f"{submitter_id} is a copy of {kept} (svd_dist={dist:.6f}), removing")
        else:
            accepted.append(submitter_id)

    logger.info(f"SVD dedup: {len(spectra_by_submitter)} submitters -> {len(accepted)} unique")
    return set(accepted)
