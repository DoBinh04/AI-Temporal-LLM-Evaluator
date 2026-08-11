"""Spectral comparison — telling a copy from an independent model.

The primitive is the singular-value spectrum of each 2D weight matrix, and
the question it answers is: is this model a lightly-modified copy of that one?

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
from typing import Optional, Sequence

import torch

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


class SvdGate:
    """Per-year rejection gate: too close to a known baseline → the year fails.

    Holds pre-computed baseline spectra keyed by cutoff year. A candidate is
    compared against every baseline for that year and judged on the *minimum*
    distance — being far from one published variant does not help if it is a
    copy of another. A year with no baselines passes by construction.
    """

    def __init__(
        self,
        baselines: Optional[dict[int, list[dict[str, torch.Tensor]]]] = None,
        threshold: float = DEFAULT_SVD_THRESHOLD,
        top_ratio: float = DEFAULT_SVD_TOP_RATIO,
    ):
        self.baselines = baselines or {}
        self.threshold = threshold
        self.top_ratio = top_ratio

    def __bool__(self) -> bool:
        return any(self.baselines.values())

    def add_baseline(self, year: int, spectra: dict[str, torch.Tensor]) -> None:
        self.baselines.setdefault(year, []).append(spectra)

    def check(
        self, candidate: dict[str, torch.Tensor], year: int
    ) -> tuple[bool, float]:
        """Gate one candidate for one year. Returns `(passed, min_distance)`.

        Incomparable baselines (no shared matrix shapes) cannot be copies of
        the candidate, so they do not fail it.
        """
        distances = [
            distance
            for baseline in self.baselines.get(year, [])
            if (distance := compare_spectra(candidate, baseline, self.top_ratio)) is not None
        ]
        if not distances:
            return True, 1.0
        min_distance = min(distances)
        return min_distance >= self.threshold, min_distance


def dedup_by_svd(
    spectra: dict[str, dict[str, torch.Tensor]],
    precedence: Optional[Sequence[str]] = None,
    threshold: float = DEFAULT_SVD_THRESHOLD,
    top_ratio: float = DEFAULT_SVD_TOP_RATIO,
) -> set[str]:
    """Drop near-identical models from a set, keeping the first of each pair.

    `precedence` is the order in which claims are honoured — in a submission
    race that is submission time, earliest first; by default it is the given
    dict order. Returns the ids that survive.
    """
    ordered = list(dict.fromkeys(
        k for k in (precedence if precedence is not None else spectra) if k in spectra
    ))
    accepted: list[str] = []

    for key in ordered:
        duplicate_of = next(
            (
                kept
                for kept in accepted
                if (d := compare_spectra(spectra[key], spectra[kept], top_ratio)) is not None
                and d < threshold
            ),
            None,
        )
        if duplicate_of is None:
            accepted.append(key)
        else:
            logger.warning(f"{key} is a spectral copy of {duplicate_of}, dropped")

    logger.info(f"SVD dedup: {len(spectra)} models -> {len(accepted)} distinct")
    return set(accepted)


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



