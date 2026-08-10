"""The SVD anti-copy gate.

The property that matters most is invariance: a copy disguised by an
orthogonal transform must still be recognised, because that is exactly the
attack that defeats cosine similarity on raw weights.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wigin_tllm.scoring.svd_gate import (  # noqa: E402
    SvdGate,
    compare_spectra,
    dedup_by_svd,
    svd_spectra,
)


def random_state(seed: int = 0, shape=(32, 32)) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {"layer.weight": torch.randn(*shape, generator=generator)}


# ─── spectrum extraction ─────────────────────────────────────────────────


def test_only_2d_matrices_are_summarised():
    state = {
        "w2d": torch.randn(8, 8),
        "bias": torch.randn(8),
        "degenerate": torch.randn(1, 8),
        "w3d": torch.randn(2, 4, 4),
    }
    assert set(svd_spectra(state)) == {"w2d"}


def test_spectrum_length_matches_smaller_dimension():
    spectra = svd_spectra({"w": torch.randn(16, 8)})
    assert spectra["w"].shape[0] == 8


# ─── comparison ──────────────────────────────────────────────────────────


def test_identical_weights_have_zero_distance():
    spectra = svd_spectra(random_state(1))
    assert compare_spectra(spectra, spectra) == pytest.approx(0.0, abs=1e-6)


def test_unrelated_weights_are_far_apart():
    a = svd_spectra(random_state(1))
    b = svd_spectra(random_state(2))
    assert compare_spectra(a, b) > 0.01


def test_orthogonal_transform_does_not_hide_a_copy():
    """W' = P·W·Q keeps the singular values, so the copy stays visible."""
    weight = random_state(3)["layer.weight"]
    q_left, _ = torch.linalg.qr(torch.randn(32, 32, generator=torch.Generator().manual_seed(11)))
    q_right, _ = torch.linalg.qr(torch.randn(32, 32, generator=torch.Generator().manual_seed(12)))
    disguised = q_left @ weight @ q_right

    # Element-wise the two matrices look unrelated...
    cosine = torch.nn.functional.cosine_similarity(
        weight.flatten(), disguised.flatten(), dim=0
    ).abs()
    assert cosine < 0.5
    # ...but the spectra match.
    distance = compare_spectra(
        svd_spectra({"layer.weight": disguised}), svd_spectra({"layer.weight": weight})
    )
    assert distance == pytest.approx(0.0, abs=1e-4)


def test_global_rescale_does_not_hide_a_copy():
    weight = random_state(4)["layer.weight"]
    distance = compare_spectra(
        svd_spectra({"layer.weight": weight * 7.5}), svd_spectra({"layer.weight": weight})
    )
    assert distance == pytest.approx(0.0, abs=1e-5)


def test_tiny_noise_still_reads_as_a_copy():
    weight = random_state(5)["layer.weight"]
    noised = weight + torch.randn_like(weight) * 1e-6
    distance = compare_spectra(
        svd_spectra({"layer.weight": noised}), svd_spectra({"layer.weight": weight})
    )
    assert distance < 0.01


def test_no_common_tensors_is_not_comparable():
    assert compare_spectra(svd_spectra({"a": torch.randn(8, 8)}),
                           svd_spectra({"b": torch.randn(8, 8)})) is None


def test_shape_mismatch_on_all_tensors_is_not_comparable():
    assert compare_spectra(svd_spectra({"w": torch.randn(8, 8)}),
                           svd_spectra({"w": torch.randn(16, 16)})) is None


# ─── baseline gate ───────────────────────────────────────────────────────


def test_gate_without_baselines_passes_everything():
    gate = SvdGate(baselines={})
    assert not gate.enabled
    passed, distance = gate.check(svd_spectra(random_state(6)), 2013)
    assert passed and distance == 1.0


def test_gate_rejects_a_baseline_copy():
    gate = SvdGate(baselines={2013: ["local:/unused"]})
    baseline = svd_spectra(random_state(7))
    gate._spectra = {2013: [baseline]}
    passed, distance = gate.check(baseline, 2013)
    assert not passed
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_gate_accepts_an_independent_model():
    gate = SvdGate(baselines={2013: ["local:/unused"]})
    gate._spectra = {2013: [svd_spectra(random_state(8))]}
    passed, distance = gate.check(svd_spectra(random_state(9)), 2013)
    assert passed and distance >= 0.01


def test_gate_uses_the_closest_baseline():
    """Matching any one baseline is enough to be rejected."""
    target = svd_spectra(random_state(10))
    gate = SvdGate(baselines={2013: ["a", "b"]})
    gate._spectra = {2013: [svd_spectra(random_state(11)), target]}
    passed, _ = gate.check(target, 2013)
    assert not passed


def test_unload_clears_state():
    gate = SvdGate(baselines={2013: ["x"]})
    gate._spectra = {2013: [svd_spectra(random_state(12))]}
    gate.unload()
    assert gate._spectra == {}


# ─── pairwise dedup ──────────────────────────────────────────────────────


def test_earliest_submitter_keeps_the_model():
    shared = svd_spectra(random_state(13))
    spectra = {"late": shared, "early": shared, "other": svd_spectra(random_state(14))}
    times = {"early": "2026-01-01T00:00", "late": "2026-01-02T00:00", "other": "2026-01-03T00:00"}
    assert dedup_by_svd(spectra, times) == {"early", "other"}


def test_dedup_keeps_distinct_models():
    spectra = {f"m{i}": svd_spectra(random_state(20 + i)) for i in range(3)}
    times = {f"m{i}": f"2026-01-0{i + 1}T00:00" for i in range(3)}
    assert dedup_by_svd(spectra, times) == set(spectra)


def test_missing_timestamp_sorts_last():
    shared = svd_spectra(random_state(15))
    spectra = {"undated": shared, "dated": shared}
    assert dedup_by_svd(spectra, {"dated": "2026-01-01T00:00"}) == {"dated"}
