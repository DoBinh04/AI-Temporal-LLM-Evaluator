"""Spectral comparison — telling a copy from an independent model.

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


# ─── the per-year gate ───────────────────────────────────────────────────


def gate_for(year: int, *states, threshold=0.01) -> SvdGate:
    gate = SvdGate(threshold=threshold)
    for state in states:
        gate.add_baseline(year, svd_spectra(state))
    return gate


def test_a_copy_of_a_baseline_fails_its_year():
    passed, distance = gate_for(2013, random_state(1)).check(svd_spectra(random_state(1)), 2013)
    assert not passed
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_an_independent_model_passes():
    passed, distance = gate_for(2013, random_state(1)).check(svd_spectra(random_state(2)), 2013)
    assert passed
    assert distance > 0.01


def test_the_minimum_distance_over_all_baselines_decides():
    """Being far from one published variant does not excuse copying another."""
    gate = gate_for(2013, random_state(1), random_state(2))
    passed, distance = gate.check(svd_spectra(random_state(2)), 2013)
    assert not passed
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_a_year_with_no_baselines_passes():
    passed, distance = gate_for(2013, random_state(1)).check(svd_spectra(random_state(1)), 2014)
    assert passed
    assert distance == 1.0


def test_an_incomparable_baseline_does_not_fail_the_candidate():
    gate = gate_for(2013, {"other.weight": torch.randn(8, 8)})
    passed, _ = gate.check(svd_spectra(random_state(1)), 2013)
    assert passed


def test_an_empty_gate_is_falsy():
    assert not SvdGate()
    assert gate_for(2013, random_state(1))


# ─── pairwise dedup ──────────────────────────────────────────────────────


def spectra_set(**states) -> dict:
    return {name: svd_spectra(state) for name, state in states.items()}


def test_distinct_models_all_survive():
    spectra = spectra_set(a=random_state(1), b=random_state(2), c=random_state(3))
    assert dedup_by_svd(spectra) == {"a", "b", "c"}


def test_the_first_of_a_duplicate_pair_wins():
    spectra = spectra_set(late=random_state(1), early=random_state(1))
    assert dedup_by_svd(spectra, precedence=["early", "late"]) == {"early"}
    assert dedup_by_svd(spectra, precedence=["late", "early"]) == {"late"}


def test_dict_order_is_the_default_precedence():
    spectra = spectra_set(first=random_state(1), second=random_state(1))
    assert dedup_by_svd(spectra) == {"first"}


def test_a_disguised_copy_is_still_deduped():
    weight = random_state(3)["layer.weight"]
    q, _ = torch.linalg.qr(torch.randn(32, 32, generator=torch.Generator().manual_seed(9)))
    spectra = spectra_set(
        original={"layer.weight": weight},
        disguised={"layer.weight": q @ weight},
    )
    assert dedup_by_svd(spectra) == {"original"}


