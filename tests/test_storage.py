"""The SQLite store: resume cache, outbox, and the weight-hash registry."""

from __future__ import annotations

import pytest

from wigin_tllm.storage import EvaluationStore
from wigin_tllm.types import YearEvaluation


@pytest.fixture
def store(tmp_path):
    with EvaluationStore(str(tmp_path / "eval.db")) as s:
        yield s


def evaluation(submitter_id="alice", year=2013, ref="local:/m", passed=True, score=-5.0):
    return YearEvaluation(
        submitter_id=submitter_id, year=year, model_ref=ref, passed=passed, score=score,
        score_unknown=-8.0, score_known=-3.0,
    )


# ─── evaluation cache ────────────────────────────────────────────────────


def test_unknown_evaluation_is_a_miss(store):
    assert store.get_cached_evaluation("alice", 2013, "local:/m", 1) is None


def test_saved_evaluation_round_trips(store):
    store.save_evaluation(1, evaluation())
    hit = store.get_cached_evaluation("alice", 2013, "local:/m", 1)
    assert hit.passed is True
    assert hit.score == -5.0
    assert hit.score_unknown == -8.0


def test_cache_is_scoped_to_the_round(store):
    store.save_evaluation(1, evaluation())
    assert store.get_cached_evaluation("alice", 2013, "local:/m", 2) is None


def test_cache_is_scoped_to_the_model_reference(store):
    """Re-submitting different weights for the same year must be re-scored."""
    store.save_evaluation(1, evaluation())
    assert store.get_cached_evaluation("alice", 2013, "local:/other", 1) is None


def test_resaving_replaces_the_row(store):
    store.save_evaluation(1, evaluation(score=-5.0))
    store.save_evaluation(1, evaluation(score=-6.0))
    assert store.get_cached_evaluation("alice", 2013, "local:/m", 1).score == -6.0


# ─── outbox ──────────────────────────────────────────────────────────────


def test_new_evaluations_start_unsynced(store):
    store.save_evaluation(1, evaluation())
    assert len(store.unsynced_evaluations()) == 1


def test_marking_synced_clears_the_outbox(store):
    store.save_evaluation(1, evaluation())
    store.mark_synced(1, "alice", 2013)
    assert store.unsynced_evaluations() == []


def test_saving_as_synced_skips_the_outbox(store):
    store.save_evaluation(1, evaluation(), synced=True)
    assert store.unsynced_evaluations() == []


def test_outbox_spans_rounds(store):
    store.save_evaluation(1, evaluation(year=2013))
    store.save_evaluation(2, evaluation(year=2014))
    assert {r for r, _ in store.unsynced_evaluations()} == {1, 2}


# ─── round completion ────────────────────────────────────────────────────


def test_round_starts_incomplete(store):
    assert not store.is_round_complete(1)


def test_marking_complete_is_idempotent(store):
    store.mark_round_complete(1)
    store.mark_round_complete(1)
    assert store.is_round_complete(1)


def test_round_results_round_trip(store):
    store.save_round_results(1, {"round_id": 1, "submitters": [{"submitter_id": "alice"}]})
    assert store.get_round_results(1)["submitters"][0]["submitter_id"] == "alice"
    assert store.get_round_results(2) is None


def test_clear_round_resets_everything(store):
    store.save_evaluation(1, evaluation())
    store.mark_round_complete(1)
    store.save_round_results(1, {"round_id": 1, "submitters": []})
    store.claim_weight_hash("h", "alice", "t", 1)

    store.clear_round(1)

    assert not store.is_round_complete(1)
    assert store.get_round_results(1) is None
    assert store.get_cached_evaluation("alice", 2013, "local:/m", 1) is None
    # The hash is free again, so a re-run can re-claim it.
    assert store.claim_weight_hash("h", "bob", "t", 1) == (True, "bob")


def test_clear_round_leaves_other_rounds_alone(store):
    store.save_evaluation(1, evaluation())
    store.save_evaluation(2, evaluation())
    store.clear_round(1)
    assert store.get_cached_evaluation("alice", 2013, "local:/m", 2) is not None


# ─── weight-hash registry ────────────────────────────────────────────────


def test_first_claim_is_granted(store):
    assert store.claim_weight_hash("hash-1", "alice", "2026-01-01", 1) == (True, "alice")


def test_second_claimant_is_rejected(store):
    store.claim_weight_hash("hash-1", "alice", "2026-01-01", 1)
    allowed, owner = store.claim_weight_hash("hash-1", "bob", "2026-01-02", 1)
    assert allowed is False
    assert owner == "alice"


def test_reclaiming_your_own_weights_is_allowed(store):
    """Keeps re-runs idempotent."""
    store.claim_weight_hash("hash-1", "alice", "2026-01-01", 1)
    assert store.claim_weight_hash("hash-1", "alice", "2026-01-01", 1) == (True, "alice")


def test_claims_persist_across_rounds(store):
    store.claim_weight_hash("hash-1", "alice", "2026-01-01", 1)
    allowed, owner = store.claim_weight_hash("hash-1", "bob", "2026-02-01", 2)
    assert allowed is False and owner == "alice"


def test_different_weights_are_independent(store):
    store.claim_weight_hash("hash-1", "alice", "2026-01-01", 1)
    assert store.claim_weight_hash("hash-2", "bob", "2026-01-02", 1) == (True, "bob")


# ─── SVD spectra ─────────────────────────────────────────────────────────


def test_spectra_round_trip(store):
    torch = pytest.importorskip("torch")
    spectra = {"w": torch.tensor([3.0, 2.0, 1.0])}
    store.save_spectra("alice", 1, spectra)
    loaded = store.load_spectra(1)
    assert torch.allclose(loaded["alice"]["w"], spectra["w"])


def test_spectra_are_scoped_to_the_round(store):
    torch = pytest.importorskip("torch")
    store.save_spectra("alice", 1, {"w": torch.tensor([1.0, 2.0])})
    assert store.load_spectra(2) == {}


def test_one_spectrum_per_submitter_even_with_several_models(store):
    """A submitter submitting different models per year is represented once,
    by the last spectrum saved."""
    torch = pytest.importorskip("torch")
    store.save_spectra("alice", 1, {"w": torch.tensor([1.0, 2.0])})
    store.save_spectra("alice", 1, {"w": torch.tensor([9.0, 8.0])})
    loaded = store.load_spectra(1)
    assert list(loaded) == ["alice"]
    assert torch.allclose(loaded["alice"]["w"], torch.tensor([9.0, 8.0]))
