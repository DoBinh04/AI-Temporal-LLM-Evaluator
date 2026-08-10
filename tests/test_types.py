"""Model references and submission validation."""

from __future__ import annotations

import pytest

from wigin_tllm.types import ModelRef, Submission, SubmissionError, validate_submission

YEARS = [2013, 2014, 2015]
SHA = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"


# ─── reference parsing ───────────────────────────────────────────────────


def test_parses_bare_hf_reference():
    ref = ModelRef.parse(f"owner/model@{SHA}")
    assert (ref.scheme, ref.location, ref.revision) == ("hf", "owner/model", SHA)
    assert ref.is_pinned


def test_parses_explicit_hf_scheme():
    assert ModelRef.parse(f"hf:owner/model@{SHA}").location == "owner/model"


def test_parses_local_reference():
    ref = ModelRef.parse("local:/tmp/model")
    assert (ref.scheme, ref.location, ref.revision) == ("local", "/tmp/model", None)
    assert not ref.is_pinned


def test_unpinned_hf_reference_is_not_pinned():
    assert not ModelRef.parse("owner/model").is_pinned
    assert not ModelRef.parse("owner/model@main").is_pinned


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
def test_rejects_empty_reference(bad):
    with pytest.raises(SubmissionError):
        ModelRef.parse(bad)


def test_roundtrips_through_str():
    for raw in [f"owner/model@{SHA}", "local:/tmp/m", "hf:owner/model"]:
        assert str(ModelRef.parse(raw)) == raw


# ─── submission validation ───────────────────────────────────────────────


def test_accepts_pinned_submission():
    missing = validate_submission({"2013": f"user/m@{SHA}"}, YEARS)
    assert missing == [2014, 2015]


def test_reports_no_missing_years_when_complete():
    models = {str(y): f"user/m-{y}@{SHA}" for y in YEARS}
    assert validate_submission(models, YEARS) == []


@pytest.mark.parametrize("ref", ["user/model@main", "user/model", "user/model@abc123", f"user@{SHA}"])
def test_rejects_unpinned_references(ref):
    with pytest.raises(SubmissionError):
        validate_submission({"2013": ref}, YEARS)


def test_allows_unpinned_when_configured():
    assert validate_submission({"2013": "user/model"}, YEARS, require_pinned_revision=False) == [2014, 2015]


def test_rejects_year_outside_range():
    with pytest.raises(SubmissionError, match="not in"):
        validate_submission({"1999": f"user/m@{SHA}"}, YEARS)


def test_rejects_non_integer_year():
    with pytest.raises(SubmissionError, match="integer"):
        validate_submission({"twenty-thirteen": f"user/m@{SHA}"}, YEARS)


def test_rejects_non_dict():
    with pytest.raises(SubmissionError, match="dict"):
        validate_submission([f"user/m@{SHA}"], YEARS)


def test_rejects_empty_year_range():
    with pytest.raises(SubmissionError, match="allowed_years"):
        validate_submission({}, [])


def test_local_reference_must_exist(tmp_path):
    with pytest.raises(SubmissionError, match="does not exist"):
        validate_submission({"2013": "local:/definitely/not/here"}, YEARS)
    assert validate_submission({"2013": f"local:{tmp_path}"}, YEARS) == [2014, 2015]


# ─── submission helpers ──────────────────────────────────────────────────


def test_submission_lookup_by_year():
    sub = Submission("alice", {"2013": f"user/m@{SHA}"}, "2026-01-01T00:00:00")
    assert sub.ref_for_year(2013).location == "user/m"
    assert sub.ref_for_year("2013").location == "user/m"
    assert sub.ref_for_year(2014) is None


def test_submissions_sort_by_time_then_id():
    early = Submission("zeta", {}, "2026-01-01T00:00:00")
    late = Submission("alpha", {}, "2026-01-02T00:00:00")
    undated = Submission("beta", {})
    assert [s.submitter_id for s in sorted([undated, late, early], key=lambda s: s.sort_key)] == [
        "zeta",
        "alpha",
        "beta",  # undated sorts last, forfeiting anti-copy priority
    ]


def test_revision_is_split_at_the_last_separator():
    """A stray '@' belongs to the location, not the revision."""
    ref = ModelRef.parse(f"owner/re@po@{SHA}")
    assert ref.location == "owner/re@po"
    assert ref.revision == SHA
    assert not ref.is_pinned  # '@' is not valid in an owner or repo name


def test_reference_without_a_revision_has_none():
    ref = ModelRef.parse("owner/repo")
    assert ref.revision is None
    assert not ref.is_pinned
