"""Model references and submission validation."""

from __future__ import annotations

import pytest

from wigin_tllm.types import ModelRef, SubmissionError, validate_manifest

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
    missing = validate_manifest({"2013": f"user/m@{SHA}"}, YEARS)
    assert missing == [2014, 2015]


def test_reports_no_missing_years_when_complete():
    models = {str(y): f"user/m-{y}@{SHA}" for y in YEARS}
    assert validate_manifest(models, YEARS) == []


@pytest.mark.parametrize("ref", ["user/model@main", "user/model", "user/model@abc123", f"user@{SHA}"])
def test_rejects_unpinned_references(ref):
    with pytest.raises(SubmissionError):
        validate_manifest({"2013": ref}, YEARS)


def test_allows_unpinned_when_configured():
    assert validate_manifest({"2013": "user/model"}, YEARS, require_pinned_revision=False) == [2014, 2015]


def test_rejects_year_outside_range():
    with pytest.raises(SubmissionError, match="not in"):
        validate_manifest({"1999": f"user/m@{SHA}"}, YEARS)


def test_rejects_non_integer_year():
    with pytest.raises(SubmissionError, match="integer"):
        validate_manifest({"twenty-thirteen": f"user/m@{SHA}"}, YEARS)


def test_rejects_non_dict():
    with pytest.raises(SubmissionError, match="dict"):
        validate_manifest([f"user/m@{SHA}"], YEARS)


def test_rejects_empty_year_range():
    with pytest.raises(SubmissionError, match="allowed_years"):
        validate_manifest({}, [])


def test_local_reference_must_exist(tmp_path):
    with pytest.raises(SubmissionError, match="does not exist"):
        validate_manifest({"2013": "local:/definitely/not/here"}, YEARS)
    assert validate_manifest({"2013": f"local:{tmp_path}"}, YEARS) == [2014, 2015]


