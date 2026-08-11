"""The .env loader: convenience without surprises.

The contract that matters is precedence — a variable set in the real
environment must never be silently replaced by a file.
"""

from __future__ import annotations

import os

from wigin_tllm.env import load_dotenv


def write_env(tmp_path, content: str) -> str:
    path = tmp_path / ".env"
    path.write_text(content)
    return str(path)


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(str(tmp_path / "nope.env")) == {}


def test_values_land_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("WIGIN_TEST_KEY", raising=False)
    path = write_env(tmp_path, "WIGIN_TEST_KEY=sk-abc123\n")
    assert load_dotenv(path) == {"WIGIN_TEST_KEY": "sk-abc123"}
    assert os.environ["WIGIN_TEST_KEY"] == "sk-abc123"
    monkeypatch.delenv("WIGIN_TEST_KEY", raising=False)


def test_the_real_environment_always_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("WIGIN_TEST_KEY", "from-env")
    path = write_env(tmp_path, "WIGIN_TEST_KEY=from-file\n")
    assert load_dotenv(path) == {}
    assert os.environ["WIGIN_TEST_KEY"] == "from-env"


def test_comments_blanks_export_and_quotes(tmp_path, monkeypatch):
    for key in ("WIGIN_A", "WIGIN_B", "WIGIN_C"):
        monkeypatch.delenv(key, raising=False)
    path = write_env(tmp_path, (
        "# a comment\n"
        "\n"
        "export WIGIN_A=plain\n"
        'WIGIN_B="double quoted"\n'
        "WIGIN_C='single quoted'\n"
    ))
    applied = load_dotenv(path)
    assert applied == {"WIGIN_A": "plain", "WIGIN_B": "double quoted",
                       "WIGIN_C": "single quoted"}
    for key in ("WIGIN_A", "WIGIN_B", "WIGIN_C"):
        monkeypatch.delenv(key, raising=False)


def test_malformed_lines_are_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.delenv("WIGIN_OK", raising=False)
    path = write_env(tmp_path, "not a pair\n=novalue\nWIGIN_OK=yes\n")
    assert load_dotenv(path) == {"WIGIN_OK": "yes"}
    monkeypatch.delenv("WIGIN_OK", raising=False)


def test_value_may_contain_equals_signs(tmp_path, monkeypatch):
    monkeypatch.delenv("WIGIN_URL", raising=False)
    path = write_env(tmp_path, "WIGIN_URL=http://host/v1?a=b&c=d\n")
    assert load_dotenv(path) == {"WIGIN_URL": "http://host/v1?a=b&c=d"}
    monkeypatch.delenv("WIGIN_URL", raising=False)
