"""Run-harness guards: index argument, resume filter, Gemini HTTP timeout."""
import sys
import types
from pathlib import Path

import pytest

from vvrag.eval.harness import select_resume_rows, validate_index_arg


def test_index_guard_accepts_directory_and_rejects_sqlite_file():
    assert validate_index_arg("data/index") == Path("data/index")
    assert validate_index_arg(Path("data/index")) == Path("data/index")
    with pytest.raises(SystemExit) as exc:
        validate_index_arg("data/index/meta.db")
    assert "--index expects the index DIRECTORY" in str(exc.value)
    with pytest.raises(SystemExit):
        validate_index_arg(Path("META.DB"))


def test_resume_filter_selects_errored_rows_for_requested_models():
    rows = [
        {"model": "model-a", "error": "boom"},
        {"model": "model-b", "error": "boom"},
        {"model": "model-a"},  # completed
        {"model": "model-a", "error": "PENDING: not run yet"},
        {"model": "model-b"},  # completed
    ]
    assert select_resume_rows(rows, ["model-a"]) == [0, 3]
    assert select_resume_rows(rows, ["model-b"]) == [1]
    assert select_resume_rows(rows, ["missing"]) == []
    assert select_resume_rows(rows, None) == [0, 1, 3]


def test_gemini_reader_http_timeout(monkeypatch):
    from vvrag.reader.client import GeminiReader

    class FakeHttpOptions:
        def __init__(self, timeout=None):
            self.timeout = timeout

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            self.api_key = api_key
            self.http_options = http_options

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.Client = FakeClient
    genai_types = types.ModuleType("google.genai.types")
    genai_types.HttpOptions = FakeHttpOptions
    genai.types = genai_types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)

    client = GeminiReader("gemini-test", "key", timeout_s=123.0)._ensure()
    assert client.api_key == "key"
    assert client.http_options.timeout == 123_000
