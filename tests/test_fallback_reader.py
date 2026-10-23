"""Fallback reader tests (suggestion #9: survive quota errors)."""
import pytest

from vvrag.ratelimit import RateLimitError
from vvrag.reader.client import (
    CANNED_JSON,
    FallbackReader,
    MockReader,
    SequenceMockReader,
)


class _QuotaReader(MockReader):
    def generate(self, images, prompt, temperature, n):
        raise RateLimitError("429 quota exceeded")


class _BoomReader(MockReader):
    def generate(self, images, prompt, temperature, n):
        raise ValueError("hard failure, cannot proceed")


def test_fallback_used_on_quota_error():
    fb = FallbackReader(_QuotaReader(), MockReader())
    out = fb.generate([], "p", 0.0, 1)
    assert out == [CANNED_JSON] and fb.fallback_used == 1


def test_non_quota_error_does_not_fall_back():
    fb = FallbackReader(_BoomReader(), MockReader())
    with pytest.raises(ValueError):
        fb.generate([], "p", 0.0, 1)
    assert fb.fallback_used == 0


def test_no_fallback_reraises():
    fb = FallbackReader(_QuotaReader(), None)
    with pytest.raises(RateLimitError):
        fb.generate([], "p", 0.0, 1)


def test_primary_success_skips_fallback():
    fb = FallbackReader(MockReader(), _QuotaReader())
    out = fb.generate([], "p", 0.0, 1)
    assert out == [CANNED_JSON] and fb.fallback_used == 0


def test_quota_message_detection():
    assert FallbackReader._is_quota(RateLimitError("x"))
    assert FallbackReader._is_quota(RuntimeError("HTTP 503 unavailable"))
    assert FallbackReader._is_quota(RuntimeError("RESOURCE_EXHAUSTED"))
    assert FallbackReader._is_quota(RuntimeError(
        "403 PERMISSION_DENIED. Your project has been denied access."))
    assert FallbackReader._is_quota(RuntimeError("429 too many requests"))
    assert not FallbackReader._is_quota(ValueError("bad prompt"))


class _ScriptedReader:
    """Records which key it was built with; fails on any scripted key."""

    fail_on: set = set()
    built: list = []

    def __init__(self, model, api_key, max_output_tokens=512):
        self.api_key = api_key
        _ScriptedReader.built.append(api_key)

    def generate(self, images, prompt, temperature, n):
        if self.api_key in _ScriptedReader.fail_on:
            raise RuntimeError("429 RESOURCE_EXHAUSTED (scripted)")
        return [f"ok:{self.api_key}"]


def test_rotating_key_reader_skips_exhausted_key():
    from vvrag.reader.client import RotatingKeyReader

    _ScriptedReader.fail_on = {"k2"}
    _ScriptedReader.built = []
    rk = RotatingKeyReader("m", ["k1", "k2", "k3"], reader_cls=_ScriptedReader)
    out = rk.generate([], "p", 0.0, 1)
    assert out == ["ok:k1"]
    assert rk.key_rotations == 0
    _ScriptedReader.fail_on = {"k1"}
    out = rk.generate([], "p", 0.0, 1)
    assert out == ["ok:k2"]
    assert rk.key_rotations == 1


def test_rotating_key_reader_rotates_on_permission_denied():
    from vvrag.reader.client import RotatingKeyReader

    class _Denied(_ScriptedReader):
        def generate(self, images, prompt, temperature, n):
            if self.api_key == "kd":
                raise RuntimeError(
                    "403 PERMISSION_DENIED. Your project has been denied access.")
            return [f"ok:{self.api_key}"]

    _ScriptedReader.built = []
    rk = RotatingKeyReader("m", ["kd", "kg"], reader_cls=_Denied)
    assert rk.generate([], "p", 0.0, 1) == ["ok:kg"]
    assert rk.key_rotations == 1


def test_provider_key_discovery_includes_numbered_groq_keys():
    from vvrag.reader.client import _groq_keys

    env = {"GROQ_API_KEY": "gsk_a", "GROQ_API_KEY2": "grok_b",
           "GROQ_API_KEY_3": "gsk_a", "OTHER": "x"}
    assert _groq_keys(env.get) == ["gsk_a"]
