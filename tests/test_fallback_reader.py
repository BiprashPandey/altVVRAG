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
    assert not FallbackReader._is_quota(ValueError("bad prompt"))
