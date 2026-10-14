"""Rate limiter + retry under forced 429 (Phase 1 / Section 7.3 requirement)."""
from vvrag.cache import DiskCache
from vvrag.ratelimit import RateLimiter, retry_with_backoff
from vvrag.reader.client import CachedReader, FlakyMockReader


def test_retry_survives_forced_429(tmp_path):
    cache = DiskCache(tmp_path / "c")
    limiter = RateLimiter(requests_per_minute=600)
    reader = CachedReader(FlakyMockReader(fail_times=2), cache, limiter, "mock", "m", max_retries=5)
    outs = reader.generate([], "prompt", 0.0, 1)
    assert len(outs) == 1
    assert "mock answer" in outs[0]


def test_cached_reader_second_call_no_inner_call(tmp_path):
    cache = DiskCache(tmp_path / "c")
    limiter = RateLimiter(requests_per_minute=600)
    inner_calls = {"n": 0}
    from vvrag.reader.client import MockReader

    inner = MockReader(calls=inner_calls)
    reader = CachedReader(inner, cache, limiter, "mock", "m", max_retries=2)
    reader.generate([], "same prompt", 0.0, 1)
    reader.generate([], "same prompt", 0.0, 1)
    assert inner_calls["n"] == 1


def test_retry_with_backoff_zero_delay():
    attempts = {"n": 0}
    from vvrag.ratelimit import RateLimitError

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("429")
        return "ok"

    assert retry_with_backoff(fn, max_retries=5, base_delay=0.0) == "ok"
    assert attempts["n"] == 3
