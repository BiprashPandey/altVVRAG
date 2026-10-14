"""Token-bucket rate limiter + retry with exponential backoff (Section 13)."""
from __future__ import annotations

import random
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class RateLimitError(RuntimeError):
    """Raised on 429 / quota exhaustion (also used to simulate forced 429s in tests)."""


class TokenBucket:
    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.capacity = float(requests_per_minute)
        self.tokens = float(requests_per_minute)
        self.refill_per_sec = float(requests_per_minute) / 60.0
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.refill_per_sec)
                self._last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError("rate limiter acquire timed out")
            time.sleep(0.05)


class RateLimiter:
    def __init__(self, requests_per_minute: float = 10) -> None:
        self.bucket = TokenBucket(requests_per_minute)

    def acquire(self, timeout: float = 60.0) -> None:
        self.bucket.acquire(timeout=timeout)


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 6,
    base_delay: float = 1.0,
    jitter: float = 0.25,
    retry_on: tuple[type[BaseException], ...] = (RateLimitError,),
) -> T:
    """Call fn(); retry on retry_on with exponential backoff + jitter."""
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on:
            if attempt >= max_retries:
                raise
            delay = base_delay * (2**attempt)
            delay = delay * (1.0 + random.uniform(-jitter, jitter))
            if delay > 0:
                time.sleep(delay)
            attempt += 1
