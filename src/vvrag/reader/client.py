"""Reader provider abstraction (Section 7.3). Phase 1: mock provider only."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from vvrag.cache import DiskCache
from vvrag.ratelimit import RateLimiter, retry_with_backoff

CANNED_JSON = json.dumps(
    {
        "answer": "mock answer",
        "claims": [{"text": "The mock page supports the mock answer.", "answer_span": "mock answer"}],
    }
)


class ReaderClient(ABC):
    @abstractmethod
    def generate(
        self, images: Sequence[str | Path], prompt: str, temperature: float, n: int
    ) -> list[str]:
        ...


class MockReader(ReaderClient):
    """Deterministic canned reader for tests / offline smoke runs. No network."""

    def __init__(self, canned: str = CANNED_JSON, calls: dict | None = None) -> None:
        self.canned = canned
        self.calls = calls if calls is not None else {"n": 0}

    def generate(self, images, prompt, temperature, n) -> list[str]:
        self.calls["n"] += 1
        return [self.canned for _ in range(n)]


class FlakyMockReader(ReaderClient):
    """Fails `fail_times` with RateLimitError, then returns canned output. For 429 tests."""

    def __init__(self, fail_times: int = 1, canned: str = CANNED_JSON) -> None:
        from vvrag.ratelimit import RateLimitError

        self.fail_times = fail_times
        self.canned = canned
        self.attempts = 0
        self._RateLimitError = RateLimitError

    def generate(self, images, prompt, temperature, n) -> list[str]:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise self._RateLimitError("429 simulated")
        return [self.canned for _ in range(n)]


class CachedReader(ReaderClient):
    """Wraps any ReaderClient with DiskCache + RateLimiter + retry."""

    def __init__(
        self,
        inner: ReaderClient,
        cache: DiskCache,
        limiter: RateLimiter,
        provider: str = "mock",
        model: str = "mock",
        max_retries: int = 6,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.limiter = limiter
        self.provider = provider
        self.model = model
        self.max_retries = max_retries

    def generate(self, images, prompt, temperature, n) -> list[str]:
        image_hashes: list[str] = []
        for im in images:
            p = Path(str(im))
            if p.exists():
                import hashlib

                image_hashes.append(hashlib.sha256(p.read_bytes()).hexdigest()[:16])
            else:
                image_hashes.append(f"missing:{im}")
        outs: list[str] = []
        for i in range(n):
            key = {
                "provider": self.provider,
                "model": self.model,
                "prompt": prompt,
                "images": image_hashes,
                "temperature": temperature,
                "sample_index": i,
            }
            cached = self.cache.get(key)
            if cached is not None:
                outs.append(cached)
                continue
            self.limiter.acquire()
            text = retry_with_backoff(
                lambda: self.inner.generate(images, prompt, temperature, 1)[0],
                max_retries=self.max_retries,
                base_delay=0.0,
            )
            self.cache.set(key, text)
            outs.append(text)
        return outs


def create_reader(
    provider: str,
    model: str,
    cache: DiskCache,
    limiter: RateLimiter,
    max_retries: int = 6,
) -> ReaderClient:
    provider = provider.lower()
    if provider in ("mock", "test"):
        return CachedReader(MockReader(), cache, limiter, provider, model, max_retries)
    raise NotImplementedError(
        f"Reader provider '{provider}' is not implemented until Phase 5 "
        "(gemini/groq/openrouter). Use provider='mock' for now."
    )
