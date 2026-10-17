"""Reader provider abstraction (Section 7.3).

One method: generate(images, prompt, temperature, n) -> list[str].
- mock: deterministic canned output (tests / offline smoke).
- gemini (default): official google-genai SDK (v2, verified by inspection).
  run `probe_gemini()` once a GEMINI_API_KEY exists and record the model list.
- groq / openrouter: alternative vision-chat implementations of the same interface.

Every external call must go through CachedReader (DiskCache + RateLimiter + retry).
"""
from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from vvrag.cache import DiskCache
from vvrag.ratelimit import RateLimiter, retry_with_backoff
from vvrag.schemas import ReaderSample

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


class SequenceMockReader(ReaderClient):
    """Returns queued outputs in order (for retry-once tests)."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.temps: list[float] = []

    def generate(self, images, prompt, temperature, n) -> list[str]:
        self.temps.extend([temperature] * n)
        return [self.outputs.pop(0) if self.outputs else CANNED_JSON for _ in range(n)]


class GeminiReader(ReaderClient):
    """Gemini multimodal reader. Requires GEMINI_API_KEY + google-genai (new SDK).

    Verified 2026-10-14 against google-genai 2.30.1 via signature inspection
    (Client(api_key=...), models.generate_content(model, contents, config)).
    """

    def __init__(self, model: str, api_key: str, max_output_tokens: int = 512) -> None:
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError("pip install google-genai for the gemini reader") from e
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, images, prompt, temperature, n) -> list[str]:
        from PIL import Image

        client = self._ensure()
        pil_images = [Image.open(p) for p in images]
        outs: list[str] = []
        for _ in range(n):
            resp = client.models.generate_content(
                model=self.model,
                contents=[*pil_images, prompt],
                config={
                    "temperature": temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "response_mime_type": "application/json",
                },
            )
            outs.append(resp.text)
        return outs


def probe_gemini(api_key: str) -> list[str]:
    """List available Gemini models. Run once with a key; record output in decisions.md."""
    from google import genai

    client = genai.Client(api_key=api_key)
    return [m.name for m in client.models.list()]


class GroqReader(ReaderClient):
    """Groq vision-chat reader. Requires GROQ_API_KEY + groq SDK + vision model."""

    def __init__(self, model: str, api_key: str, max_output_tokens: int = 512) -> None:
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens

    def generate(self, images, prompt, temperature, n) -> list[str]:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        content: list[dict] = [{"type": "text", "text": prompt + "\nReturn JSON only."}]
        for p in images:
            b64 = base64.b64encode(Path(str(p)).read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        outs = []
        for _ in range(n):
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=self.max_output_tokens,
                response_format={"type": "json_object"},
            )
            outs.append(resp.choices[0].message.content or "")
        return outs


class OpenRouterReader(ReaderClient):
    """OpenRouter vision reader (OpenAI-compatible). Requires OPENROUTER_API_KEY."""

    def __init__(self, model: str, api_key: str, max_output_tokens: int = 512) -> None:
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens

    def generate(self, images, prompt, temperature, n) -> list[str]:
        from openai import OpenAI

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        content: list[dict] = [{"type": "text", "text": prompt + "\nReturn JSON only."}]
        for p in images:
            b64 = base64.b64encode(Path(str(p)).read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        outs = []
        for _ in range(n):
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=self.max_output_tokens,
                response_format={"type": "json_object"},
            )
            outs.append(resp.choices[0].message.content or "")
        return outs


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


def run_reader_samples(
    reader: ReaderClient,
    images: Sequence[str | Path],
    prompt: str,
    n_samples: int = 3,
    temperature: float = 0.7,
    max_claims: int = 5,
) -> tuple[list[ReaderSample], int]:
    """Sample 0 at temperature 0 (primary), rest at `temperature`.

    Invalid JSON is retried once per sample, then marked failed
    (answer UNANSWERABLE, raw kept). Returns (samples, n_api_calls).
    """
    from vvrag.reader.claims import parse_reader_output_full

    samples: list[ReaderSample] = []
    api_calls = 0

    def _one(temp: float) -> tuple[ReaderSample, int]:
        raw = reader.generate(images, prompt, temp, 1)[0]
        calls = 1
        answer, quote, sids, claims, ok = parse_reader_output_full(raw, max_claims)
        if not ok:  # retry once
            raw = reader.generate(images, prompt, temp, 1)[0]
            calls += 1
            answer, quote, sids, claims, ok = parse_reader_output_full(raw, max_claims)
        return ReaderSample(sample_id=0, answer=answer, claims=claims, raw=raw,
                            quote=quote, sentence_ids=sids), calls

    for s in range(n_samples):
        temp = 0.0 if s == 0 else temperature
        sample, calls = _one(temp)
        sample.sample_id = s
        api_calls += calls
        samples.append(sample)
    return samples, api_calls


class FallbackReader(ReaderClient):
    """Primary reader with a fallback model for quota/availability errors (#9).

    On a `RateLimitError` (or any exception whose message mentions quota /
    rate / 429 / 503) from the primary, tries the fallback once. If the
    fallback is None or also fails, the original error propagates so the
    caller can abstain cleanly.
    """

    _QUOTA_HINTS = ("429", "quota", "rate limit", "rate_limit", "resource_exhausted",
                    "503", "overloaded", "unavailable", "high demand")

    def __init__(self, primary: ReaderClient, fallback: ReaderClient | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.fallback_used = 0

    @classmethod
    def _is_quota(cls, exc: BaseException) -> bool:
        from vvrag.ratelimit import RateLimitError

        if isinstance(exc, RateLimitError):
            return True
        msg = str(exc).lower()
        return any(h in msg for h in cls._QUOTA_HINTS)

    def generate(self, images, prompt, temperature, n) -> list[str]:
        try:
            return self.primary.generate(images, prompt, temperature, n)
        except Exception as e:  # noqa: BLE001 - only quota-like errors fall back
            if self.fallback is None or not self._is_quota(e):
                raise
            self.fallback_used += 1
            return self.fallback.generate(images, prompt, temperature, n)


def _build_reader(provider: str, model: str, max_output_tokens: int) -> ReaderClient:
    """Construct a raw (uncached) reader for one provider."""
    from vvrag.config import env

    provider = provider.lower()
    if provider in ("mock", "test"):
        return MockReader()
    if provider == "gemini":
        key = env("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing (see .env.example)")
        return GeminiReader(model, key, max_output_tokens)
    if provider == "groq":
        key = env("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing (see .env.example)")
        return GroqReader(model, key, max_output_tokens)
    if provider == "openrouter":
        key = env("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY missing (see .env.example)")
        return OpenRouterReader(model, key, max_output_tokens)
    raise ValueError(f"unknown reader provider: {provider}")


def create_reader(
    provider: str,
    model: str,
    cache: DiskCache,
    limiter: RateLimiter,
    max_retries: int = 6,
    max_output_tokens: int = 512,
    fallback_provider: str = "",
    fallback_model: str = "",
) -> ReaderClient:
    """Build a CachedReader, optionally wrapped with a FallbackReader (#9)."""
    inner: ReaderClient = _build_reader(provider, model, max_output_tokens)
    if fallback_provider or fallback_model:
        try:
            fb = _build_reader(fallback_provider or provider,
                               fallback_model or model, max_output_tokens)
            inner = FallbackReader(inner, fb)
        except Exception:
            pass  # no key/sdk for the fallback: keep primary only
    return CachedReader(inner, cache, limiter, provider, model, max_retries)
