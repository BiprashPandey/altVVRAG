"""NLI entailment on crop text (Section 7.4, contribution C3).

Premise = crop_text (windowed over sentences, max entailment).
Hypothesis = claim text. Softmax over the model's OWN label order read from
its config — never assumed.

VERIFIED 2026-10-14: cross-encoder/nli-deberta-v3-base ->
label2id = {contradiction: 0, entailment: 1, neutral: 2} (from model config).
CrossEncoder.predict returns LOGITS; softmax applied here.
"""
from __future__ import annotations

import re

import numpy as np

from vvrag.logging_utils import get_logger

log = get_logger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def label_order(model) -> dict[str, int]:
    """Read entailment label order from the model config."""
    cfg = model.model.config if hasattr(model, "model") else model.config
    l2i = dict(getattr(cfg, "label2id", {}) or {})
    if not l2i:
        i2l = dict(getattr(cfg, "id2label", {}) or {})
        l2i = {v: int(k) for k, v in i2l.items()}
    if "entailment" not in l2i:
        raise ValueError(f"NLI model has no entailment label: {l2i}")
    return l2i


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(np.asarray(logits, dtype=np.float64) - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def split_windows(premise: str, max_chars: int = 1500) -> list[str]:
    """Split a long premise into sentence windows (each <= max_chars)."""
    sents = [s.strip() for s in _SENT_SPLIT.split(premise) if s.strip()] or [premise]
    windows, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 > max_chars and cur:
            windows.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        windows.append(cur)
    return windows


class NLIEntailer:
    def __init__(self, model_id: str = "cross-encoder/nli-deberta-v3-base",
                 device: str = "cpu", max_length: int = 512) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self._model = None
        self._entail_idx = None

    def load(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_id, device=self.device,
                                   max_length=self.max_length)
        self._entail_idx = label_order(self._model)["entailment"]
        log.info("NLI loaded: %s entail_idx=%d", self.model_id, self._entail_idx)

    def entail_probs(self, premise: str, hypotheses: list[str]) -> list[float]:
        """Max-over-windows entailment probability per hypothesis."""
        if self._model is None:
            raise RuntimeError("call load() first")
        windows = split_windows(premise)
        pairs = [(w, h) for w in windows for h in hypotheses]
        logits = np.asarray(self._model.predict(pairs))
        if logits.ndim == 1:  # binary head -> single logit
            probs = 1.0 / (1.0 + np.exp(-logits))
            mat = probs.reshape(len(windows), len(hypotheses))
        else:
            mat = softmax(logits)[:, self._entail_idx].reshape(len(windows), len(hypotheses))
        return [float(mat[:, j].max()) for j in range(len(hypotheses))]


def aggregate_entailment(probs: list[float]) -> tuple[float, float]:
    """Strict minimum (every claim supported) + mean. Empty -> (0.5, 0.5)."""
    if not probs:
        return 0.5, 0.5
    return float(min(probs)), float(sum(probs) / len(probs))


class FakeNLI:
    """Deterministic overlap heuristic for tests (no model download)."""

    def entail_probs(self, premise: str, hypotheses: list[str]) -> list[float]:
        pt = set(premise.lower().split())
        out = []
        for h in hypotheses:
            ht = set(h.lower().split())
            out.append(len(pt & ht) / max(1, len(ht)))
        return out
