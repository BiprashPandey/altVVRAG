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


def split_windows(premise: str, max_chars: int = 500) -> list[str]:
    """Split a long premise into sentence windows (each <= max_chars).

    Kept small on purpose (verified 2026-10-15 on a real OCR page): the
    cross-encoder's verdict on a VERBATIM sentence flips from 0.995 to 0.002
    when its window also contains garbled OCR word-salad elsewhere on the
    page. Max-over-fine-windows isolates evidence sentences; max_chars=1500
    lumped a whole page into one window and buried them.
    """
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
                 device: str = "auto", max_length: int = 512) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self._model = None
        self._entail_idx = None

    @staticmethod
    def resolve_device(want: str) -> str:
        if want != "auto":
            return want
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def load(self) -> None:
        from sentence_transformers import CrossEncoder

        self.device = self.resolve_device(self.device)
        self._model = CrossEncoder(self.model_id, device=self.device,
                                   max_length=self.max_length)
        self._entail_idx = label_order(self._model)["entailment"]
        log.info("NLI loaded: %s on %s (entail_idx=%d)",
                 self.model_id, self.device, self._entail_idx)

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


def qa_to_statement(question: str, answer: str) -> str:
    """Turn (question, answer) into a declarative statement for NLI (#4).

    "What is the title of the document?" + "Sample Document for PDF Testing"
    -> "The title of the document is Sample Document for PDF Testing."
    Falls back to "The answer to '<q>' is '<a>'." when the question can't be
    templated. This makes the entailment check read as claim support rather
    than question-matching.
    """
    q = " ".join((question or "").strip().rstrip("?").split())
    a = " ".join((answer or "").strip().split())
    if not q or not a:
        return a
    low = q.lower()
    for lead in ("what is the ", "what was the ", "what is ", "what was ",
                 "which ", "who is the ", "who was the ", "where is the ",
                 "when is the ", "how many ", "how much "):
        if low.startswith(lead):
            subject = q[len(lead):].strip()
            if subject:
                return f"The {subject} is {a}."
    return f"The answer to \"{q}\" is {a}."


def entail_statement(
    entailer,
    premise_texts: list[str],
    question: str,
    answer: str,
) -> float:
    """Entailment prob of the declarative Q+A statement by the cited sentences."""
    if not premise_texts:
        return 0.5
    statement = qa_to_statement(question, answer)
    if not statement:
        return 0.5
    premise = " ".join(premise_texts)
    return float(entailer.entail_probs(premise, [statement])[0])


def minimal_entailing_span(
    claim: str,
    sentences: list[str],
    entail_fn,
    threshold: float = 0.5,
) -> list[str]:
    """Two-stage narrow (rec 3): shrink a region to its minimal entailing window.

    `sentences` is the region (heatmap/span pick) in order. `entail_fn(window,
    claim) -> float` is any entailment scorer (e.g. NLIEntailer over the joined
    window). Greedily drop leading/trailing sentences while entailment stays
    >= threshold; the last passing window is the minimal supporting span.
    Handles paraphrased answers (q4) where plain text matching fails.
    Returns [] when even the full region fails.
    """
    sents = [s.strip() for s in sentences if s and s.strip()]
    if not sents or not claim.strip():
        return []
    join = lambda ws: " ".join(ws).strip()
    if entail_fn(join(sents), claim) < threshold:
        return []
    window = list(sents)
    improved = True
    while improved and len(window) > 1:
        improved = False
        for drop_first in (True, False):
            if len(window) <= 1:
                break
            cand = window[1:] if drop_first else window[:-1]
            try:
                score = entail_fn(join(cand), claim)
            except Exception:
                continue
            if score >= threshold:
                window = cand
                improved = True
    return window


class FakeNLI:
    """Deterministic overlap heuristic for tests (no model download)."""

    def entail_probs(self, premise: str, hypotheses: list[str]) -> list[float]:
        pt = set(premise.lower().split())
        out = []
        for h in hypotheses:
            ht = set(h.lower().split())
            out.append(len(pt & ht) / max(1, len(ht)))
        return out
