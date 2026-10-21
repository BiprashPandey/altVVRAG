"""Deterministic lexical reranker (Run 9, Phase 4).

Why: keyword overlap counts matching tokens flat, so a table page with the
question's numbers buried under 60 other tokens beats the sentence page that
actually answers it. This reranker is a pure function (no model, no GPU, no
network):

    score = 0.6 * unigram IDF coverage + 0.3 * bigram phrase hits
            + 0.1 * number/date token bonus

IDF is computed over the candidate page set itself (page frequency), so rare
tokens (years, product names, 'Eismann') dominate common ones ('sales').
Wired into `ui.qa.answer_question` when `retriever.rerank` is true; never
changes scores when `ranked` has one page.
"""
from __future__ import annotations

import math
import re

from vvrag.schemas import WordBox

_NUM = re.compile(r"^\d+(?:[.,]\d+)*%?$")


def _tok(text: str) -> str:
    return re.sub(r"[^a-z0-9%.,']", "", str(text).lower()).strip(".,'")


def page_tokens(words: list[WordBox]) -> list[str]:
    return [_tok(w.text) for w in words if _tok(w.text)]


def lexical_rerank(question: str,
                   ranked: list[tuple[str, int, float, list[WordBox]]],
                   k: int = 5) -> list[tuple[str, int, float, list[WordBox]]]:
    """Reorder `ranked` [(doc, page, score, words)] best-first by IDF match.

    Deterministic; ties fall back to the incoming order (the original index is
    the sort key's second element). The returned `score` is the rank score
    1/(position+1): downstream margin features stay positive and comparable
    (the incoming scores come from different scales per retrieval mode).
    """
    if len(ranked) <= 1:
        return ranked
    q = [_tok(t) for t in re.findall(r"[A-Za-z0-9%.']+", question)]
    q = [t for t in q if t]
    if not q:
        return ranked
    pages = [page_tokens(words) for _d, _p, _s, words in ranked]
    df: dict[str, int] = {}
    for toks in pages:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n_pages = len(ranked)

    def idf(t: str) -> float:
        return math.log(1.0 + n_pages / (1.0 + df.get(t, 0)))

    q_u = q
    q_bigrams = {f"{a} {b}" for a, b in zip(q, q[1:])}
    scored = []
    for i, ((d, p, base, words), toks) in enumerate(zip(ranked, pages)):
        tset = set(toks)
        uni = sum(idf(t) for t in set(q_u) if t in tset)
        uni_norm = uni / max(1e-9, sum(idf(t) for t in set(q_u)))
        big = 0.0
        if q_bigrams and len(toks) > 1:
            tbigrams = {f"{a} {b}" for a, b in zip(toks, toks[1:])}
            big = len(q_bigrams & tbigrams) / len(q_bigrams)
        num_q = {t for t in set(q_u) if _NUM.match(t)}
        num_hit = len(num_q & tset) / len(num_q) if num_q else 0.0
        score = 0.6 * uni_norm + 0.3 * big + 0.1 * num_hit
        scored.append((-score, i, (d, p, base, words)))
    scored.sort(key=lambda t: (t[0], t[1]))
    rows = [row for _neg, _i, row in scored]
    if k:
        rows = rows[:k]
    return [(d, p, 1.0 / (i + 1), w) for i, (d, p, _s, w) in enumerate(rows)]
