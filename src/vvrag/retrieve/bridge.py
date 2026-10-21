"""Bridge-page resolution for multi-hop questions (Run 9, Phase 4).

The dominant honest-abstention class in Runs 6-8 was the multi-hop question:
"in the year <Company> achieved <QUALIFIER>, what was X?" -- the answer page
contains X but not <QUALIFIER>; the qualifier phrase lives on a different
slide (or not in OCR at all). The reader is shown up to `reader.max_pages`
pages, so if the qualifier page can be found deterministically, adding it to
the context fixes the hop without any model.

This module is pure and offline:

- `question_tokens`: content tokens of the question (stopword-free, lowercase);
- `missing_tokens`: tokens of the question absent from the primary page's OCR;
- `resolve_bridge`: scan every page for pages covering the missing tokens
  (IDF-weighted, computed over the corpus candidate set), and return up to
  `max_extra` pages, best first, never the primary.

Wired in `ui.qa.answer_question` behind `retriever.bridge_resolution`
(default true). All returned pages are OCR pages already in the index; nothing
is fetched or encoded. The reader alone decides whether the bridge page helps
(the prompt labels pages p0/p1/p2 and citations name the page).
"""
from __future__ import annotations

import math
import re

from vvrag.schemas import WordBox

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9%.,'-]*")
_STOP = {
    "what", "was", "were", "is", "are", "the", "a", "an", "in", "of", "for",
    "to", "and", "or", "did", "does", "do", "how", "many", "much", "which",
    "who", "when", "where", "that", "this", "it", "its", "on", "at", "by",
    "from", "with", "as", "per", "year", "years", "company", "companies",
}


def _tok(text: str) -> str:
    return re.sub(r"[^a-z0-9%.,'-]", "", str(text).lower()).strip(".,'")


def question_tokens(question: str) -> list[str]:
    toks = []
    for m in _WORD.findall(question):
        t = _tok(m)
        if not t or t in _STOP:
            continue
        if len(t) == 1 and not t.isdigit():
            continue
        toks.append(t)
    return toks


def _page_tokens(words: list[WordBox]) -> set[str]:
    return {_tok(w.text) for w in words if _tok(w.text)}


def missing_tokens(question: str, primary_words: list[WordBox]) -> list[str]:
    have = _page_tokens(primary_words)
    return [t for t in question_tokens(question) if t not in have]


def resolve_bridge(question: str,
                   primary_key: tuple[str, int],
                   ctx_keys: list[tuple[str, int]],
                   words_map: dict[tuple[str, int], list[WordBox]],
                   max_extra: int = 1,
                   min_missing: int = 2) -> list[tuple[str, int, float]]:
    """Pages covering the primary page's missing question tokens.

    Returns [(doc, page, score)] best-first; empty when the question has fewer
    than `min_missing` tokens missing from the primary page (single-hop), or
    when no other page covers any of them.
    """
    if max_extra <= 0:
        return []
    primary_words = words_map.get(primary_key) or []
    missing = missing_tokens(question, primary_words)
    if len(missing) < min_missing:
        return []
    df: dict[str, int] = {}
    page_sets = {k: _page_tokens(w) for k, w in words_map.items()}
    for toks in page_sets.values():
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(page_sets))

    def idf(t: str) -> float:
        return math.log(1.0 + n / (1.0 + df.get(t, 0)))

    total_w = sum(idf(t) for t in missing) or 1.0
    excluded = {primary_key, *ctx_keys}
    scored = []
    for key, toks in page_sets.items():
        if key in excluded:
            continue
        hit_w = sum(idf(t) for t in missing if t in toks)
        if hit_w <= 0:
            continue
        score = hit_w / total_w
        scored.append((score, key))
    scored.sort(key=lambda t: (-t[0], str(t[1][0]), t[1][1]))
    return [(d, p, round(s, 4)) for s, (d, p) in scored[:max_extra]]
