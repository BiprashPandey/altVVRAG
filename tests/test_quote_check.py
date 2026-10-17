"""Deterministic quote-check tests (suggestion #4). No model involved."""
from vvrag.schemas import SentenceSpan
from vvrag.verify.quote_check import (
    build_quote_retry_prompt,
    citation_texts,
    quote_match_score,
    quote_matches,
)


def _sents():
    return [SentenceSpan("d", 0, 0, "The Aurora revenue was 42 million in Q3.", (0, 0, 10, 10)),
            SentenceSpan("d", 0, 1, "Beacon load reached 17 percent.", (0, 20, 10, 30))]


def test_exact_substring_scores_one():
    texts = [s.text for s in _sents()]
    assert quote_match_score("Aurora revenue was 42 million", texts) == 1.0
    assert quote_match_score("aurora REVENUE was 42 million", texts) == 1.0  # normalized


def test_fuzzy_match_and_threshold():
    short = ["Aurora revenue was 42 million"]
    # OCR-style typo still passes at >= 0.8 (single-sentence source)
    assert quote_match_score("Aurora ravenue was 42 million", short) >= 0.8
    assert quote_matches("Aurora ravenue was 42 million", short)
    # unrelated quote fails
    assert not quote_matches("completely unrelated words here", short)
    assert quote_match_score("", short) == 0.0


def test_citation_texts_validates_range():
    sents = _sents()
    texts, ok = citation_texts([0, 1], sents)
    assert ok and len(texts) == 2
    texts, ok = citation_texts([0, 999], sents)
    assert not ok  # made-up id
    texts, ok = citation_texts([], sents)
    assert not ok and texts == []


def test_retry_prompt_mentions_failure():
    p = build_quote_retry_prompt("base prompt", "bad quote")
    assert "base prompt" in p and "bad quote" in p and "UNANSWERABLE" in p
