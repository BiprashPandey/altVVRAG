"""Gold-span matching incl. article-crossing regression (testRun2, 2026-10-15)."""
from vvrag.eval.gold_regions import derive_gold_box
from vvrag.schemas import WordBox


def _words(texts):
    return [WordBox("d", 0, t, (i * 10, 0, i * 10 + 8, 10), 0, 0)
            for i, t in enumerate(texts)]


def test_exact_span():
    words = _words("The quick brown fox jumps".split())
    assert derive_gold_box("quick brown fox", words) == (10, 0, 38, 10)


def test_span_crossing_removed_article():
    # normalize_answer drops "the" -> empty slot; the span must still match.
    words = _words("about the Department".split())
    assert derive_gold_box("about the Department", words) == (0, 0, 28, 10)


def test_no_match_returns_none():
    words = _words("hello world".split())
    assert derive_gold_box("quarterly revenue", words) is None
