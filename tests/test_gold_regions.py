"""Gold-span matching incl. article-crossing regression (testRun2, 2026-10-15)."""
from vvrag.eval.gold_regions import derive_answer_words, derive_gold_box
from vvrag.ground.boxes import answer_to_line_boxes
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


def test_fuzzy_window_handles_ocr_typo():
    # rec 1: predicted "Rastra" still matches OCR "Rasta"
    words = [WordBox("d", 0, t, (i * 10, 0, i * 10 + 8, 10), 0, 0)
             for i, t in enumerate("acknowledge Nepal Rasta Bank as source".split())]
    span, score = derive_answer_words("Nepal Rastra Bank", words)
    assert [w.text for w in span] == ["Nepal", "Rasta", "Bank"]
    assert score >= 0.8
    # exact path scores 1.0
    span2, score2 = derive_answer_words("Nepal Rasta Bank", words)
    assert score2 == 1.0 and len(span2) == 3


def test_multiline_answer_splits_per_line():
    words = [WordBox("d", 0, "public", (0, 0, 40, 10), 0, 0),
             WordBox("d", 0, "access", (45, 0, 90, 10), 0, 0),
             WordBox("d", 0, "activities", (0, 20, 60, 30), 0, 1)]
    span, score = derive_answer_words("public access activities", words)
    assert score == 1.0
    boxes = answer_to_line_boxes(span)
    assert len(boxes) == 2  # rec 4: one tight box per line, not one big union
    assert boxes[0][3] <= 10 and boxes[1][1] >= 20


def test_miss_returns_empty_with_zero_score():
    words = _words("hello world".split())
    span, score = derive_answer_words("quarterly revenue", words)
    assert span == [] and score == 0.0
