"""Sentence segmentation tests (suggestion #1: cite sentences, not words)."""
from vvrag.ingest.sentences import box_frac, build_sentences
from vvrag.schemas import WordBox


def _line(words, y=0, start=0, h=20, block=0, line=0, w=50, gap=5):
    out = []
    x = start
    for t in words:
        out.append(WordBox("d", 0, t, (x, y, x + w, y + h), block, line))
        x += w + gap
    return out


def test_split_on_terminal_punctuation():
    words = _line(["Hello", "world.", "Second", "sentence."])
    sents, word_to = build_sentences(words)
    assert [s.text for s in sents] == ["Hello world.", "Second sentence."]
    assert len(word_to) == 4 and word_to[0] == 0 and word_to[3] == 1


def test_multi_line_sentence_merges_box():
    l1 = _line(["The", "quick"], y=0, line=0)
    l2 = _line(["fox", "jumps."], y=30, line=1)
    sents, _ = build_sentences(l1 + l2)
    assert len(sents) == 1
    assert sents[0].text == "The quick fox jumps."
    # union box spans both lines
    assert sents[0].box[1] == 0 and sents[0].box[3] == 50


def test_new_block_starts_new_sentence():
    b0 = _line(["Header", "words"], y=0, block=0, line=0)
    b1 = _line(["Body", "text.", "More."], y=60, block=1, line=0)
    sents, _ = build_sentences(b0 + b1)
    assert [s.text for s in sents] == ["Header words", "Body text.", "More."]


def test_list_marker_starts_new_unit():
    words = _line(["1.", "First", "item", "2.", "Second", "item"])
    sents, _ = build_sentences(words)
    assert sents[0].text == "1. First item"
    assert sents[1].text == "2. Second item"


def test_table_cell_gap_splits():
    # same line, a big x-gap before the next cell (median height 20 -> gap 50)
    a = [WordBox("d", 0, "Row", (0, 0, 40, 20), 0, 0)]
    b = [WordBox("d", 0, "1", (200, 0, 210, 20), 0, 0)]
    c = [WordBox("d", 0, "Col", (240, 0, 280, 20), 0, 0)]
    sents, _ = build_sentences(a + b + c)
    assert len(sents) >= 2  # cell boundary split


def test_vertical_gap_splits_heading():
    head = _line(["Title"], y=0, block=0, line=0)
    para = _line(["Body", "starts", "here."], y=120, block=0, line=1)
    sents, _ = build_sentences(head + para)
    assert [s.text for s in sents] == ["Title", "Body starts here."]


def test_empty_page():
    assert build_sentences([]) == ([], [])


def test_box_frac():
    frac = box_frac((100, 50, 300, 250), 1000, 500)
    assert frac == (0.1, 0.1, 0.3, 0.5)
    assert box_frac((-10, -10, 2000, 2000), 100, 100) == (0.0, 0.0, 1.0, 1.0)
