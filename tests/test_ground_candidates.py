"""Deterministic document-grounded candidates for claims."""
from vvrag.ground.candidates import (candidates_for_claim, from_reader_region,
                                     from_sentence, from_words,
                                     rank_candidates)
from vvrag.schemas import SentenceSpan, WordBox


def _sent(text, box, sid=0, source="textlayer"):
    return SentenceSpan(doc_id="d", page=0, sent_id=sid, text=text,
                        box=box, source=source)


def test_table_cell_beats_wide_row_for_number_claim():
    row = _sent("Trading Operating Profit 12.5 7.5 3.9", (10, 10, 500, 40), 1,
                source="table")
    cell = _sent("12.5 bn", (100, 60, 160, 85), 2, source="table")
    other = _sent("Revenue declined in Europe", (10, 100, 400, 130), 3)
    ranked = rank_candidates("12.5 bn", [
        from_sentence(row, "d", 0, 600, 800),
        from_sentence(cell, "d", 0, 600, 800),
        from_sentence(other, "d", 0, 600, 800),
    ])
    assert ranked and ranked[0].region.text == "12.5 bn"
    assert ranked[0].region.resolution == "cell"
    assert ranked[0].region.box == (100, 60, 160, 85)
    assert ranked[0].region.box_frac == (100 / 600, 60 / 800, 160 / 600, 85 / 800)


def test_short_number_does_not_match_longer_number():
    region = from_sentence(_sent("112.5", (0, 0, 50, 20)),
                           "d", 0, 500, 500)
    assert rank_candidates("12.5", [region]) == []


def test_reader_region_is_a_candidate_not_trusted():
    r = from_reader_region([0.1, 0.2, 0.5, 0.6], "d", 2, 1000, 800,
                           text="65%")
    assert r is not None and r.modality == "visual"
    assert r.box == (100, 160, 500, 480)
    assert r.source == "reader"


def test_word_line_candidate_and_unrelated_claim():
    words = [WordBox(doc_id="d", page=0, text="Aurora", box=(0, 0, 60, 20),
                     block_id=1, line_id=1),
             WordBox(doc_id="d", page=0, text="42", box=(65, 0, 85, 20),
                     block_id=1, line_id=1),
             WordBox(doc_id="d", page=0, text="million", box=(90, 0, 160, 20),
                     block_id=1, line_id=1)]
    sents = [_sent("Unrelated paragraph", (0, 100, 400, 130))]
    ranked = candidates_for_claim("Aurora 42 million", sents, "d", 0, 600, 800,
                                  words=words)
    assert ranked and ranked[0].region.text == "Aurora 42 million"
    assert ranked[0].region.source == "snap"
    assert candidates_for_claim("Belgium headquarters", sents, "d", 0, 600, 800,
                                words=words) == []


def test_from_words_empty_returns_none():
    assert from_words([], "d", 0, 100, 100) is None


def test_focus_answer_beats_longer_relational_claim_match():
    from vvrag.ground.candidates import candidates_for_claim
    from vvrag.schemas import SentenceSpan

    page_sents = [
        SentenceSpan(sent_id=0, doc_id="d", page=0,
                     text="Appendiceal distension",
                     box=(40, 100, 160, 120), source="ocr"),
        SentenceSpan(sent_id=1, doc_id="d", page=0,
                     text="Irritation of parietal peritoneum (localised)",
                     box=(300, 100, 460, 120), source="ocr"),
    ]
    cands = candidates_for_claim(
        "Appendiceal distension happens before Irritation of parietal peritoneum",
        page_sents, "d", 0, 500, 300, focus="Appendiceal distension")
    assert cands[0].region.text == "Appendiceal distension"
