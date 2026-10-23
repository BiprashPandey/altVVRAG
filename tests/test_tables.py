"""Run 9 Phase 4: table-aware citation units."""
from __future__ import annotations

from vvrag.ingest.tables import (build_sentences_with_tables,
                                  number_unit_units, table_units)
from vvrag.schemas import WordBox


def _grid(rows: list[list[tuple[str, int]]], y0: int = 0,
          row_h: int = 20) -> list[WordBox]:
    out = []
    for i, cells in enumerate(rows):
        y = y0 + i * row_h
        for j, (text, x) in enumerate(cells):
            for k, tok in enumerate(text.split()):
                out.append(WordBox("doc", 0, tok,
                                   (x + k * 25, y, x + k * 25 + 22, y + 12),
                                   block_id=0, line_id=i))
    return out


TABLE_ROWS = [
    [("Revenue", 10), ("12.5", 300), ("+4.2%", 450)],
    [("Costs", 10), ("8.1", 300), ("-1.0%", 450)],
    [("Margin", 10), ("4.4", 300), ("+7.5%", 450)],
]


def test_table_units_rows_and_cells():
    words = _grid(TABLE_ROWS)
    units = table_units(words, page_width=600, first_id=7)
    assert units, "table must be detected"
    assert all(u.source == "table" for u in units)
    assert units[0].sent_id == 7
    ids = [u.sent_id for u in units]
    assert ids == list(range(7, 7 + len(units)))  # contiguous, no collisions
    row_texts = [u.text for u in units if " | " in u.text]
    assert "Revenue | 12.5 | +4.2%" in row_texts
    cell_texts = [u.text for u in units if " | " not in u.text]
    assert "12.5" in cell_texts and "+4.2%" in cell_texts
    # cell boxes are tight around the cell, not the whole row
    cell = next(u for u in units if u.text == "12.5")
    assert cell.box[0] == 300 and cell.box[2] <= 330


def test_plain_paragraph_is_not_a_table():
    words = []
    for i, line in enumerate(["This is a normal sentence about sales.",
                              "Another paragraph line continues here.",
                              "A third line finishes the thought."]):
        for k, tok in enumerate(line.split()):
            words.append(WordBox("doc", 0, tok,
                                 (10 + k * 30, i * 20, 10 + k * 30 + 26,
                                  i * 20 + 12),
                                 block_id=0, line_id=i))
    assert table_units(words, page_width=600) == []


def test_two_column_lines_table():
    rows = [
        [("Item", 10), ("Value", 400)],
        [("Alpha", 10), ("1", 400)],
        [("Beta", 10), ("2", 400)],
        [("Gamma", 10), ("3", 400)],
    ]
    units = table_units(_grid(rows), page_width=600, first_id=0)
    assert any(u.text == "Alpha | 1" for u in units)
    assert any(u.text == "3" for u in units)


def test_build_sentences_with_tables_ids_follow_text():
    from vvrag.schemas import SentenceSpan

    words = _grid(TABLE_ROWS)
    sents = [SentenceSpan(doc_id="doc", page=0, sent_id=0, text="Title",
                          box=(0, 0, 10, 10), source="textlayer")]
    combined = [sents[0]] + table_units(words, 600, first_id=1)
    assert combined[0].sent_id == 0
    assert combined[1].sent_id == 1
    assert all(u.source == "table" for u in combined[1:])
    # integration helper agrees with the manual splice
    got = build_sentences_with_tables([*words], 600)
    assert [u.source for u in got[-len(combined) + 1:]] == \
        [u.source for u in combined[1:]]


def test_number_unit_units_for_split_ocr_rows():
    words = [
        WordBox("doc", 0, "83.6", (100, 315, 170, 344), block_id=20,
                line_id=1),
        WordBox("doc", 0, "bn", (180, 315, 218, 344), block_id=20,
                line_id=1),
        WordBox("doc", 0, "12.5", (100, 563, 170, 592), block_id=20,
                line_id=1),
        WordBox("doc", 0, "bn", (180, 563, 218, 592), block_id=20,
                line_id=1),
        WordBox("doc", 0, "9.5bn", (100, 744, 196, 773), block_id=20,
                line_id=2),
        WordBox("doc", 0, "15.0%", (100, 656, 196, 683), block_id=20,
                line_id=1),
    ]
    units = number_unit_units(words, page_width=1000, first_id=7)
    texts = [u.text for u in units]
    assert "83.6 bn" in texts and "12.5 bn" in texts
    assert "9.5bn" in texts and "15.0%" in texts
    assert all(u.source == "table" for u in units)
    assert [u.sent_id for u in units] == list(range(7, 7 + len(units)))


def test_number_unit_units_do_not_pair_across_lines():
    words = [
        WordBox("doc", 0, "12.5", (100, 100, 170, 130), block_id=0,
                line_id=0),
        WordBox("doc", 0, "bn", (180, 200, 218, 230), block_id=0,
                line_id=1),
    ]
    assert number_unit_units(words, page_width=1000) == []
