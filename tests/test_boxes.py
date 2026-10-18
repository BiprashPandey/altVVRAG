"""Box + crop + crop-text tests, incl. Section 11 coordinate test."""
import numpy as np
import pymupdf

from vvrag.ground.boxes import (
    answer_to_line_boxes,
    grid_to_boxes,
    refine_box_to_quote,
    snap_box_to_lines,
    snap_box_to_words,
    snap_boxes_to_word_lines,
)
from vvrag.ground.crop import crop_union, overlay_heatmap
from vvrag.ground.heatmap import heat_concentration
from vvrag.ground.text_in_box import extract_crop_text, words_in_boxes
from vvrag.schemas import WordBox, grid_cell_range_to_pixel_box


def _hot_grid(gh, gw, blocks):
    g = np.zeros((gh, gw), dtype=np.float32)
    for (r0, c0, r1, c1, v) in blocks:
        g[r0:r1, c0:c1] = v
    return g


def test_coordinate_known_cell(tmp_path):
    """Section 11: known lit cell block -> pixel box within one patch tolerance."""
    W, H, gh, gw = 1000, 800, 4, 5
    g = _hot_grid(gh, gw, [(1, 2, 3, 4, 1.0)])
    boxes, mask = grid_to_boxes(g, W, H, threshold_rel=0.5, min_box_area_frac=0.0,
                                max_boxes=3, padding_px=0)
    assert len(boxes) == 1
    expected = grid_cell_range_to_pixel_box(1, 2, 3, 4, W, H, gh, gw)
    assert boxes[0] == expected == (400, 200, 800, 600)
    # tolerance: patch size is 200x200; box edges within one patch of truth
    for got, want in zip(boxes[0], expected):
        assert abs(got - want) <= 200
    assert heat_concentration(g, mask) == 1.0


def test_boxes_sorted_and_capped():
    # ranking is by SUMMED heat (spec): 2x2 @0.7 (sum 2.8) beats 1 cell @1.0
    g = _hot_grid(6, 6, [(0, 0, 1, 1, 1.0), (4, 4, 6, 6, 0.7)])
    boxes, _ = grid_to_boxes(g, 600, 600, threshold_rel=0.5, min_box_area_frac=0.0,
                             max_boxes=1, padding_px=0)
    assert boxes == [(400, 400, 600, 600)]
    # ...unless the single cell's summed heat is higher
    g2 = _hot_grid(6, 6, [(0, 0, 1, 1, 3.0), (4, 4, 6, 6, 0.7)])
    boxes2, _ = grid_to_boxes(g2, 600, 600, threshold_rel=0.5, min_box_area_frac=0.0,
                              max_boxes=1, padding_px=0)
    assert boxes2 == [(0, 0, 100, 100)]


def test_min_area_drops_specks():
    g = _hot_grid(10, 10, [(0, 0, 1, 1, 1.0)])
    boxes, _ = grid_to_boxes(g, 500, 500, threshold_rel=0.5,
                             min_box_area_frac=0.05, max_boxes=3, padding_px=0)
    assert boxes == []


def test_snap_expands_to_line_bounds():
    words = [
        WordBox("d", 0, "total", (100, 100, 150, 120), 0, 0),
        WordBox("d", 0, "$42", (155, 100, 200, 120), 0, 0),
        WordBox("d", 0, "other", (400, 400, 450, 420), 1, 0),
    ]
    snapped = snap_box_to_lines((110, 105, 180, 115), words)
    assert snapped == (100, 100, 200, 120)
    assert snap_box_to_lines((0, 0, 10, 10), words) == (0, 0, 10, 10)  # nothing inside


def test_snap_to_words_word_level():
    words = [
        WordBox("d", 0, "total", (100, 100, 150, 120), 0, 0),
        WordBox("d", 0, "$42", (155, 100, 200, 120), 0, 0),
        WordBox("d", 0, "other", (400, 400, 450, 420), 1, 0),
    ]
    inside, union = snap_box_to_words((110, 105, 180, 115), words)
    assert [w.text for w in inside] == ["total", "$42"]
    assert union == (100, 100, 200, 120)  # tight, no padding
    assert snap_box_to_words((0, 0, 10, 10), words) == ([], (0, 0, 10, 10))


def test_per_line_boxes_never_one_big_union():
    # rec 4: multi-line answer -> one tight box per line
    words = [
        WordBox("d", 0, "Nepal", (10, 10, 60, 30), 0, 0),
        WordBox("d", 0, "Rastra", (65, 10, 120, 30), 0, 0),
        WordBox("d", 0, "Bank", (10, 40, 60, 60), 0, 1),
    ]
    boxes = answer_to_line_boxes(words)
    assert len(boxes) == 2
    assert boxes[0] == (10, 10, 120, 30)
    assert boxes[1] == (10, 40, 60, 60)
    assert answer_to_line_boxes([]) == []


def test_snap_boxes_to_word_lines_strips_padding():
    words = [
        WordBox("d", 0, "a", (100, 100, 140, 120), 0, 0),
        WordBox("d", 0, "b", (10, 300, 50, 320), 1, 0),
    ]
    inside, lines = snap_boxes_to_word_lines([(0, 0, 500, 500)], words)
    assert len(lines) == 2 and len(inside) == 2
    assert snap_boxes_to_word_lines([(0, 0, 5, 5)], words) == ([], [])


def test_crop_and_text_in_box(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        page.insert_text((50, 60), "Evidence answer words here")
        pix = page.get_pixmap(dpi=100)
        img = tmp_path / "p.png"
        pix.save(img)
    words = [WordBox("d", 0, t, (50 + i * 30, 40, 75 + i * 30, 60), 0, 0)
             for i, t in enumerate("Evidence answer words here".split())]
    box = (40, 30, 200, 70)
    inside = words_in_boxes(words, [box])
    assert [w.text for w in inside] == ["Evidence", "answer", "words", "here"]
    text, has = extract_crop_text(words, [box])
    assert has and text == "Evidence answer words here"
    # multi-line visual order across blocks
    mixed = [WordBox("d", 0, "B2", (100, 60, 140, 80), 9, 9),
             WordBox("d", 0, "A1", (10, 10, 50, 30), 0, 0),
             WordBox("d", 0, "B1", (10, 60, 50, 80), 9, 9),
             WordBox("d", 0, "A2", (60, 10, 100, 30), 0, 0)]
    from vvrag.ground.text_in_box import join_words

    assert join_words(mixed) == "A1 A2 B1 B2"
    crop_path = crop_union(img, [box], tmp_path / "crop.png")
    assert tmp_path.joinpath("crop.png").exists() and crop_path
    ov = overlay_heatmap(img, np.ones((4, 4), dtype=np.float32), [box],
                         tmp_path / "ov.png")
    assert tmp_path.joinpath("ov.png").exists() and ov
    empty_text, empty_has = extract_crop_text([], [(0, 0, 5, 5)], None)
    assert (empty_text, empty_has) == ("", False)


def test_refine_box_to_quote_tightens_table_cell():
    # P4/E: a wide table-row box collapses to the single cell the quote names.
    row = [WordBox("d", 0, t, (100 + i * 60, 100, 150 + i * 60, 120), 0, 0)
           for i, t in enumerate("Revenue 42 bn Margin 17 pct".split())]
    row_box = (100, 100, 100 + 4 * 60, 120)  # whole row
    refined = refine_box_to_quote(row, "42 bn", row_box)
    assert refined != row_box
    assert refined[0] >= 100 and refined[2] <= row_box[2]  # inside the row
    # tightened to just the value words (not the "Revenue"/"Margin" labels)
    assert refined[0] > row_box[0] and refined[2] < row_box[2]


def test_refine_box_to_quote_falls_back_when_no_match():
    row = [WordBox("d", 0, "Alpha", (0, 0, 50, 20), 0, 0),
           WordBox("d", 0, "Beta", (60, 0, 110, 20), 0, 0)]
    box = (0, 0, 110, 20)
    assert refine_box_to_quote(row, "totally absent quote", box) == box
    assert refine_box_to_quote(row, "", box) == box
    assert refine_box_to_quote([], "Alpha", box) == box
