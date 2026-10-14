"""Schema + coordinate-helper tests (Section 11: coordinate test)."""
from vvrag.schemas import (
    Evidence,
    Result,
    clip_box_to_page,
    grid_cell_range_to_pixel_box,
)


def test_grid_cell_to_pixel_box_center_cell():
    # Synthetic page 1000x800, grid 4x5. Light up cell (r=1..3, c=2..4).
    box = grid_cell_range_to_pixel_box(1, 2, 3, 4, 1000, 800, 4, 5)
    # x: 2*1000/5=400 .. 4*1000/5=800 ; y: 1*800/4=200 .. 3*800/4=600
    assert box == (400, 200, 800, 600)


def test_grid_cell_to_pixel_box_with_padding_and_clip():
    box = grid_cell_range_to_pixel_box(0, 0, 1, 1, 100, 100, 10, 10, padding_px=8)
    # raw (0,0,10,10) padded -> (-8,-8,18,18) clipped -> (0,0,18,18)
    assert box == (0, 0, 18, 18)


def test_clip_box_to_page():
    assert clip_box_to_page((-5, -5, 2000, 2000), 100, 100) == (0, 0, 100, 100)


def test_result_serialization_roundtrip():
    r = Result(
        question="q",
        answer="a",
        evidence=Evidence(doc_id="d", page=1, boxes=[(1, 2, 3, 4)]),
    )
    r2 = Result.from_dict(r.to_dict())
    assert r2.question == "q" and r2.evidence.boxes == [(1, 2, 3, 4)]
