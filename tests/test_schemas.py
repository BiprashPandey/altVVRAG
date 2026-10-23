"""Schema + coordinate-helper tests (Section 11: coordinate test)."""
from vvrag.schemas import (
    Answer,
    Claim,
    Evidence,
    GroundedRegion,
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


def test_grounded_region_roundtrip():
    r = GroundedRegion(
        doc_id="deck", page=3, box=(10, 20, 110, 220),
        box_frac=(0.1, 0.2, 0.5, 0.8), text="12.5 bn",
        modality="table", resolution="cell", score=1.0,
        source="gold",
    )
    r2 = GroundedRegion.from_dict(r.to_dict())
    assert r2 == r


def test_claim_and_answer_contracts_roundtrip():
    c = Claim(
        text="Revenue was 12.5 bn", evidence_sentence_ids=[33],
        regions=[GroundedRegion(doc_id="d", page=1, box=(0, 0, 10, 10),
                                box_frac=(0.0, 0.0, 0.1, 0.1),
                                text="12.5 bn")],
        label="supported", reason="", confidence=0.9,
        citation_specific=True, withheld=False,
    )
    c2 = Claim.from_dict(c.to_dict())
    assert c2.text == c.text
    assert c2.regions[0].text == "12.5 bn"
    assert c2.citation_specific is True
    a = Answer(question="q", text="12.5 bn", claims=[c],
               state="verified", state_reason="", verified=True)
    a2 = Answer.from_dict(a.to_dict())
    assert a2.state == "verified" and a2.verified
    assert a2.claims[0].regions[0].doc_id == "d"
