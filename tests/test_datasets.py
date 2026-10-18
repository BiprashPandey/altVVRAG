"""SlideVQA row -> QAItem mapping tests (P3/C: equivalent golds, 0-indexed page).

Hermetic: feeds plain dicts to _slidevqa_item, no network / image decoding.
"""
from vvrag.eval.datasets import _slidevqa_item, build_synthetic_set, deck_split


def test_slidevqa_row_maps_variants_and_page():
    row = {
        "deck_name": "nestle-roadshow",
        "qa_id": 7,
        "question": "How much is the Trading Operating Profit in 2011?",
        "answer": ["12.5 bn", "12.5 billion"],
        "evidence_pages": [5],
    }
    it = _slidevqa_item(row, 0)
    assert it.deck_id == "nestle-roadshow"
    assert it.gold_doc == "nestle-roadshow"
    assert it.gold_page == 4  # evidence_pages 1-indexed -> 0-indexed
    assert it.gold_answer == "12.5 bn"
    assert it.acceptable_answers == ["12.5 billion"]
    assert it.golds() == ["12.5 bn", "12.5 billion"]  # P3: both are correct
    assert it.is_unanswerable is False


def test_slidevqa_row_unanswerable_and_nested_evidence():
    row = {
        "document": {"id": "accel-ecom"},
        "qa_id": "accel-ecom-0",
        "question": "What is the missing figure?",
        "answer": [],
        "evidence": {"pages": [3]},
    }
    it = _slidevqa_item(row, 2)
    assert it.deck_id == "accel-ecom"
    assert it.gold_page == 2
    assert it.is_unanswerable is True
    assert it.gold_answer == "UNANSWERABLE"


def test_synthetic_set_items_are_answerable_and_have_golds():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        es = build_synthetic_set(d, n_docs=2, pages_per_doc=1, seed=0,
                                 unanswerable_per_doc=1)
    answerables = [it for it in es.items if not it.is_unanswerable]
    assert answerables and all(it.golds() for it in answerables)
    train, calib, test = deck_split(es.items, calib_frac=0.3, seed=0)
    assert len(train) + len(calib) + len(test) == len(es.items)
