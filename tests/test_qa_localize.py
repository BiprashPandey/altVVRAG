"""Post-reader localization + gate rescue (rec 1+4+6) unit tests."""
from vvrag.schemas import WordBox
from vvrag.ui.qa import localize_answer_to_boxes, span_box


def _words():
    return [WordBox("d", 0, t, (i * 60, 0, i * 60 + 50, 20), 0, 0)
            for i, t in enumerate("Fiscal Year 2024/25 results".split())]


def test_span_box_returns_per_line_list():
    words = [WordBox("d", 0, "Fiscal", (0, 0, 50, 20), 0, 0),
             WordBox("d", 0, "Year", (55, 0, 100, 20), 0, 0),
             WordBox("d", 0, "noise", (0, 100, 50, 120), 1, 0)]
    boxes = span_box("Which fiscal year?", words)
    assert isinstance(boxes, list) and len(boxes) >= 1
    # tight: must not stretch from y=0 to y=120 (old big-union bug)
    assert boxes[0][3] <= 20


def test_localize_exact_answer():
    words = _words()
    cov, boxes, score, valid = localize_answer_to_boxes("Fiscal Year 2024/25", words)
    assert [w.text for w in cov] == ["Fiscal", "Year", "2024/25"]
    assert score == 1.0 and valid and len(boxes) == 1


def test_localize_unanswerable_auto_abstain():
    words = _words()
    cov, boxes, score, valid = localize_answer_to_boxes("UNANSWERABLE", words)
    assert (cov, boxes, score, valid) == ([], [], 0.0, False)


def test_gate_rescue_logic():
    from vvrag.gate.fixed import apply_fixed
    from vvrag.gate.scorer import hand_weighted
    # q5/q7-heat case: correct answer just under tau but perfectly localized
    s = hand_weighted(0.5, 0.5, 0.2, 1.0, True)
    assert s >= 0.5  # rescued by localize (old weights would give ~0.44)
    assert apply_fixed(s, 0.5)
    # garbage stays out
    assert not apply_fixed(hand_weighted(0.0, 0.1, 0.0, 0.0, False), 0.5)
