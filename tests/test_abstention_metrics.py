"""Abstention + sentence-citation metric tests (suggestions #1, #6, #8)."""
import numpy as np

from vvrag.eval.metrics import abstention_metrics, sentence_precision_recall


def test_sentence_precision_recall_basic():
    assert sentence_precision_recall([1, 2], [1, 2]) == (1.0, 1.0, 1.0)
    p, r, f = sentence_precision_recall([1, 2, 3], [1, 2])
    assert p == 2 / 3 and r == 1.0
    p, r, f = sentence_precision_recall([1], [1, 2, 3])
    assert p == 1.0 and r == 1 / 3
    assert sentence_precision_recall([], [1]) == (0.0, 0.0, 0.0)
    assert sentence_precision_recall([], []) == (1.0, 1.0, 1.0)


def test_abstention_metrics_splits_populations():
    # 4 questions: 3 answerable (2 answered+correct, 1 abstained),
    # 1 unanswerable (correctly abstained)
    answered = np.array([True, True, False, False])
    correct = np.array([True, True, False, False])
    unanswerable = np.array([False, False, False, True])
    m = abstention_metrics(answered, correct, unanswerable)
    assert m["n"] == 4
    assert m["coverage"] == 0.5
    assert m["answerable_coverage"] == 2 / 3  # 2 of 3 answerable answered
    assert m["unanswerable_abstention_rate"] == 1.0  # abstained on the fake
    assert m["false_answer_rate_on_unanswerable"] == 0.0
    assert m["n_answerable"] == 3 and m["n_unanswerable"] == 1


def test_abstention_metrics_without_mask():
    answered = np.array([True, False])
    correct = np.array([True, False])
    m = abstention_metrics(answered, correct)
    assert m["coverage"] == 0.5 and "unanswerable_abstention_rate" not in m


def test_abstention_metrics_gate_that_answers_everything():
    answered = np.array([True, True])
    correct = np.array([True, False])
    una = np.array([False, True])
    m = abstention_metrics(answered, correct, una)
    assert m["false_answer_rate_on_unanswerable"] == 1.0
    assert m["unanswerable_abstention_rate"] == 0.0
