"""Determinism: seeded helpers must be stable across processes (no hash())."""
from vvrag.eval.arms import stable_seed


def test_stable_seed():
    assert stable_seed("a", "1") == stable_seed("a", "1")
    assert stable_seed("a", "1") != stable_seed("a", "2")
    assert stable_seed("a", "1") != stable_seed("b", "1")
