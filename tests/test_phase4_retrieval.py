"""Run 9 Phase 4: lexical reranker, bridge resolution, online-encoder guard.

All pure CPU, no network, no model.
"""
from __future__ import annotations

import numpy as np
import pymupdf
import pytest

from vvrag.ingest.index import connect_sqlite
from vvrag.ingest.run_ingest import run_ingest
from vvrag.retrieve.bridge import (missing_tokens, question_tokens,
                                   resolve_bridge)
from vvrag.retrieve.fusion import rrf_fuse
from vvrag.retrieve.rerank import lexical_rerank
from vvrag.schemas import WordBox


def _w(text: str, x: int = 0, y: int = 0) -> WordBox:
    return WordBox(doc_id="d", page=0, text=text, box=(x, y, x + 10, y + 10))


def test_lexical_rerank_prefers_rare_token_page():
    q = "What was the Organic Growth in 2011?"
    words_big = [_w("sales"), _w("of"), _w("things")] * 30  # long noise page
    words_hit = [_w("Organic"), _w("Growth"), _w("was"), _w("14.9%")]
    ranked = [("docA", 0, 0.9, words_big), ("docB", 1, 0.2, words_hit)]
    out = lexical_rerank(q, ranked, k=5)
    assert (out[0][0], out[0][1]) == ("docB", 1)  # rare tokens beat noise
    # scores are rank-normalized so downstream margins are positive
    assert out[0][2] > out[1][2] > 0
    # k limits the output
    assert len(lexical_rerank(q, ranked, k=1)) == 1
    # single candidate is a no-op
    one = [("docA", 0, 0.5, words_big)]
    assert lexical_rerank(q, one) == one


def test_lexical_rerank_number_bonus_and_determinism():
    q = "How many employees in 2010?"
    hit = [("docA", 0, 0.5, [_w("employees"), _w("2010"), _w("3,274")])]
    miss = [("docB", 0, 0.6, [_w("employees"), _w("and"), _w("growth")])]
    out1 = lexical_rerank(q, hit + miss, k=5)
    out2 = lexical_rerank(q, hit + miss, k=5)
    assert out1 == out2  # deterministic
    assert out1[0][0] == "docA"


def test_rrf_weights_prefer_visual_and_reject_negative():
    kw = [("docA", 0), ("docB", 1)]
    vis = [("docB", 1), ("docA", 0)]
    equal = rrf_fuse([kw, vis], k=60)
    assert equal[0][2] == equal[1][2]  # symmetric ranks tie
    visual_heavy = rrf_fuse([kw, vis], k=60, weights=[1.0, 5.0])
    assert (visual_heavy[0][0], visual_heavy[0][1]) == ("docB", 1)
    with pytest.raises(ValueError, match="non-negative"):
        rrf_fuse([kw, vis], weights=[1.0, -1.0])


def test_bridge_missing_tokens_and_resolution():
    q = ("In the year Nestle achieved the third largest Organic Growth in "
         "10 years, what was the dividend?")
    primary_words = [_w("dividend"), _w("3.6"), _w("percent")]
    missing = missing_tokens(q, primary_words)
    assert "organic" in missing and "growth" in missing
    assert "dividend" not in missing

    words_map = {
        ("d", 0): primary_words,
        ("d", 1): [_w("Unrelated"), _w("page")],
        ("d", 2): [_w("third"), _w("largest"), _w("Organic"), _w("Growth"),
                   _w("in"), _w("10"), _w("years")],
    }
    got = resolve_bridge(q, ("d", 0), [("d", 0)], words_map, max_extra=1)
    assert got and (got[0][0], got[0][1]) == ("d", 2)
    assert got[0][2] > 0
    # pages already in context are never returned
    assert resolve_bridge(q, ("d", 0), [("d", 0), ("d", 2)], words_map,
                          max_extra=1) == []
    # single-hop (no missing tokens) -> no bridge
    assert resolve_bridge("What is the dividend?", ("d", 0), [("d", 0)],
                          words_map) == []


def test_bridge_min_missing_threshold():
    q = "In the year Organic Growth happened, what was the dividend?"
    primary = [_w("dividend"), _w("percent")]
    words_map = {("d", 0): primary, ("d", 1): [_w("Organic"), _w("Growth")]}
    # only one missing token ("organic" xor "growth" both missing and hit ->
    # 2 missing >= threshold 2 -> fires; with min_missing=5 -> never)
    got = resolve_bridge(q, ("d", 0), [("d", 0)], words_map, min_missing=5)
    assert got == []
    assert question_tokens("What was the Organic Growth?") == ["organic",
                                                               "growth"]


def test_online_encoder_guard_and_fallback(monkeypatch, tmp_path):
    """Hybrid + no cache + encoder unavailable => keyword fallback, honest note."""
    from vvrag.ui import qa

    def _boom(cfg_ret):
        raise RuntimeError("forced-unavailable")

    monkeypatch.setattr(qa, "_check_online_encoder", _boom)
    res = qa.answer_question(
        "What was the Organic Growth of Nestle?", "data/index",
        str(tmp_path / "w"), reader_mode="mock", retrieval_mode="hybrid")
    rd = res.get("retrieval_debug") or {}
    assert rd.get("mode_used") == "keyword"
    assert "forced-unavailable" in (rd.get("note") or "")


def test_visual_scoring_failure_falls_back(monkeypatch, tmp_path):
    """Encoder succeeds but scoring raises (dim mismatch) => keyword fallback."""
    from vvrag.ui import qa
    from vvrag.retrieve import query_vectors as qv

    class _Fake:
        dim = 4

        def __init__(self, *a, **k):
            pass

        def get(self, text):
            import numpy as np
            return np.ones((2, 4), dtype="float32"), ["q1", "q2"]

    monkeypatch.setattr(qa, "_check_online_encoder", lambda cfg: None)
    monkeypatch.setattr(qv, "QueryVectorStore", _Fake)
    res = qa.answer_question(
        "What was the Organic Growth of Nestle?", "data/index",
        str(tmp_path / "w2"), reader_mode="mock", retrieval_mode="hybrid")
    rd = res.get("retrieval_debug") or {}
    assert rd.get("mode_used") == "keyword"
    assert "visual scoring failed" in (rd.get("note") or "")


def test_visual_scope_all_can_promote_page_keyword_missed(monkeypatch, tmp_path):
    """Report parity: a page keyword ranked second can win the visual order."""
    from vvrag.ui import qa
    from vvrag.retrieve import query_vectors as qv
    from vvrag.retrieve import fusion

    raw = tmp_path / "raw"
    raw.mkdir()
    with pymupdf.open() as doc:
        p0 = doc.new_page()
        p0.insert_text((50, 50), "Alpha appendix material")
        p1 = doc.new_page()
        p1.insert_text((50, 50), "Beta target page")
        doc.save(raw / "deck.pdf")
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=100, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))
    con = connect_sqlite(index / "meta.db")
    page0_key = ("deck", 0)
    page1_key = ("deck", 1)
    assert page0_key != page1_key

    class _Store:
        def __init__(self, *a, **k):
            pass

        def get(self, text):
            return np.ones((1, 4), dtype="float32"), ["q"]

    monkeypatch.setattr(qv, "QueryVectorStore", _Store)
    monkeypatch.setattr(qa, "_check_online_encoder", lambda cfg: None)
    monkeypatch.setattr(
        fusion, "visual_page_scores",
        lambda index_dir, pages, qvecs: {page0_key: 0.1, page1_key: 0.9})

    res = qa.answer_question("Beta target", str(index), str(tmp_path / "w"),
                             reader_mode="mock", retrieval_mode="visual",
                             n_samples=1)
    rd = res.get("retrieval_debug") or {}
    assert rd["mode_used"] == "visual"
    assert rd["pages_context"][0] == {"doc_id": "deck", "page": 1}
