"""Retrieval acceptance (Section 12, Phase 3): Hit@3 on a 20-question debug set
+ local MaxSim ordering matches Qdrant's multivector ordering."""
import numpy as np
import pymupdf

from vvrag.ingest import index as index_mod
from vvrag.ingest.embed import load_embedding
from vvrag.ingest.run_ingest import run_ingest
from vvrag.retrieve.maxsim import maxsim_rank
from vvrag.retrieve.query_embed import FakeQueryEmbedder
from vvrag.retrieve.two_stage import TwoStageRetriever, retrieval_margin


def _make_pdf(path, seed_text, pages=2):
    with pymupdf.open() as doc:
        for p in range(pages):
            page = doc.new_page()
            y = 72
            for i in range(15):
                page.insert_text((72, y), f"{seed_text} page {p} filler words {i}")
                y += 20
        doc.save(path)


def _build_index(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("docA", "docB", "docC"):
        _make_pdf(raw / f"{name}.pdf", f"deck {name} content", pages=2)
    out = tmp_path / "index"
    run_ingest(raw, out, dpi=100, embedder="fake", skip_ocr=True,
               qdrant_path=str(out / "qdrant"))
    client = index_mod.get_qdrant_client("local", str(out / "qdrant"))
    return out, client


def _query_from_gold(out, doc_id, page, variant, n_tokens=4, noise=0.1):
    """Query tokens = perturbed copies of gold-page patches (guaranteed near)."""
    pe = load_embedding(out / "embeddings", doc_id, page)
    import hashlib

    seed = int(hashlib.sha256(f"{doc_id}/{page}/{variant}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    idx = rng.choice(pe.vectors.shape[0], size=n_tokens, replace=False)
    q = pe.vectors[idx] + noise * rng.standard_normal((n_tokens, pe.dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    return q.astype(np.float32)


def test_fake_query_embedder_deterministic():
    e = FakeQueryEmbedder()
    v1, t1 = e.embed_query("What is the total revenue?")
    v2, t2 = e.embed_query("What is the total revenue?")
    assert t1 == ["what", "is", "the", "total", "revenue?"]
    np.testing.assert_array_equal(v1, v2)


def test_two_stage_hit_rate(tmp_path):
    out, client = _build_index(tmp_path)
    ret = TwoStageRetriever(client, top_k=3, prefetch_k_multiplier=4)
    pages = [(d, p) for d in ("docA", "docB", "docC") for p in (0, 1)]
    queries = [(d, p, v) for (d, p) in pages for v in range(4)][:20]
    assert len(queries) == 20
    hits = 0
    for doc_id, page, variant in queries:
        q = _query_from_gold(out, doc_id, page, variant)
        res = ret.search(q)
        assert len(res) == 3
        assert retrieval_margin(res) >= 0.0
        if any(h.doc_id == doc_id and h.page == page for h in res):
            hits += 1
    assert hits / len(queries) >= 0.8, f"Hit@3 = {hits}/20"


def test_local_maxsim_matches_qdrant_ordering(tmp_path):
    out, client = _build_index(tmp_path)
    pages = [(d, p) for d in ("docA", "docB", "docC") for p in (0, 1)]
    full_vecs = [load_embedding(out / "embeddings", d, p).vectors for d, p in pages]
    q = _query_from_gold(out, "docB", 1, variant=0)
    local_order, _ = maxsim_rank(q, full_vecs)
    res = client.query_points("pages_full", query=q.tolist(), limit=6)
    qdrant_order = [(pt.payload["doc_id"], pt.payload["page"]) for pt in res.points]
    assert qdrant_order[0] == pages[local_order[0]]
    assert [pages[i] for i in local_order[:3]] == qdrant_order[:3]
