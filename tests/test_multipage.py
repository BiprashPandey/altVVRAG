"""Phase 2: multi-page reading + namespaced sentence citations.

Two-page fake-embedder index. The fake reader cites a sentence on the SECOND
page via the namespaced id "p1.sN" (or "p0.sN" after a next-page promotion);
the pipeline must ground boxes on that page, keep the legacy primary-page
fields intact, and record the promotion.
"""
import pymupdf

import vvrag.reader.client as client_mod
from vvrag.ingest.index import connect_sqlite
from vvrag.ingest.run_ingest import run_ingest
from vvrag.ingest.sentences import sentences_from_page
from vvrag.schemas import ReaderSample
from vvrag.ui.qa import answer_question


def _write_pdf2(raw):
    with pymupdf.open() as doc:
        p1 = doc.new_page(width=600, height=800)
        p1.insert_text((50, 40), "Quarterly report")
        p1.insert_text((50, 120), "The Aurora revenue grew in Q2.")
        p2 = doc.new_page(width=600, height=800)
        p2.insert_text((50, 40), "Notes")
        p2.insert_text((50, 120), "The Aurora team increased the price in June.")
        doc.save(raw / "deck.pdf")


def test_citation_on_secondary_page(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_pdf2(raw)
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=150, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))
    con = connect_sqlite(index / "meta.db")
    doc_id = con.execute("SELECT DISTINCT doc_id FROM pages").fetchone()[0]
    sents2, _ = sentences_from_page(con, doc_id, 1)
    target = next(s for s in sents2 if "June" in s.text)

    def fake_run(reader, images, prompt, n_samples=3, temperature=0.0, max_claims=5):
        return [ReaderSample(sample_id=0, answer="in June",
                             quote="increased the price in June",
                             sentence_ids=[target.sent_id],
                             sentence_refs=[[1, target.sent_id]], claims=[])], 1

    monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)
    monkeypatch.setattr(client_mod, "create_reader", lambda *a, **k: object())

    r = answer_question("What did the Aurora revenue do?", str(index),
                        str(tmp_path / "work"), reader_mode="live", n_samples=1,
                        grounding="span", gate="none")

    assert not r["abstained"]
    ctx = r["retrieval_debug"]["pages_context"]
    assert [c["page"] for c in ctx][:2] == [0, 1]  # page 0 is the primary
    ev = r["evidence"]
    assert ev["localize"]["source"] == "sentence_ids"
    assert ev["localize"]["quote_ok"] is True
    cit = r["cited_sentences"][0]
    assert cit["page"] == 1 and cit["doc_id"] == doc_id
    assert cit["sent_id"] == target.sent_id
    pages = ev["pages"]
    assert len(pages) == 1 and pages[0]["page"] == 1
    (x0, y0, x1, y1) = pages[0]["boxes"][0]
    (tx0, ty0, tx1, ty1) = target.box
    assert tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1
    assert ev["refs"] == [[doc_id, 1, target.sent_id]]


def test_next_page_retry_promotes_second_page(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_pdf2(raw)
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=150, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))
    con = connect_sqlite(index / "meta.db")
    doc_id = con.execute("SELECT DISTINCT doc_id FROM pages").fetchone()[0]
    sents2, _ = sentences_from_page(con, doc_id, 1)
    target = next(s for s in sents2 if "June" in s.text)

    calls = {"n": 0}

    def fake_run(reader, images, prompt, n_samples=3, temperature=0.0, max_claims=5):
        calls["n"] += 1
        if calls["n"] <= 2:  # original + same-page corrective retry: abstain
            return [ReaderSample(sample_id=0, answer="UNANSWERABLE",
                                 quote="", sentence_ids=[], claims=[])], 1
        # next-page attempt: the promoted page is now label p0
        return [ReaderSample(sample_id=0, answer="in June",
                             quote="increased the price in June",
                             sentence_ids=[target.sent_id],
                             sentence_refs=[[0, target.sent_id]], claims=[])], 1

    monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)
    monkeypatch.setattr(client_mod, "create_reader", lambda *a, **k: object())

    r = answer_question("What did the Aurora revenue do?", str(index),
                        str(tmp_path / "work"), reader_mode="live", n_samples=1,
                        grounding="span", gate="none")

    assert calls["n"] == 3
    assert not r["abstained"]
    assert r["answer"] == "in June"
    pro = r["retrieval_debug"]["primary_promoted"]
    assert pro == {"doc_id": doc_id, "page": 1}
    ev = r["evidence"]
    assert ev["doc_id"] == doc_id and ev["page"] == 1
    assert ev["localize"]["quote_ok"] is True
    assert r["cited_sentences"][0]["page"] == 1
