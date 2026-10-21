"""Run 9 (guideline s7): the three-state truth table, end to end in mock mode.

verified        -- examiner confirms + quote on page + details present + citations resolve
not_confident   -- answer shown, suffix + reason (quote failure, examiner failure, ...)
nothing_found   -- the only abstention: reader truly saw nothing (no_evidence)
"""
import pymupdf

import vvrag.reader.client as client_mod
from vvrag.ingest.index import connect_sqlite
from vvrag.ingest.run_ingest import run_ingest
from vvrag.ingest.sentences import sentences_from_page
from vvrag.schemas import ReaderSample
from vvrag.ui.qa import answer_question


def _write_pdf(raw):
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 40), "Quarterly report")
        page.insert_text((50, 120), "The Aurora revenue was 42 million in 2011.")
        doc.save(raw / "deck.pdf")


def _setup(tmp_path, samples):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_pdf(raw)
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=150, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))
    con = connect_sqlite(index / "meta.db")
    doc_id, page = con.execute("SELECT doc_id, page FROM pages LIMIT 1").fetchone()
    sents, _ = sentences_from_page(con, doc_id, page)
    return index, doc_id, sents


def _patch(monkeypatch, sample_factory):
    def fake_run(reader, images, prompt, n_samples=3, temperature=0.0, max_claims=5):
        return [sample_factory()], 1

    monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)
    monkeypatch.setattr(client_mod, "create_reader", lambda *a, **k: object())


def test_state_tables(monkeypatch, tmp_path):
    index, doc_id, sents = _setup(tmp_path, None)
    target = next(s for s in sents if "42 million" in s.text)

    # ---- verified: verbatim quote + answer in the cited sentences ---------
    # (the year lives in the NEXT sentence unit; a careful reader cites both)
    year_sid = [target.sent_id, target.sent_id + 1]
    _patch(monkeypatch, lambda: ReaderSample(
        sample_id=0, answer="42 million",
        quote="Aurora revenue was 42 million in 2011",
        sentence_ids=year_sid, claims=[], found_in="body text",
        confidence=0.9))
    r = answer_question("What was the Aurora revenue in 2011?", str(index),
                        str(tmp_path / "w1"), reader_mode="mock", n_samples=1,
                        grounding="span", gate="none", doc_id=doc_id)
    assert r["state"] == "verified", r["state_reason"]
    assert r["verified"] is True and r["abstained"] is False
    assert r["tier"] == "text"
    assert r["examiner"]["ran"] and r["examiner"]["confirmed"]
    assert r["examiner"]["served_by"] == "mock/examiner"
    assert r["reader_served_by"] == "mock/mock"
    assert r["details"]["ok"] is True
    assert r["evidence"]["sentence_ids"] == year_sid

    # ---- not_confident: fabricated quote -> shown with a reason -----------
    _patch(monkeypatch, lambda: ReaderSample(
        sample_id=0, answer="42 million", quote="totally unrelated words here",
        sentence_ids=[target.sent_id], claims=[]))
    r2 = answer_question("What was the Aurora revenue in 2011?", str(index),
                         str(tmp_path / "w2"), reader_mode="mock", n_samples=1,
                         grounding="span", gate="none", doc_id=doc_id)
    assert r2["state"] == "not_confident", r2["state_reason"]
    assert r2["abstained"] is False and r2["verified"] is False
    assert r2["answer"] == "42 million"
    assert r2["state_reason"]
    assert r2["evidence"]["boxes"]  # source still shown

    # ---- not_confident: asked-for year missing from the evidence ----------
    _patch(monkeypatch, lambda: ReaderSample(
        sample_id=0, answer="42 million", quote="Aurora revenue was 42 million",
        sentence_ids=[target.sent_id], claims=[]))
    r3 = answer_question("What was the Aurora revenue in 2021?", str(index),
                         str(tmp_path / "w3"), reader_mode="mock", n_samples=1,
                         grounding="span", gate="none", doc_id=doc_id)
    assert r3["state"] == "not_confident"
    assert any("2021" in w for w in r3["details"]["warnings"]), r3["details"]
    assert "2021" in r3["state_reason"]

    # ---- nothing_found: the only abstention --------------------------------
    _patch(monkeypatch, lambda: ReaderSample(
        sample_id=0, answer="UNANSWERABLE", quote="", sentence_ids=[],
        claims=[], no_evidence=True))
    r4 = answer_question("What is the CEO's salary?", str(index),
                         str(tmp_path / "w4"), reader_mode="mock", n_samples=1,
                         grounding="span", gate="none", doc_id=doc_id)
    assert r4["state"] == "nothing_found" and r4["abstained"] is True
    assert r4["state_label"] == "Nothing found"
    assert r4["nothing_found_text"]
    assert r4["closest_passages"] == [] or isinstance(r4["closest_passages"], list)
