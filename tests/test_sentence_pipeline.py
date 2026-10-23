"""End-to-end sentence-citation path (suggestions #1, #4, #6).

Uses a real (fake-embedder) index built from a text-layer PDF and a stubbed
reader that cites a specific sentence, then checks the final evidence boxes are
exactly that sentence's box (never a heat blob), plus the quote/verifier wiring.
"""
import pymupdf

import vvrag.reader.client as client_mod
from vvrag.ingest.index import connect_sqlite
from vvrag.ingest.run_ingest import run_ingest
from vvrag.ingest.sentences import sentences_from_page
from vvrag.schemas import Claim, ReaderSample
from vvrag.ui.qa import answer_question


def _write_pdf(raw):
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 40), "Quarterly report")
        page.insert_text((50, 120), "The Aurora revenue was 42 million in this quarter.")
        page.insert_text((50, 220), "Page 12")
        doc.save(raw / "deck.pdf")


def test_citation_boxes_come_from_cited_sentence(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_pdf(raw)
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=150, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))

    con = connect_sqlite(index / "meta.db")
    doc_id, page = con.execute("SELECT doc_id, page FROM pages LIMIT 1").fetchone()
    sents, _ = sentences_from_page(con, doc_id, page)
    target = next(s for s in sents if "42 million" in s.text)

    def fake_run(reader, images, prompt, n_samples=3, temperature=0.0, max_claims=5):
        return [ReaderSample(sample_id=0, answer="42 million",
                             quote="Aurora revenue was 42 million",
                             sentence_ids=[target.sent_id],
                             claims=[Claim(text="Aurora revenue was 42 million")])], 1

    monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)
    monkeypatch.setattr(client_mod, "create_reader", lambda *a, **k: object())

    r = answer_question("What was the Aurora revenue?", str(index), str(tmp_path / "work"),
                        reader_mode="live", n_samples=1, grounding="span",
                        gate="none", doc_id=doc_id)

    ev = r["evidence"]
    assert not r["abstained"]
    assert ev["localize"]["source"] == "sentence_ids"
    assert ev["localize"]["quote_ok"] is True
    assert ev["sentence_ids"] == [target.sent_id]
    # P4/E: the box is tightened to the quote's words, but stays inside the
    # cited sentence (never a heat blob, never outside the cited unit).
    (x0, y0, x1, y1) = ev["boxes"][0]
    (tx0, ty0, tx1, ty1) = target.box
    assert tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1
    assert (x1 - x0) < (tx1 - tx0)  # strictly tighter than the whole sentence
    assert r["cited_sentences"][0]["sent_id"] == target.sent_id
    assert r["cited_sentences"][0]["text"] == target.text
    assert r["claim_regions"]
    assert r["claim_regions"][0]["text"] == "Aurora revenue was 42 million"
    assert r["claim_regions"][0]["best"] is not None
    assert r["claim_regions"][0]["best"]["doc_id"] == doc_id


def test_quote_only_reader_derives_unique_sentence(monkeypatch, tmp_path):
    # P3: a table answer may be returned with a verbatim quote but no
    # sentence_ids. If exactly one shown sentence contains that quote, derive
    # the citation deterministically (auditable via derived_from_quote).
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_pdf(raw)
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=150, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))
    con = connect_sqlite(index / "meta.db")
    doc_id, page = con.execute("SELECT doc_id, page FROM pages LIMIT 1").fetchone()
    sents, _ = sentences_from_page(con, doc_id, page)
    target = next(s for s in sents if "42 million" in s.text)

    def fake_run(reader, images, prompt, n_samples=3, temperature=0.0, max_claims=5):
        return [ReaderSample(sample_id=0, answer="42 million",
                             quote="Aurora revenue was 42 million",
                             sentence_ids=[], claims=[])], 1

    monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)
    monkeypatch.setattr(client_mod, "create_reader", lambda *a, **k: object())

    r = answer_question("What was the Aurora revenue?", str(index),
                        str(tmp_path / "work"), reader_mode="live",
                        n_samples=1, grounding="span", gate="none",
                        doc_id=doc_id)
    assert r["citations"]["derived_from_quote"] is True
    assert r["evidence"]["sentence_ids"] == [target.sent_id]
    assert r["evidence"]["localize"]["quote_ok"] is True
    assert r["cited_sentences"][0]["sent_id"] == target.sent_id


def test_bad_quote_triggers_retry_then_not_confident(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_pdf(raw)
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=150, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))
    con = connect_sqlite(index / "meta.db")
    doc_id, page = con.execute("SELECT doc_id, page FROM pages LIMIT 1").fetchone()
    sents, _ = sentences_from_page(con, doc_id, page)
    sid = sents[0].sent_id

    calls = {"n": 0}

    def fake_run(reader, images, prompt, n_samples=3, temperature=0.0, max_claims=5):
        calls["n"] += 1
        # both original and retry cite a real sentence but quote unrelated words
        return [ReaderSample(sample_id=0, answer="nonsense answer",
                             quote="totally unrelated fabricated words",
                             sentence_ids=[sid], claims=[])], 1

    monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)
    monkeypatch.setattr(client_mod, "create_reader", lambda *a, **k: object())

    r = answer_question("What was the Aurora revenue?", str(index), str(tmp_path / "work"),
                        reader_mode="live", n_samples=1, grounding="span",
                        gate="none", doc_id=doc_id)
    assert calls["n"] == 2  # original + one retry
    # Run 9 (guideline s7): a failed quote means "not confident enough" --
    # the answer is still shown with its best source, never hidden.
    assert r["abstained"] is False
    assert r["state"] == "not_confident"
    assert r["answer"] == "nonsense answer"
    assert "quote" in r["state_reason"], r["state_reason"]
    assert r["low_confidence_suffix"]


def test_unanswerable_retry_recovers_ocr_noise_answer(monkeypatch, tmp_path):
    # suggestion #7 (legacy opt-in): reader first says UNANSWERABLE (OCR
    # noise), but a corrective retry with the closest passages recovers a
    # verifiable answer. retry_on_unanswerable now defaults to False; the
    # wiring is exercised here with an explicit config.
    import vvrag.config as config_mod

    cfg = config_mod.AppConfig()
    cfg.reader.retry_on_unanswerable = True
    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: cfg)

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_pdf(raw)
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=150, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))
    con = connect_sqlite(index / "meta.db")
    doc_id, page = con.execute("SELECT doc_id, page FROM pages LIMIT 1").fetchone()
    sents, _ = sentences_from_page(con, doc_id, page)
    target = next(s for s in sents if "42 million" in s.text)

    calls = {"n": 0}

    def fake_run(reader, images, prompt, n_samples=3, temperature=0.0, max_claims=5):
        calls["n"] += 1
        if calls["n"] == 1:  # first pass: abstain (false negative on OCR)
            return [ReaderSample(sample_id=0, answer="UNANSWERABLE",
                                 quote="", sentence_ids=[], claims=[])], 1
        # retry pass: valid cited answer
        return [ReaderSample(sample_id=0, answer="42 million",
                             quote="Aurora revenue was 42 million",
                             sentence_ids=[target.sent_id], claims=[])], 1

    monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)
    monkeypatch.setattr(client_mod, "create_reader", lambda *a, **k: object())

    r = answer_question("What was the Aurora revenue?", str(index), str(tmp_path / "work"),
                        reader_mode="live", n_samples=1, grounding="span",
                        gate="none", doc_id=doc_id)
    assert calls["n"] == 2  # original + one unanswerable retry
    assert r["abstained"] is False
    assert r["answer"] == "42 million"
    assert r["evidence"]["localize"]["quote_ok"] is True
    assert r["evidence"]["sentence_ids"] == [target.sent_id]


def test_page_number_citation_is_not_specific(monkeypatch, tmp_path):
    # Report gap: a true claim can be attached to a fabricated citation. The
    # deterministic precheck must reject a page-number-only cited sentence even
    # when the mock examiner finds that quoted page number.
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_pdf(raw)
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=150, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))
    con = connect_sqlite(index / "meta.db")
    doc_id, page = con.execute("SELECT doc_id, page FROM pages LIMIT 1").fetchone()
    sents, _ = sentences_from_page(con, doc_id, page)
    page_num = next(s for s in sents if "Page 12" in s.text)

    def fake_run(reader, images, prompt, n_samples=3,
                 temperature=0.0, max_claims=5):
        return [ReaderSample(sample_id=0, answer="42 million",
                             quote="Page 12",
                             sentence_ids=[page_num.sent_id], claims=[])], 1

    monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)

    r = answer_question("What was the Aurora revenue?", str(index),
                        str(tmp_path / "work"), reader_mode="mock",
                        n_samples=1, grounding="span", gate="none",
                        doc_id=doc_id)
    assert r["answer"] == "42 million"          # still shown
    assert r["abstained"] is False
    assert r["state"] == "not_confident"
    assert r["citation_check"]["ok"] is False
    assert r["citation_check"]["numeric_sink"] is True
    assert r["examiner"]["citation_specific"] is False
    assert r["verified"] is False


def test_claim_level_labels_and_display_region(monkeypatch, tmp_path):
    import json

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_pdf(raw)
    index = tmp_path / "index"
    run_ingest(raw, index, dpi=150, embedder="fake", skip_ocr=True,
               qdrant_path=str(index / "qdrant"))
    con = connect_sqlite(index / "meta.db")
    doc_id, page = con.execute("SELECT doc_id, page FROM pages LIMIT 1").fetchone()
    sents, _ = sentences_from_page(con, doc_id, page)
    target = next(s for s in sents if "42 million" in s.text)

    def fake_run(reader, images, prompt, n_samples=3, temperature=0.0, max_claims=5):
        return [ReaderSample(sample_id=0, answer="42 million",
                             quote="Aurora revenue was 42 million",
                             sentence_ids=[target.sent_id],
                             claims=[Claim(text="Aurora revenue was 42 million")])], 1

    class FakeExaminer:
        def generate(self, images, prompt, temperature, n):
            assert "claims" in prompt
            return [json.dumps({
                "quote_found": True,
                "quote_seen": "Aurora revenue was 42 million",
                "answer_supported": "yes",
                "answers_the_question": True,
                "citation_specific": True,
                "region_supports": True,
                "problem": "",
                "claims": [{
                    "index": 0, "label": "supported",
                    "citation_specific": True, "region_supports": True,
                    "quote_found": True,
                    "quote_seen": "Aurora revenue was 42 million",
                    "problem": "",
                }],
            })]

    monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)
    monkeypatch.setattr(client_mod, "create_reader", lambda *a, **k: FakeExaminer())

    r = answer_question("What was the Aurora revenue?", str(index),
                        str(tmp_path / "work"), reader_mode="live",
                        n_samples=1, grounding="span", gate="none",
                        doc_id=doc_id)

    assert r["state"] == "verified", r["state_reason"]
    assert r["claims"][0]["label"] == "supported"
    assert r["claim_regions"][0]["label"] == "supported"
    assert r["claim_level"]["ran"] is True
    assert r["claim_level"]["n_supported"] == 1
    assert r["evidence"]["display_source"] == "claim_region"
