"""UI server tests: keys/status/upload/ask over HTTP on an ephemeral port.

Run 8: the /api/ask happy path now goes through the full verifier (quote check
+ answer-support veto), so the stubbed reader cites a real sentence of the
uploaded page and quotes it verbatim; the assertions then require the answer
to be text-verified (tier "text"), not merely gate-accepted.
"""
import base64
import json
import threading
import urllib.request

import pymupdf

import vvrag.reader.client as client_mod
from vvrag.ingest.index import connect_sqlite
from vvrag.ingest.sentences import sentences_from_page
from vvrag.schemas import ReaderSample
from vvrag.ui import app


def _post(port, path, obj):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _pdf_bytes():
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 40), "Quarterly report deckUI page 0.")
        page.insert_text((100, 200), "The Aurora revenue was 42 million in this quarter.")
        return doc.tobytes()


def test_ui_end_to_end(monkeypatch, tmp_path):
    app.INDEX_DIR = str(tmp_path / "uindex")
    work = str(tmp_path / "uwork")
    srv = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    srv.index_dir = app.INDEX_DIR
    srv.work_dir = work
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status").read())
        assert st["docs"] == 0 and st["sdk"]["gemini"] is True
        m = _post(port, "/api/upload", {
            "files": [{"name": "deckUI.pdf",
                       "b64": base64.b64encode(_pdf_bytes()).decode()}],
            "skip_ocr": True})
        assert m["stats"]["pages"] == 1, m

        con = connect_sqlite(tmp_path / "uindex" / "meta.db")
        doc_id, page = con.execute(
            "SELECT doc_id, page FROM pages LIMIT 1").fetchone()
        sents, _ = sentences_from_page(con, doc_id, page)
        target = next(s for s in sents if "42 million" in s.text)

        def fake_run(reader, images, prompt, n_samples=3, temperature=0.0,
                     max_claims=5):
            return [ReaderSample(sample_id=0, answer="42 million",
                                 quote=target.text,
                                 sentence_ids=[target.sent_id],
                                 claims=[])], 1

        monkeypatch.setattr(client_mod, "run_reader_samples", fake_run)
        monkeypatch.setattr(client_mod, "create_reader", lambda *a, **k: object())

        r = _post(port, "/api/ask", {"question": "What is Aurora revenue?",
                                     "reader_mode": "mock", "n_samples": 2,
                                     "grounding": "span", "gate": "fixed",
                                     "fixed_tau": 0.0})
        assert r["answer"] == "42 million" and r["abstained"] is False, r
        assert r["verified"] is True and r["tier"] == "text", r
        assert r["evidence"]["localize"]["quote_ok"] is True
        assert r["evidence"]["sentence_ids"] == [target.sent_id]
        assert "42 million" in r["evidence"]["crop_text"], r["evidence"]["crop_text"]
        assert r["evidence"]["boxes"]
        media = urllib.request.urlopen(
            f"http://127.0.0.1:{port}" + r["evidence"]["crop_url"]).read()
        assert len(media) > 1000
    finally:
        srv.shutdown()
        t.join(timeout=10)
