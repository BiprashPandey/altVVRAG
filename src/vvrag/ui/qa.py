"""Ad-hoc Q&A for the localhost UI (suggestions #1-#9 implemented).

Pipeline:
  retrieve (never abstain) -> sentence units -> reader cites {answer, quote,
  sentence_ids} -> deterministic quote check (retry once) -> statement NLI on
  the cited sentences -> confidence score -> abstain gate -> draw boxes on
  the cited sentences only.

Retrieval is keyword overlap over SQLite word boxes (the MaxSim query encoder
only runs on the GPU box). It NEVER abstains: a zero-overlap question falls
back to the pages in reading order and lets the verifier decide (#3).

Grounding: the heatmap/keyword region is only a retrieval PRIOR used to build
the reader crop. The FINAL evidence boxes come from the cited sentences the
reader returned and the verifier approved (#1, #6). Boxes are emitted both in
pixels and as page fractions (0-1) for a zoom-stable viewer (#9).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from vvrag.eval.gold_regions import derive_answer_words
from vvrag.ground.boxes import answer_to_line_boxes, grid_to_boxes, snap_boxes_to_word_lines
from vvrag.ground.crop import crop_union, overlay_heatmap
from vvrag.ground.heatmap import build_heatmap, heat_concentration
from vvrag.ground.text_in_box import extract_crop_text, group_lines
from vvrag.ingest.embed import load_embedding
from vvrag.ingest.sentences import box_frac, sentences_from_page
from vvrag.logging_utils import get_logger
from vvrag.reader.claims import normalize_answer, validate_sentence_ids
from vvrag.schemas import WordBox

log = get_logger(__name__)

STOPWORDS = frozenset(
    "what which who whom whose when where why how is are was were be been being "
    "do does did done have has had having a an the and or of on in to for with "
    "by from as at it its this that these those then than so such no not nor "
    "please tell give show find answer question page document".split()
)

_nli = None


def get_nli():
    global _nli
    if _nli is None:
        from vvrag.verify.nli import NLIEntailer

        _nli = NLIEntailer()
        _nli.load()
    return _nli


def content_tokens(question: str) -> list[str]:
    return [t for t in normalize_answer(question).split()
            if t not in STOPWORDS and len(t) > 2]


def page_words(con: sqlite3.Connection, doc_id: str, page: int) -> list[WordBox]:
    rows = con.execute(
        "SELECT text,x0,y0,x1,y1,block_id,line_id FROM word_boxes WHERE doc_id=? AND page=?",
        (doc_id, page)).fetchall()
    return [WordBox(doc_id, page, r[0], (r[1], r[2], r[3], r[4]), r[5], r[6]) for r in rows]


def keyword_rank(question: str, con: sqlite3.Connection,
                 limit: int = 5, doc_id: str | None = None,
                 ) -> list[tuple[str, int, float, list[WordBox]]]:
    """Rank pages by keyword overlap. Returns (doc, page, score, words).

    NEVER returns an empty list when the index has pages: zero-overlap pages
    are ranked 0.0 and returned in reading order (suggestion #3: retrieval
    must not abstain; abstention is the verifier's job).
    `doc_id` optionally restricts ranking to one document.
    """
    toks = set(content_tokens(question))
    if not toks:
        toks = set(normalize_answer(question).split())
    if doc_id is None:
        pages = con.execute("SELECT doc_id, page FROM pages ORDER BY doc_id, page").fetchall()
    else:
        pages = con.execute("SELECT doc_id, page FROM pages WHERE doc_id=? ORDER BY page",
                            (doc_id,)).fetchall()
    ranked = []
    for d, page in pages:
        words = page_words(con, d, page)
        hit = sum(1 for w in words if normalize_answer(w.text) in toks)
        ranked.append((d, page, hit / max(1, len(toks)), words))
    # stable: by score desc, ties keep reading order (Python sort is stable)
    ranked.sort(key=lambda r: r[2], reverse=True)
    return ranked[:limit]


def closest_passages(question: str, sentences: list, k: int = 3) -> list[dict]:
    """Best-matching sentence units for the abstain message (#9)."""
    toks = set(content_tokens(question))
    if not toks:
        toks = set(normalize_answer(question).split())
    scored = []
    for s in sentences:
        st = set(normalize_answer(s.text).split())
        score = len(toks & st) / max(1, len(toks))
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [{"sent_id": s.sent_id, "text": s.text, "score": round(sc, 3),
             "box": list(s.box)} for sc, s in scored[:k]]


def span_box(question: str, words: list[WordBox]) -> list[tuple[int, int, int, int]]:
    """Evidence region from keyword hits: one tight box per hit line (#7).

    Returns a per-line list (possibly empty), never one big multi-line union.
    """
    toks = set(content_tokens(question))
    hits = [w for w in words if normalize_answer(w.text) in toks]
    if not hits:
        # fuzzy fallback: reuse answer-localization on the longest keywords
        toks_sorted = sorted(toks, key=len, reverse=True)
        for t in toks_sorted[:3]:
            span, score = derive_answer_words(t, words)
            if span:
                return answer_to_line_boxes(span)
        return []
    hit_ids = {id(w) for w in hits}
    hit_lines = [line for line in group_lines(words)
                 if any(id(w) in hit_ids for w in line)]
    hit_lines.sort(key=lambda ln: sum(1 for w in ln if id(w) in hit_ids), reverse=True)
    hit_lines = hit_lines[:3]
    boxes: list[tuple[int, int, int, int]] = []
    for line in hit_lines:
        boxes.append((min(w.box[0] for w in line), min(w.box[1] for w in line),
                      max(w.box[2] for w in line), max(w.box[3] for w in line)))
    return boxes


def heatmap_box_from_span(question: str, words: list[WordBox], page_vecs: np.ndarray,
                          grid_h: int, grid_w: int, page_w: int, page_h: int,
                          mode: str = "max",
                          query_text: str | None = None,
                          query_encoder: str = "auto",
                          ) -> tuple[list[tuple], float, np.ndarray] | tuple[None, None, None]:
    """Heatmap region prior (retrieval only; #6): real ColQwen vectors when
    available, else keyword-seeded pseudo-query. Output is snapped to OCR word
    lines; the FINAL boxes come from the cited sentences, never the heat blob.
    """
    q: np.ndarray | None = None
    labels: list[str] | None = None
    if query_encoder in ("colqwen", "real"):
        from vvrag.retrieve.query_embed import create_query_embedder

        enc = create_query_embedder("colqwen")
        enc.load()
        qvecs, labels = enc.embed_query(query_text or question)
        if qvecs.shape[1] != page_vecs.shape[1]:
            raise RuntimeError(f"query dim {qvecs.shape[1]} != index dim {page_vecs.shape[1]}")
        q = qvecs.astype(np.float32)
    if q is None:
        toks = set(content_tokens(question))
        hits = [w for w in words if normalize_answer(w.text) in toks]
        if not hits:
            return None, None, None
        cx = np.mean([(w.box[0] + w.box[2]) / 2 for w in hits])
        cy = np.mean([(w.box[1] + w.box[3]) / 2 for w in hits])
        gc = int(cx / page_w * grid_w)
        gr = int(cy / page_h * grid_h)
        cells = [(r, c) for r in (gr - 1, gr, gr + 1) for c in (gc - 1, gc, gc + 1)
                 if 0 <= r < grid_h and 0 <= c < grid_w] or [(gr, gc)]
        idx = [r * grid_w + c for r, c in cells]
        rng = np.random.default_rng(0)
        q = page_vecs[idx] + 0.05 * rng.standard_normal((len(idx), page_vecs.shape[1])).astype(np.float32)
        labels = [f"kw{i}" for i in range(len(idx))]
    _, norm = build_heatmap(q, labels or ["q"], page_vecs, grid_h, grid_w, mode=mode)
    boxes, mask = grid_to_boxes(norm, page_w, page_h)
    heat = heat_concentration(norm, mask)
    _, line_boxes = snap_boxes_to_word_lines(boxes, words)
    if line_boxes:
        return line_boxes, heat, norm
    return boxes, heat, norm


def localize_answer_to_boxes(
    answer: str, words: list[WordBox],
) -> tuple[list[WordBox], list[tuple[int, int, int, int]], float, bool]:
    """Fallback answer-text localization (used when the reader cites nothing).

    Returns (cover_words, line_boxes, match_score, evidence_valid).
    UNANSWERABLE -> ([], [], 0.0, False) so the gate auto-abstains.
    """
    if normalize_answer(answer) == "unanswerable" or not normalize_answer(answer):
        return [], [], 0.0, False
    span, score = derive_answer_words(answer, words)
    if not span:
        return [], [], 0.0, False
    return span, answer_to_line_boxes(span), float(score), bool(score >= 0.8)


def load_gate_artifacts(cfg) -> tuple:
    """Load the calibrated scorer + tau if the calibration script produced them."""
    scorer = None
    tau = None
    scorer_path = getattr(cfg.gate, "scorer_path", "")
    tau_path = getattr(cfg.gate, "tau_path", "")
    try:
        if scorer_path and Path(scorer_path).exists():
            from vvrag.gate.scorer import LogisticScorer

            scorer = LogisticScorer.load(scorer_path)
    except Exception as e:  # noqa: BLE001 - calibrated scorer is optional
        log.warning("could not load scorer %s: %s", scorer_path, e)
    try:
        if tau_path and Path(tau_path).exists():
            tau = float(json.loads(Path(tau_path).read_text())["tau"])
    except Exception as e:  # noqa: BLE001
        log.warning("could not load tau %s: %s", tau_path, e)
    return scorer, tau


def answer_question(question: str, index_dir: str, work_dir: str,
                    reader_mode: str = "mock", provider: str = "gemini", model: str = "",
                    n_samples: int = 3, temperature: float = 0.0,
                    grounding: str = "span", gate: str = "none",
                    fixed_tau: float = 0.5, query_encoder: str = "auto",
                    doc_id: str | None = None) -> dict:
    """Full ad-hoc pipeline -> JSON-able result dict."""
    t0 = time.time()
    index_dir, work_dir = Path(index_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    from vvrag.ingest.index import connect_sqlite

    con = connect_sqlite(index_dir / "meta.db")  # migrates old indexes (sent_id/sentences)

    from vvrag.config import AppConfig, load_config

    try:
        cfg: AppConfig = load_config("config.yaml")
    except Exception as e:  # noqa: BLE001 - config is optional for mock mode
        log.warning("config load failed, using defaults: %s", e)
        cfg = AppConfig()
    quote_threshold = float(getattr(cfg.verify, "quote_threshold", 0.8))

    ranked = keyword_rank(question, con, doc_id=doc_id)
    if not ranked:
        # genuinely empty index: the only honest abstention
        return {"answer": "UNANSWERABLE", "abstained": True,
                "abstain_reason": "index is empty", "closest_passages": [],
                "score": 0.0, "timings": {"total": time.time() - t0}}
    doc_id, page, rank_score, words = ranked[0]
    margin = (ranked[0][2] - ranked[1][2]) if len(ranked) > 1 else 0.0
    meta = con.execute(
        "SELECT width,height,grid_h,grid_w,image_path FROM pages WHERE doc_id=? AND page=?",
        (doc_id, page)).fetchone()
    W, H, gh, gw, img_path = meta

    # sentence units (#1) — built on the fly for indexes without them
    sentences, _ = sentences_from_page(con, doc_id, page)

    # region prior for the reader crop (heatmap/keyword); never the final box
    region_boxes, heat_conc, norm = None, 0.0, None
    if grounding == "heatmap":
        try:
            pe = load_embedding(index_dir / "embeddings", doc_id, page)
            region_boxes, heat_conc, norm = heatmap_box_from_span(
                question, words, pe.vectors, gh or pe.grid_h, gw or pe.grid_w, W, H,
                query_text=question, query_encoder=query_encoder)
        except Exception as e:
            log.warning("heatmap demo failed, falling back to span: %s", e)
    if not region_boxes:
        region_boxes = span_box(question, words) or [(0, 0, W, H)]
        heat_conc = 0.0

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"q{int(time.time()*1000)%100000000}")
    crop_path = str(work_dir / f"{safe}_crop.png")
    ov_path = str(work_dir / f"{safe}_overlay.png")
    crop_union(img_path, region_boxes, crop_path)
    if norm is not None:
        overlay_heatmap(img_path, norm, region_boxes, ov_path)
    else:
        overlay_heatmap(img_path, np.zeros((gh or 8, gw or 6), dtype=np.float32), region_boxes, ov_path)
    crop_text, has_text = extract_crop_text(words, region_boxes, crop_path)

    # reader with sentence citations (#1)
    t_r = time.time()
    if reader_mode == "mock":
        from vvrag.reader.client import MockReader

        reader: object = MockReader()
        images: list = []
    else:
        from vvrag.cache import DiskCache
        from vvrag.ratelimit import RateLimiter
        from vvrag.reader.client import create_reader

        model = model or cfg.reader.model
        reader = create_reader(provider, model, DiskCache(cfg.cache_dir),
                               RateLimiter(cfg.ratelimit.requests_per_minute),
                               fallback_provider=getattr(cfg.reader, "fallback_provider", ""),
                               fallback_model=getattr(cfg.reader, "fallback_model", ""))
        images = [img_path, crop_path]
    from vvrag.reader.client import run_reader_samples
    from vvrag.reader.prompts import build_prompt, build_sentence_prompt

    if reader_mode == "mock":
        prompt = build_prompt(question, crop_text if has_text else None, max_claims=5)
    else:
        prompt = build_sentence_prompt(question, sentences,
                                       crop_text if has_text else None, max_claims=5)
    samples, api_calls = run_reader_samples(reader, images, prompt, n_samples,
                                            temperature if n_samples > 1 else 0.0)
    t_r = time.time() - t_r

    # deterministic quote check (#4): quote must appear in cited sentences
    from vvrag.verify.quote_check import build_quote_retry_prompt, citation_texts, quote_match_score

    ocr_norm = bool(getattr(cfg.verify, "ocr_normalize", True))

    def _derive(sample):
        a = sample.answer
        una = normalize_answer(a) == "unanswerable"
        ids = list(sample.sentence_ids or [])
        texts, valid = citation_texts(ids, sentences)
        qs = (quote_match_score(sample.quote, texts, ocr_normalize=ocr_norm)
              if texts else 0.0)
        return a, una, ids, texts, valid, qs

    (answer, is_unanswerable, cited_ids, cited_texts,
     citations_valid, qscore) = _derive(samples[0])
    quote_ok = citations_valid and qscore >= quote_threshold
    max_retries = int(getattr(cfg.reader, "max_answer_retries", 1))

    # (#7) reduce false abstentions: reader said UNANSWERABLE but the page has
    # close passages -> one corrective retry showing those passages verbatim.
    if (is_unanswerable and reader_mode != "mock" and sentences
            and bool(getattr(cfg.reader, "retry_on_unanswerable", True))
            and max_retries > 0):
        from vvrag.reader.prompts import build_passage_retry_prompt

        cand = closest_passages(question, sentences, k=5)
        top = cand[0]["score"] if cand else 0.0
        if top >= float(getattr(cfg.reader, "unanswerable_retry_overlap", 0.2)):
            retry_prompt = build_passage_retry_prompt(prompt, cand, question)
            retry_samples, extra = run_reader_samples(reader, images, retry_prompt, 1, 0.0)
            api_calls += extra
            (r_answer, r_una, r_ids, r_texts,
             r_valid, r_qscore) = _derive(retry_samples[0])
            if not r_una and r_valid and r_qscore >= quote_threshold:
                samples[0] = retry_samples[0]
                answer, is_unanswerable = r_answer, r_una
                cited_ids, cited_texts = r_ids, r_texts
                citations_valid, qscore, quote_ok = r_valid, r_qscore, True

    if not is_unanswerable and not quote_ok and reader_mode != "mock" and sentences:
        # retry once with a corrective prompt (#4)
        retry_prompt = build_quote_retry_prompt(prompt, samples[0].quote)
        retry_samples, extra = run_reader_samples(reader, images, retry_prompt, 1, 0.0)
        api_calls += extra
        r_answer = retry_samples[0].answer
        r_ids = list(retry_samples[0].sentence_ids or [])
        r_texts, r_valid = citation_texts(r_ids, sentences)
        r_qscore = (quote_match_score(retry_samples[0].quote, r_texts, ocr_normalize=ocr_norm)
                    if r_texts else 0.0)
        if r_valid and r_qscore >= quote_threshold and normalize_answer(r_answer) != "unanswerable":
            samples[0] = retry_samples[0]
            answer = r_answer
            cited_ids, cited_texts = r_ids, r_texts
            citations_valid, qscore, quote_ok = True, r_qscore, True
        else:
            # keep the original answer but mark evidence unverified
            qscore = max(qscore, r_qscore)
            quote_ok = False

    # statement NLI on the CITED sentences (#4)
    from vvrag.verify.consistency import agreement
    from vvrag.verify.features import build_features
    from vvrag.verify.nli import aggregate_entailment, entail_statement

    agr, _ = agreement(samples)
    entail_min: float | None = None
    claim_probs: list[float] = []
    if not is_unanswerable:
        premise = cited_texts if cited_texts else ([crop_text] if has_text else [])
        if premise:
            try:
                nli = get_nli()
                entail_min = entail_statement(nli, premise, question, answer)
            except Exception as e:
                log.warning("statement NLI failed: %s", e)
        if has_text:
            try:
                nli = get_nli()
                for c in samples[0].claims:
                    claim_probs.extend(nli.entail_probs(crop_text, [c.text]))
            except Exception as e:
                log.warning("claim NLI failed: %s", e)
    if claim_probs:
        cm, _ = aggregate_entailment(claim_probs)
        entail_min = cm if entail_min is None else min(entail_min, cm)

    # final evidence boxes = cited sentences only (#1, #4, #6)
    cited_sentences = []
    final_boxes: list[tuple[int, int, int, int]] = []
    final_frac: list[tuple[float, float, float, float]] = []
    loc_score = 0.0
    loc_words: list[WordBox] = []
    loc_source = "none"
    evidence_valid = False
    if not is_unanswerable:
        if quote_ok and cited_texts:
            from vvrag.ground.boxes import refine_box_to_quote

            by_id = {s.sent_id: s for s in sentences}
            quote_text = samples[0].quote or ""
            for sid in dict.fromkeys(int(x) for x in cited_ids):
                s = by_id.get(sid)
                if s is None:
                    continue
                # P4/E: tighten the cited-sentence box to the quote's words so a
                # wide table row collapses to the cell/value the quote names.
                box = refine_box_to_quote(words, quote_text, s.box)
                cited_sentences.append({
                    "sent_id": s.sent_id, "text": s.text,
                    "box": list(box), "box_frac": list(box_frac(box, W, H)),
                })
                final_boxes.append(box)
                final_frac.append(box_frac(box, W, H))
            evidence_valid = True
            loc_score = 1.0
            loc_source = "sentence_ids"
        else:
            # fallback: answer-text localization (fuzzy)
            span, score = derive_answer_words(answer, words)
            if span:
                loc_words, final_boxes = span, answer_to_line_boxes(span)
                final_frac = [box_frac(b, W, H) for b in final_boxes]
                loc_score, evidence_valid = float(score), bool(score >= 0.8)
                loc_source = "fuzzy"
                cited_sentences = [{
                    "sent_id": -1, "text": " ".join(w.text for w in span),
                    "box": list(b), "box_frac": list(frac),
                } for b, frac in zip(final_boxes, final_frac)]

    if not final_boxes:
        final_boxes = region_boxes
        final_frac = [box_frac(b, W, H) for b in region_boxes]

    feats = build_features(agreement=agr, entail_min=entail_min, retrieval_margin=margin,
                           heat_concentration=heat_conc, n_claims=len(samples[0].claims),
                           crop_has_text=has_text,
                           localize_score=loc_score, evidence_valid=evidence_valid,
                           quote_match=qscore)
    from vvrag.gate.fixed import apply_fixed
    from vvrag.gate.scorer import features_to_matrix, hand_weighted

    # confidence: calibrated scorer when available, else hand-weighted (#5)
    scorer, cal_tau = load_gate_artifacts(cfg)
    score = hand_weighted(agr, entail_min, heat_conc, loc_score, evidence_valid, qscore)
    if scorer is not None:
        try:
            score = float(scorer.predict_score(features_to_matrix([feats]))[0])
        except Exception as e:  # noqa: BLE001
            log.warning("learned scorer failed, using hand-weighted: %s", e)

    tau = fixed_tau
    abstain_reason = ""
    if is_unanswerable:
        abstained = True
        abstain_reason = "reader returned UNANSWERABLE"
    elif gate in ("calibrated", "conformal") and cal_tau is not None:
        tau = cal_tau
        abstained = not apply_fixed(score, tau)
        abstain_reason = "" if not abstained else f"score {score:.3f} < tau {tau}"
        if abstained and evidence_valid and qscore >= 0.9:
            abstained = False
            abstain_reason = "verifier rescue"
    elif gate == "fixed":
        abstained = not apply_fixed(score, fixed_tau)
        abstain_reason = "" if not abstained else f"score {score:.3f} < tau {fixed_tau}"
        if abstained and evidence_valid and qscore >= 0.9:
            abstained = False
            abstain_reason = "verifier rescue"
    elif gate == "none":
        abstained = False
        # a failed quote check still blocks an answer (deterministic verifier)
        if not is_unanswerable and not quote_ok and cited_ids:
            abstained = True
            abstain_reason = "quote check failed"
    else:
        abstained = False

    passages = closest_passages(question, sentences) if abstained else []

    return {
        "answer": answer,
        "abstained": abstained,
        "abstain_reason": abstain_reason,
        "not_found": abstained,
        "score": round(score, 4),
        "tau": tau if gate in ("fixed", "calibrated", "conformal") else 0.0,
        "retrieval": [{"doc_id": d, "page": p, "keyword_score": round(s, 3)}
                      for d, p, s, _ in ranked],
        "closest_passages": passages,
        "evidence": {"doc_id": doc_id, "page": page,
                     "boxes": [list(b) for b in final_boxes],
                     "box_frac": [list(b) for b in final_frac],
                     "sentence_ids": cited_ids if quote_ok else [],
                     "region_boxes": [list(b) for b in region_boxes],
                     "crop_text": crop_text, "crop_has_text": has_text,
                     "heat_concentration": round(heat_conc, 4),
                     "page_width": W, "page_height": H,
                     "page_url": "/media/" + Path(img_path).name,
                     "localize": {"score": round(loc_score, 4), "valid": evidence_valid,
                                  "source": loc_source,
                                  "quote_score": round(qscore, 4),
                                  "quote_ok": quote_ok,
                                  "citations_valid": citations_valid,
                                  "n_words": len(loc_words),
                                  "words": [w.text for w in loc_words]},
                     "crop_url": "/media/" + Path(crop_path).name,
                     "overlay_url": "/media/" + Path(ov_path).name},
        "cited_sentences": cited_sentences,
        "claims": [c.to_dict() for c in samples[0].claims],
        "samples": [{"answer": s.answer, "quote": s.quote,
                     "sentence_ids": s.sentence_ids} for s in samples],
        "features": feats.to_dict(),
        "api_calls": api_calls,
        "timings": {"total": round(time.time() - t0, 3), "reader": round(t_r, 3)},
    }
