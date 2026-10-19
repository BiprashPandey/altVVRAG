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
                 limit: int | None = 5, doc_id: str | None = None,
                 ) -> list[tuple[str, int, float, list[WordBox]]]:
    """Rank pages by keyword overlap. Returns (doc, page, score, words).

    NEVER returns an empty list when the index has pages: zero-overlap pages
    are ranked 0.0 and returned in reading order (suggestion #3: retrieval
    must not abstain; abstention is the verifier's job).
    `doc_id` optionally restricts ranking to one document.
    `limit=None` returns the full ranking (needed by RRF fusion).
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
    return ranked if limit is None else ranked[:limit]


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
                          query_vecs: np.ndarray | None = None,
                          ) -> tuple[list[tuple], float, np.ndarray] | tuple[None, None, None]:
    """Heatmap region prior (retrieval only; #6): real ColQwen vectors when
    available, else keyword-seeded pseudo-query. Output is snapped to OCR word
    lines; the FINAL boxes come from the cited sentences, never the heat blob.

    `query_vecs` (Phase 1): precomputed ColQwen query multivector (e.g. from
    the query cache) used directly -- real heatmap without loading the model.
    """
    q: np.ndarray | None = None
    labels: list[str] | None = None
    if query_vecs is not None:
        q = np.asarray(query_vecs, dtype=np.float32)
        if q.shape[1] != page_vecs.shape[1]:
            raise RuntimeError(f"query dim {q.shape[1]} != index dim {page_vecs.shape[1]}")
        labels = [f"q{i}" for i in range(q.shape[0])]
    elif query_encoder in ("colqwen", "real"):
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


def _visual_check_verdict(answer: str, cf_raw: str) -> tuple[bool, object, str]:
    """Parse a cross-family visual-check reply into (match, supports, read).

    When the reply carries an explicit "supports" verdict it is authoritative
    (truthy => match), except that a contradictory UNANSWERABLE read is never
    accepted. Without a verdict field, match = normalized string/number
    containment between the checker's re-read answer and the claimed answer.
    """
    cf_ans, cf_sup = cf_raw, None
    try:
        cf_obj = json.loads(cf_raw[cf_raw.find("{"):cf_raw.rfind("}") + 1])
        cf_ans = str(cf_obj.get("answer", ""))
        cf_sup = cf_obj.get("supports")
    except Exception:  # noqa: BLE001 - fall back to raw text
        pass
    ncf = normalize_answer(cf_ans)
    if cf_sup is not None:
        sup = cf_sup if isinstance(cf_sup, bool) else (
            str(cf_sup).strip().lower() == "true")
        # explicit verdict is authoritative; a contradictory UNANSWERABLE read
        # is never accepted
        return bool(sup and "unanswerable" not in ncf), cf_sup, cf_ans
    na = normalize_answer(answer)
    str_match = bool(na and ncf and "unanswerable" not in ncf
                     and (na == ncf or na in ncf or ncf in na))
    if not str_match and ncf and "unanswerable" not in ncf:
        # number-normalized containment ("12.5 bn" ~ "12.5", "1,234" ~ "1.234")
        try:
            from vvrag.verify.support import normalize_numbers

            nn_a = normalize_numbers(answer)
            nn_cf = normalize_numbers(cf_ans)
            str_match = bool(nn_a and nn_cf and nn_a != nn_cf
                             and (nn_a in nn_cf or nn_cf in nn_a))
        except Exception:  # noqa: BLE001
            pass
    return bool(str_match), cf_sup, cf_ans


def answer_question(question: str, index_dir: str, work_dir: str,
                    reader_mode: str = "mock", provider: str = "gemini", model: str = "",
                    n_samples: int = 3, temperature: float = 0.0,
                    grounding: str = "span", gate: str = "none",
                    fixed_tau: float = 0.5, query_encoder: str = "auto",
                    doc_id: str | None = None,
                    retrieval_mode: str | None = None) -> dict:
    """Full ad-hoc pipeline -> JSON-able result dict.

    `retrieval_mode`: "keyword" (legacy) | "visual" | "hybrid"; None reads
    config `retriever.retrieval_mode`. Visual modes need a cached query
    multivector (scripts/precompute_query_vectors.py); cache miss -> keyword
    fallback recorded in `retrieval.mode_used`.
    """
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

    # ---------------- retrieval (Phase 1: keyword | visual | hybrid) ------
    cfg_ret = getattr(cfg, "retriever", None)
    mode = (retrieval_mode or getattr(cfg_ret, "retrieval_mode", "keyword")
            or "keyword").lower()
    kw_full = keyword_rank(question, con, limit=None, doc_id=doc_id)
    if not kw_full:
        # genuinely empty index: the only honest abstention
        return {"answer": "UNANSWERABLE", "abstained": True,
                "abstain_reason": "index is empty", "closest_passages": [],
                "score": 0.0, "timings": {"total": time.time() - t0}}
    words_map = {(d, p): w for d, p, _s, w in kw_full}
    mode_used, retrieval_note = "keyword", ""
    vis_scores = None
    qvecs = None
    if mode in ("visual", "hybrid"):
        from vvrag.retrieve.query_vectors import QueryVectorStore

        store = QueryVectorStore(getattr(cfg_ret, "query_vectors_dir", "data/query_cache"))
        hit = store.get(question)
        if hit is None:
            retrieval_note = ("no cached query vector; keyword fallback "
                              "(run scripts/precompute_query_vectors.py)")
            ranked = kw_full[:5]
        else:
            qvecs = hit[0]
            from vvrag.retrieve.fusion import (deck_aggregate, rank_order,
                                               rrf_fuse, visual_page_scores)

            page_list = [(d, p) for d, p, _s, _w in kw_full]
            vis_scores = visual_page_scores(index_dir, page_list, qvecs)
            vis_order = rank_order(vis_scores)
            if mode == "visual":
                order = vis_order
                rank_scores = vis_scores
            else:
                fused = rrf_fuse([page_list, vis_order],
                                 k=int(getattr(cfg_ret, "rrf_k", 60)))
                order = [(d, p) for d, p, _s in fused]
                rank_scores = {(d, p): s for d, p, s in fused}
            ranked = [(d, p, rank_scores[(d, p)], words_map[(d, p)])
                      for d, p in order if (d, p) in words_map][:5]
            mode_used = mode
    else:
        ranked = kw_full[:5]

    def _top_pages(scores, n: int = 3):
        if not scores:
            return []
        return [{"doc_id": d, "page": p, "score": round(float(s), 4)}
                for (d, p), s in sorted(scores.items(), key=lambda kv: kv[1],
                                        reverse=True)[:n]]

    retrieval_info = {
        "mode_requested": mode,
        "mode_used": mode_used,
        "note": retrieval_note,
        "keyword_top5": [{"doc_id": d, "page": p, "score": round(float(s), 4)}
                         for d, p, s, _w in kw_full[:5]],
        "visual_top5": _top_pages(vis_scores, 5),
        "deck_scores": (sorted(deck_aggregate(vis_scores).items(),
                               key=lambda kv: kv[1], reverse=True)[:3]
                        if vis_scores else []),
    }

    # ---------------- page state + per-page views (Phase 2) --------------
    primary_key = (ranked[0][0], ranked[0][1])
    margin = (ranked[0][2] - ranked[1][2]) if len(ranked) > 1 else 0.0
    max_pages = max(1, int(getattr(cfg.reader, "max_pages", 3)))
    ctx_keys = [(d, p) for d, p, _s, _w in ranked[:max_pages]]
    if primary_key not in ctx_keys:
        ctx_keys.insert(0, primary_key)

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"q{int(time.time()*1000)%100000000}")
    _page_state: dict = {}

    def _get_state(key):
        d, p = key
        if key not in _page_state:
            meta = con.execute(
                "SELECT width,height,grid_h,grid_w,image_path FROM pages "
                "WHERE doc_id=? AND page=?", (d, p)).fetchone()
            if meta is None:
                _page_state[key] = None
                return None
            W_, H_, gh_, gw_, img_path_ = meta
            _page_state[key] = {
                "doc_id": d, "page": p,
                "words": words_map.get(key) or page_words(con, d, p),
                "sentences": sentences_from_page(con, d, p)[0],
                "W": W_, "H": H_, "gh": gh_, "gw": gw_, "img_path": img_path_,
            }
        return _page_state[key]

    def _prep_view(key, tag):
        st = _get_state(key)
        if st is None:
            return None
        region_boxes, heat_conc, norm = None, 0.0, None
        hm_src = "keyword-seeded"
        if grounding == "heatmap":
            try:
                pe = load_embedding(index_dir / "embeddings", st["doc_id"], st["page"])
                if qvecs is not None:
                    hm_src = "colqwen-cached"
                region_boxes, heat_conc, norm = heatmap_box_from_span(
                    question, st["words"], pe.vectors, st["gh"] or pe.grid_h,
                    st["gw"] or pe.grid_w, st["W"], st["H"],
                    query_text=question, query_encoder=query_encoder, query_vecs=qvecs)
            except Exception as e:
                log.warning("heatmap demo failed, falling back to span: %s", e)
        if not region_boxes:
            region_boxes = span_box(question, st["words"]) or [(0, 0, st["W"], st["H"])]
            heat_conc = 0.0
        crop_path = str(work_dir / f"{safe}_{tag}_crop.png")
        ov_path = str(work_dir / f"{safe}_{tag}_overlay.png")
        crop_union(st["img_path"], region_boxes, crop_path)
        if norm is not None:
            overlay_heatmap(st["img_path"], norm, region_boxes, ov_path)
        else:
            overlay_heatmap(st["img_path"], np.zeros((st["gh"] or 8, st["gw"] or 6),
                                                      dtype=np.float32),
                            region_boxes, ov_path)
        crop_text, has_text = extract_crop_text(st["words"], region_boxes, crop_path)
        view = dict(st)
        view.update(region_boxes=region_boxes, heat_conc=heat_conc, norm=norm,
                    crop_path=crop_path, ov_path=ov_path,
                    crop_text=crop_text, has_text=has_text, heatmap_query_source=hm_src)
        return view

    def _blocks_for(pri_key, keys):
        """Prompt blocks (primary first) + label -> (doc,page) map."""
        ordered = [pri_key] + [k for k in keys if k != pri_key]
        blocks, by = [], {}
        for i, key in enumerate(ordered):
            st = _get_state(key)
            if st is None:
                continue
            by[i] = key
            blocks.append({"label": i, "page": st["page"], "doc_id": st["doc_id"],
                           "sentences": st["sentences"],
                           "is_primary": key == pri_key})
        return blocks, by

    view = _prep_view(primary_key, "p0")
    blocks, active_by_label = _blocks_for(primary_key, ctx_keys)
    retrieval_info["heatmap_query"] = view["heatmap_query_source"]
    retrieval_info["pages_context"] = [{"doc_id": d, "page": p} for d, p in ctx_keys]

    doc_id, page = view["doc_id"], view["page"]
    words, sentences = view["words"], view["sentences"]
    W, H, gh, gw, img_path = view["W"], view["H"], view["gh"], view["gw"], view["img_path"]
    region_boxes = view["region_boxes"]
    heat_conc, norm = view["heat_conc"], view["norm"]
    crop_path, ov_path = view["crop_path"], view["ov_path"]
    crop_text, has_text = view["crop_text"], view["has_text"]

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
    from vvrag.reader.prompts import (build_multipage_sentence_prompt,
                                      build_prompt, build_sentence_prompt)

    if reader_mode == "mock":
        prompt = build_prompt(question, crop_text if has_text else None, max_claims=5)
    elif len(blocks) > 1:
        prompt = build_multipage_sentence_prompt(
            question, blocks, crop_text if has_text else None, max_claims=5)
    else:
        prompt = build_sentence_prompt(question, sentences,
                                       crop_text if has_text else None, max_claims=5)
    samples, api_calls = run_reader_samples(reader, images, prompt, n_samples,
                                            temperature if n_samples > 1 else 0.0)
    t_r = time.time() - t_r

    # deterministic quote check (#4): quote must appear in cited sentences
    from vvrag.verify.quote_check import build_quote_retry_prompt, quote_match_score

    ocr_norm = bool(getattr(cfg.verify, "ocr_normalize", True))

    def _derive(sample, by_label=None, default_key=None):
        """Parse a sample into (answer, unanswerable, cited refs, texts, valid, qscore, resolution).

        refs are (doc_id, page, sent_id) triples: prompt-block labels come from
        `by_label` (attempt-local); legacy ints cite the primary page.
        `resolution` records how the reader's raw citations resolved:
        {"status": "none|ok|partial|unresolved", "n_refs", "n_resolved",
         "unresolved": [[label, sent_id], ...]} -- unresolved refs are
        hallucinated ids or ids outside the shown pages (Run 8: no silent
        rebinding to the primary page).
        """
        a = sample.answer
        una = normalize_answer(a) == "unanswerable"
        refs = list(sample.sentence_refs or [])
        if not refs:
            refs = [[None, int(x)] for x in (sample.sentence_ids or [])]
        by_label = active_by_label if by_label is None else by_label
        key0 = primary_key if default_key is None else default_key
        resolved: list[tuple[str, int, int]] = []
        unresolved: list[list] = []
        for r in refs:
            label, sid = r[0], int(r[1])
            if label is None:
                key = key0
            else:
                key = by_label.get(int(label))
                if key is None:  # defensive: reader echoed a real page number
                    key = next((k for k in ctx_keys if k[1] == int(label)), None)
                    if key is None:
                        unresolved.append([r[0], r[1]])
                        continue
            st = _get_state((key[0], key[1]))
            s = next((x for x in (st["sentences"] if st else [])
                      if x.sent_id == sid), None)
            if s is None:
                unresolved.append([r[0], r[1]])
                continue
            resolved.append((key[0], key[1], sid))
        texts: list[str] = []
        for d, p, sid in resolved:
            st = _get_state((d, p))
            s = next((x for x in (st["sentences"] if st else [])
                      if x.sent_id == sid), None)
            if s is not None:
                texts.append(s.text)
        qs = (quote_match_score(sample.quote, texts, ocr_normalize=ocr_norm)
              if texts else 0.0)
        if not refs:
            status = "none"
        elif not resolved:
            status = "unresolved"
        elif unresolved:
            status = "partial"
        else:
            status = "ok"
        resolution = {"status": status, "n_refs": len(refs),
                      "n_resolved": len(resolved),
                      "unresolved": unresolved[:5]}
        valid = bool(resolved and texts)
        return a, una, resolved, texts, valid, qs, resolution

    (answer, is_unanswerable, cited_refs, cited_texts,
     citations_valid, qscore, citation_info) = _derive(samples[0])
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
            (r_answer, r_una, r_refs, r_texts,
             r_valid, r_qscore, r_info) = _derive(retry_samples[0])
            if not r_una and r_valid and r_qscore >= quote_threshold:
                samples[0] = retry_samples[0]
                answer, is_unanswerable = r_answer, r_una
                cited_refs, cited_texts = r_refs, r_texts
                citations_valid, qscore, quote_ok = r_valid, r_qscore, True
                citation_info = r_info

    # Phase 2: honest multi-hop -- still UNANSWERABLE and other candidate
    # pages exist -> retry once with the next page promoted to primary (its
    # own image + crop attached; all pages' sentences in the prompt).
    if (is_unanswerable and reader_mode != "mock" and len(ctx_keys) > 1
            and bool(getattr(cfg.reader, "next_page_retry", True))):
        for alt_key in ctx_keys[1:]:
            alt_view = _prep_view(alt_key, "p1")
            if alt_view is None:
                continue
            alt_blocks, alt_by = _blocks_for(alt_key, ctx_keys)
            alt_crop_text = alt_view["crop_text"] if alt_view["has_text"] else None
            alt_prompt = (build_multipage_sentence_prompt(question, alt_blocks, alt_crop_text)
                          if len(alt_blocks) > 1 else
                          build_sentence_prompt(question, alt_view["sentences"], alt_crop_text))
            alt_images = [alt_view["img_path"], alt_view["crop_path"]]
            alt_samples, extra = run_reader_samples(reader, alt_images, alt_prompt, 1, 0.0)
            api_calls += extra
            (a_answer, a_una, a_refs, a_texts,
             a_valid, a_qscore, a_info) = _derive(alt_samples[0], by_label=alt_by,
                                                  default_key=alt_key)
            if not a_una:
                samples[0] = alt_samples[0]
                answer, is_unanswerable = a_answer, a_una
                cited_refs, cited_texts = a_refs, a_texts
                citations_valid, qscore = a_valid, a_qscore
                citation_info = a_info
                quote_ok = a_valid and a_qscore >= quote_threshold
                # adopt the promoted page as the active context
                view, primary_key = alt_view, alt_key
                blocks, active_by_label = alt_blocks, alt_by
                doc_id, page = view["doc_id"], view["page"]
                words, sentences = view["words"], view["sentences"]
                W, H = view["W"], view["H"]
                region_boxes, heat_conc, norm = (view["region_boxes"],
                                                 view["heat_conc"], view["norm"])
                crop_path, ov_path = view["crop_path"], view["ov_path"]
                crop_text, has_text = view["crop_text"], view["has_text"]
                prompt, images = alt_prompt, alt_images
                retrieval_info["primary_promoted"] = {"doc_id": doc_id, "page": page}
            break

    if not is_unanswerable and not quote_ok and reader_mode != "mock" and sentences:
        # retry once with a corrective prompt (#4)
        retry_prompt = build_quote_retry_prompt(prompt, samples[0].quote)
        retry_samples, extra = run_reader_samples(reader, images, retry_prompt, 1, 0.0)
        api_calls += extra
        (r_answer, r_una, r_refs, r_texts,
         r_valid, r_qscore, r_info) = _derive(retry_samples[0])
        if not r_una and r_valid and r_qscore >= quote_threshold:
            samples[0] = retry_samples[0]
            answer = r_answer
            cited_refs, cited_texts = r_refs, r_texts
            citations_valid, qscore, quote_ok = r_valid, r_qscore, True
            citation_info = r_info
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
    stmt_entail: float | None = None
    claim_probs: list[float] = []
    if not is_unanswerable:
        premise = cited_texts if cited_texts else ([crop_text] if has_text else [])
        if premise:
            try:
                nli = get_nli()
                stmt_entail = entail_statement(nli, premise, question, answer)
                entail_min = stmt_entail
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

    # Phase 3 veto: a verbatim quote alone is not enough -- the ANSWER must be
    # contained in (or entailed by) the cited text, else evidence is unverified.
    # Run 8: question-aware (answer type + comparison values) via qa_support;
    # the statement-level entailment feeds support, claim min stays the feature.
    require_support = bool(getattr(cfg.verify, "require_answer_support", False))
    support_threshold = float(getattr(cfg.verify, "support_entail_threshold", 0.5))
    type_check = bool(getattr(cfg.verify, "answer_type_check", True))
    comparison_check = bool(getattr(cfg.verify, "comparison_check", True))
    if not require_support:
        supported, support_score, support_method = True, 1.0, "disabled"
    elif not cited_texts:
        supported, support_score, support_method = False, 0.0, "no_citation"
    else:
        from vvrag.verify.support import qa_support

        supported, support_score, support_method = qa_support(
            question, answer, cited_texts, stmt_entail, support_threshold,
            type_check=type_check, comparison_check=comparison_check)
    verified = bool(quote_ok and supported and not is_unanswerable)

    def _unsupported_reason() -> str:
        if citation_info["status"] == "unresolved":
            ex = citation_info["unresolved"][0] if citation_info["unresolved"] else None
            return (f"citation could not be resolved on the retrieved pages "
                    f"({citation_info['n_refs']} ref(s)"
                    + (f", e.g. {ex}" if ex else "") + ")")
        if support_method == "type_mismatch":
            return "answer type does not match the question"
        if support_method == "missing_comparison_values":
            return "comparison values not both present in cited text"
        return f"answer not supported by cited text ({support_method})"

    # Phase 9 (Run 8): visual middle tier -- the reader answered from something
    # visible (chart/table/figure) that no sentence verifies. The answer is
    # accepted only with an independent confirmation: cross-family vision check
    # (different provider/model family) or, when that is unavailable, strong
    # agreement across the reader's own samples. Never for unanswerable rows.
    visual_ok = False
    visual_confirm = {"attempted": False, "method": "", "verdict": False,
                      "detail": ""}
    vtier = str(getattr(cfg.verify, "visual_tier", "off") or "off").lower()
    visual_min_agr = float(getattr(cfg.verify, "visual_min_agreement", 0.66))
    region_present = bool(samples[0].region and samples[0].region.get("box_frac"))
    # Only quote-grounding failures are rescuable visually: a wrong-type or
    # comparison-incomplete answer is a hard question-aware veto, not an OCR
    # proximity problem.
    quote_ground_failure = (not quote_ok
                            and support_method not in ("type_mismatch",
                                                       "missing_comparison_values",
                                                       "empty")
                            and citation_info["status"] in ("ok", "partial"))
    if (not is_unanswerable and not verified and reader_mode != "mock"
            and vtier in ("agreement", "cross_family")
            and (region_present or quote_ground_failure)):
        visual_confirm["attempted"] = True
        agree_ok = bool(agr >= visual_min_agr)
        # Check image: the reader's own region crop (padded) when it gave one,
        # else the full page. A chart/figure answer is only confirmable where
        # the reader says it saw it.
        check_img = img_path
        _r0 = samples[0].region or {}
        _f = _r0.get("box_frac")
        if _f:
            _rkey = primary_key
            if _r0.get("page") is not None:
                _rkey = next((k for k in ctx_keys
                              if k[1] == int(_r0["page"])), primary_key)
            _st = _get_state(_rkey)
            if _st is not None:
                try:
                    from PIL import Image as _PILImage

                    _im = _PILImage.open(_st["img_path"])
                    _w, _h = _im.size
                    _pad = 0.04
                    _box = (int(max(0.0, _f[0] - _pad) * _w),
                            int(max(0.0, _f[1] - _pad) * _h),
                            int(min(1.0, _f[2] + _pad) * _w),
                            int(min(1.0, _f[3] + _pad) * _h))
                    if _box[2] > _box[0] and _box[3] > _box[1]:
                        _cp = Path(str(crop_path)).with_name(
                            Path(str(crop_path)).stem + "_vcheck.png")
                        _im.crop(_box).save(_cp)
                        check_img = str(_cp)
                except Exception:  # noqa: BLE001 - fall back to full page
                    check_img = img_path
        if vtier == "cross_family":
            try:
                from vvrag.cache import DiskCache as _DiskCache
                from vvrag.ratelimit import RateLimiter as _RateLimiter
                from vvrag.reader.client import create_reader as _create_reader
                from vvrag.reader.prompts import build_visual_check_prompt

                cf_prov = getattr(cfg.verify, "cross_family_provider", "groq")
                cf_model = getattr(cfg.verify, "cross_family_model", "")
                cf_reader = _create_reader(
                    cf_prov, cf_model, _DiskCache(cfg.cache_dir),
                    _RateLimiter(cfg.ratelimit.requests_per_minute))
                cf_imgs = [check_img]
                if (crop_path and str(crop_path) != str(check_img)
                        and Path(str(crop_path)).exists()
                        and str(crop_path) not in cf_imgs):
                    cf_imgs.append(str(crop_path))
                cf_raw = cf_reader.generate(
                    cf_imgs, build_visual_check_prompt(question, answer), 0.0, 1)[0]
                api_calls += 1
                match, cf_sup, cf_ans = _visual_check_verdict(answer, cf_raw)
                visual_confirm.update(
                    method=f"cross_family:{cf_prov}/{cf_model}",
                    verdict=match,
                    detail=f"supports={cf_sup}; read={cf_ans[:140]}")
                visual_ok = match
            except Exception as e:  # noqa: BLE001 - cache/limiter/quota issues
                log.warning("cross-family visual check failed: %s", e)
                visual_confirm["detail"] = f"cross-family unavailable: {e}"
                if agree_ok:
                    visual_ok = True
                    visual_confirm.update(
                        method=f"agreement>={visual_min_agr:g}", verdict=True,
                        detail=f"cross-family unavailable; sample agreement {agr:.2f}")
        else:
            visual_ok = bool(agree_ok)
            visual_confirm.update(method="agreement", verdict=bool(agree_ok),
                                  detail=f"sample agreement {agr:.2f}")
    tier = ("text" if verified else "visual" if visual_ok else "none")

    # final evidence boxes = cited sentences only (#1, #4, #6; Phase 2 across pages)
    cited_sentences: list[dict] = []
    final_boxes: list[tuple[int, int, int, int]] = []
    final_frac: list[tuple[float, float, float, float]] = []
    pages_evidence: dict = {}
    loc_score = 0.0
    loc_words: list[WordBox] = []
    loc_source = "none"
    evidence_valid = False
    if not is_unanswerable:
        if verified and cited_texts:
            from vvrag.ground.boxes import refine_box_to_quote

            quote_text = samples[0].quote or ""
            for d, p, sid in dict.fromkeys(cited_refs):
                if sid < 0:  # Phase 3: fabricated/fuzzy ids are a hard fail
                    continue
                st = _get_state((d, p))
                if st is None:
                    continue
                s = next((x for x in st["sentences"] if x.sent_id == sid), None)
                if s is None:
                    continue
                # P4/E: tighten the cited-sentence box to the quote's words so a
                # wide table row collapses to the cell/value the quote names.
                box = refine_box_to_quote(st["words"], quote_text, s.box)
                frac = box_frac(box, st["W"], st["H"])
                entry = {"doc_id": d, "page": p, "sent_id": s.sent_id,
                         "text": s.text, "box": list(box), "box_frac": list(frac)}
                cited_sentences.append(entry)
                grp = pages_evidence.setdefault((d, p), {
                    "doc_id": d, "page": p, "boxes": [], "box_frac": [],
                    "sentence_ids": [], "citations": [],
                    "page_url": "/media/" + Path(st["img_path"]).name,
                })
                grp["boxes"].append(list(box))
                grp["box_frac"].append(list(frac))
                grp["sentence_ids"].append(s.sent_id)
                grp["citations"].append(entry)
                if (d, p) == (doc_id, page):
                    final_boxes.append(box)
                    final_frac.append(frac)
            evidence_valid = bool(cited_sentences)
            loc_score = 1.0 if evidence_valid else 0.0
            loc_source = "sentence_ids"
        elif visual_ok:
            # Phase 9: the confirmed visual region is the evidence (chart/table/
            # figure). Falls through to the retrieval region when the reader
            # gave no box_frac.
            r = samples[0].region or {}
            f = r.get("box_frac")
            st = None
            if f:
                rkey = primary_key
                if r.get("page") is not None:
                    rkey = next((k for k in ctx_keys if k[1] == int(r["page"])),
                                primary_key)
                st = _get_state(rkey)
            if st is not None and f:
                rbox = (int(f[0] * st["W"]), int(f[1] * st["H"]),
                        int(f[2] * st["W"]), int(f[3] * st["H"]))
                frac = box_frac(rbox, st["W"], st["H"])
                entry = {"doc_id": st["doc_id"], "page": st["page"], "sent_id": -1,
                         "text": samples[0].quote or "(visual region)",
                         "box": list(rbox), "box_frac": list(frac),
                         "visual": True}
                cited_sentences = [entry]
                grp = pages_evidence.setdefault((st["doc_id"], st["page"]), {
                    "doc_id": st["doc_id"], "page": st["page"], "boxes": [],
                    "box_frac": [], "sentence_ids": [], "citations": [],
                    "page_url": "/media/" + Path(st["img_path"]).name,
                })
                grp["boxes"].append(list(rbox))
                grp["box_frac"].append(list(frac))
                grp["citations"].append(entry)
                if (st["doc_id"], st["page"]) == (doc_id, page):
                    final_boxes.append(rbox)
                    final_frac.append(frac)
                evidence_valid = True
                loc_score = 1.0
                loc_source = "visual_region"
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

    # Phase 4 features: citation lands on the retrieved top-1 page; visual gap
    top1_key = (ranked[0][0], ranked[0][1])
    cited_is_top1 = (any((d, p) == top1_key for d, p, _s in cited_refs)
                     if cited_refs else False)
    if vis_scores:
        _vs = sorted(vis_scores.values(), reverse=True)
        visual_margin = float(_vs[0] - _vs[1]) if len(_vs) > 1 else 0.0
    else:
        visual_margin = 0.0

    # Phase 9 features: citation resolution + question-aware support
    from vvrag.verify.support import answer_type_match as _answer_type_match

    at_match = (_answer_type_match(question, answer)
                if not is_unanswerable else None)
    citation_resolved = {"none": 1.0, "ok": 1.0,
                         "partial": 0.5}.get(citation_info["status"], 0.0)
    if support_method.endswith("+type"):
        support_type = 1.0
    elif support_method == "disabled":
        support_type = 0.5
    else:
        support_type = 0.0

    feats = build_features(agreement=agr, entail_min=entail_min, retrieval_margin=margin,
                           heat_concentration=heat_conc, n_claims=len(samples[0].claims),
                           crop_has_text=has_text,
                           localize_score=loc_score, evidence_valid=evidence_valid,
                           quote_match=qscore,
                           answer_supported=supported,
                           cited_is_top1=cited_is_top1,
                           visual_margin=visual_margin,
                           citation_resolved=citation_resolved,
                           answer_type_match=at_match,
                           support_type=support_type)
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

    # one tau per run: resolve the effective threshold BEFORE the abstain
    # branches so reader-abstained rows report the same tau as gate decisions.
    tau = fixed_tau
    if gate in ("calibrated", "conformal") and cal_tau is not None:
        tau = float(cal_tau)
    gate_effective = (gate if gate == "fixed"
                      or (gate in ("calibrated", "conformal") and cal_tau is not None)
                      else "none")
    abstain_reason = ""
    if is_unanswerable:
        abstained = True
        abstain_reason = "reader returned UNANSWERABLE"
    elif gate_effective == "none":
        abstained = False
        # a failed quote/support/citation check still blocks a cited answer
        if citation_info["status"] != "none" and not verified and not visual_ok:
            abstained = True
            abstain_reason = _unsupported_reason()
    else:
        abstained = not apply_fixed(score, tau)
        abstain_reason = "" if not abstained else f"score {score:.3f} < tau {tau}"
        if abstained and ((verified and evidence_valid) or visual_ok):
            abstained = False
            abstain_reason = "verifier rescue"
        if not abstained and require_support and not verified and not visual_ok:
            abstained = True
            abstain_reason = _unsupported_reason()

    passages = closest_passages(question, sentences) if abstained else []

    return {
        "answer": answer,
        "abstained": abstained,
        "abstain_reason": abstain_reason,
        "not_found": abstained,
        "verified": verified,
        "tier": tier,
        "citations": {"status": citation_info["status"],
                      "n_refs": citation_info["n_refs"],
                      "n_resolved": citation_info["n_resolved"],
                      "unresolved": citation_info["unresolved"]},
        "visual_confirm": visual_confirm,
        "support": {"score": round(float(support_score), 4),
                    "method": support_method,
                    "require": require_support},
        "score": round(score, 4),
        "tau": tau if gate_effective != "none" else 0.0,
        "gate_mode": gate_effective,
        "retrieval": [{"doc_id": d, "page": p,
                       "keyword_score": round(float(s), 3) if mode_used == "keyword" else None,
                       "score": round(float(s), 4), "rank": i + 1}
                      for i, (d, p, s, _) in enumerate(ranked)],
        "retrieval_debug": retrieval_info,
        "closest_passages": passages,
        "evidence": {"doc_id": doc_id, "page": page,
                     "boxes": [list(b) for b in final_boxes],
                     "box_frac": [list(b) for b in final_frac],
                     "sentence_ids": ([sid for _d, _p, sid in cited_refs]
                                      if verified else []),
                     "refs": ([list(r) for r in cited_refs] if verified else []),
                     "pages": list(pages_evidence.values()),
                     "region_boxes": [list(b) for b in region_boxes],
                     "crop_text": crop_text, "crop_has_text": has_text,
                     "heat_concentration": round(heat_conc, 4),
                     "page_width": W, "page_height": H,
                     "page_url": "/media/" + Path(img_path).name,
                     "localize": {"score": round(loc_score, 4), "valid": evidence_valid,
                                  "source": loc_source,
                                  "quote_score": round(qscore, 4),
                                  "quote_ok": quote_ok,
                                  "verified": verified,
                                  "support_method": support_method,
                                  "citations_valid": citations_valid,
                                  "n_words": len(loc_words),
                                  "words": [w.text for w in loc_words]},
                     "crop_url": "/media/" + Path(crop_path).name,
                     "overlay_url": "/media/" + Path(ov_path).name},
        "cited_sentences": cited_sentences,
        "claims": [c.to_dict() for c in samples[0].claims],
        "samples": [{"answer": s.answer, "quote": s.quote,
                     "sentence_ids": s.sentence_ids,
                     "sentence_refs": s.sentence_refs,
                     "region": s.region} for s in samples],
        "features": feats.to_dict(),
        "api_calls": api_calls,
        "timings": {"total": round(time.time() - t0, 3), "reader": round(t_r, 3)},
    }
