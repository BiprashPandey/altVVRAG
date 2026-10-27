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
import os
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
from vvrag.schemas import GroundedRegion, WordBox
from vvrag.verify.citation import citation_overlap, page_number_sink
from vvrag.verify.claim_verifier import (claim_confidence, claim_is_supported,
                                       normalize_claim_label)
from vvrag.verify.examiner import (examiner_confirmed, parse_examiner_output,
                                   verdict_supported_value)
from vvrag.verify.question_details import (check_question_details,
                                           corpus_detail_vocab)

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


def _check_online_encoder(cfg_ret) -> None:
    """Raise RuntimeError when the online ColQwen query encoder cannot run.

    Phase 4: the live path may encode the question on the fly on the GPU box.
    This guard is what keeps a CPU-only laptop (or a fresh HF cache) from
    triggering a multi-GB download / CPU load inside a request: CUDA is
    required unless the configured device is explicitly cpu AND
    ALLOW_CPU_QUERY_ENCODER=1 is set.
    """
    import importlib.util as _ilu
    import os as _os

    want_dev = str(getattr(cfg_ret, "online_encoder_device", "cuda")
                   or "cuda").lower()
    override = _os.environ.get("ALLOW_CPU_QUERY_ENCODER", "") == "1"
    if _ilu.find_spec("colpali_engine") is None:
        raise RuntimeError("colpali-engine not installed")
    import torch as _torch

    if (not want_dev.startswith("cpu")
            and not _torch.cuda.is_available() and not override):
        raise RuntimeError(
            "CUDA unavailable (set ALLOW_CPU_QUERY_ENCODER=1 to force the "
            "slow CPU encoder)")


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
        import os as _os
        cfg: AppConfig = load_config(_os.environ.get("VVRAG_CONFIG", "config.yaml"))
    except Exception as e:  # noqa: BLE001 - config is optional for mock mode
        log.warning("config load failed, using defaults: %s", e)
        cfg = AppConfig()
    quote_threshold = float(getattr(cfg.verify, "quote_threshold", 0.8))
    index_manifest = {}
    index_manifest_mismatches: list[str] = []
    try:
        from vvrag.ingest.index import get_manifest, manifest_mismatches

        index_manifest = get_manifest(con)
        if index_manifest:
            index_manifest_mismatches = manifest_mismatches(con, {
                "embedding_model": cfg.retriever.model_id,
                "dpi": cfg.render.dpi,
                "collection_full": cfg.qdrant.collection_full,
                "collection_pooled": cfg.qdrant.collection_pooled,
            })
    except Exception as e:  # noqa: BLE001 - legacy index without manifest
        log.warning("index manifest unavailable: %s", e)

    # ---------------- retrieval (Phase 1: keyword | visual | hybrid) ------
    cfg_ret = getattr(cfg, "retriever", None)
    mode = (retrieval_mode or getattr(cfg_ret, "retrieval_mode", "keyword")
            or "keyword").lower()
    reader_pages = max(1, int(getattr(cfg.reader, "max_pages", 3)))
    candidate_k = max(1, int(getattr(cfg_ret, "candidate_k", 5)))
    rerank_k = max(1, int(getattr(cfg_ret, "rerank_top_k", 5)))
    pool_k = max(candidate_k, rerank_k, reader_pages)
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

        qvec_dir = (os.environ.get("VVRAG_QUERY_VECTORS_DIR")
                    or getattr(cfg_ret, "query_vectors_dir",
                               "data/query_cache"))
        store = QueryVectorStore(qvec_dir)
        hit = store.get(question)
        if hit is None and bool(getattr(cfg_ret, "online_encoder", True)):
            # Phase 4: encode the question on the fly (laptop GPU) and cache
            # it; any failure (no GPU/weights) is an honest keyword fallback.
            try:
                # Phase 4 guard: never trigger a multi-GB model download /
                # CPU load inside a request (see _check_online_encoder).
                _check_online_encoder(cfg_ret)
                from vvrag.retrieve.query_embed import create_query_embedder

                enc = create_query_embedder(
                    "colqwen",
                    device=str(getattr(cfg_ret, "online_encoder_device",
                                      "auto") or "auto"))
                enc.load()
                qv, qlabels = enc.embed_query(question)
                qvecs = np.asarray(qv, dtype=np.float32)
                try:
                    store.put(question, qvecs, list(qlabels),
                              model=str(getattr(enc, "model_id", "")
                                        or getattr(enc, "model_name", "")),
                              device=str(getattr(enc, "device", "")))
                except Exception:  # noqa: BLE001 - caching is best-effort
                    pass
                retrieval_note = ("online ColQwen query encoding "
                                  "(cache miss; stored)")
            except Exception as e:  # noqa: BLE001 - no GPU/weights = keyword
                qvecs = None
                retrieval_note = (f"online query encoder unavailable "
                                  f"({type(e).__name__}: {str(e)[:80]}); "
                                  f"keyword fallback")
        elif hit is not None:
            qvecs = hit[0]
        if qvecs is None:
            if not retrieval_note:
                retrieval_note = ("no cached query vector; keyword fallback "
                                  "(run scripts/precompute_query_vectors.py)")
            ranked = kw_full[:pool_k]
        else:
            try:
                from vvrag.retrieve.fusion import (deck_aggregate, rank_order,
                                                   rrf_fuse,
                                                   visual_page_scores)

                kw_pages = [(d, p) for d, p, _s, _w in kw_full]
                if str(getattr(cfg_ret, "visual_scope", "all")).lower() == "all":
                    # Report parity: visual retrieval must be able to find a
                    # page that keyword retrieval ranked poorly. Score every
                    # embedded page; word boxes are loaded only for pages that
                    # survive fusion.
                    all_pages = [(str(d), int(p)) for d, p in con.execute(
                        "SELECT doc_id, page FROM pages ORDER BY doc_id, page")]
                    visual_pages = all_pages or kw_pages
                else:
                    visual_pages = kw_pages
                vis_scores = visual_page_scores(index_dir, visual_pages, qvecs)
                if not vis_scores:
                    raise RuntimeError(
                        "no page embeddings for visual scoring")
                vis_order = rank_order(vis_scores)
                if mode == "visual":
                    order = vis_order
                    rank_scores = vis_scores
                else:
                    fused = rrf_fuse(
                        [kw_pages, vis_order],
                        k=int(getattr(cfg_ret, "rrf_k", 60)),
                        weights=[float(getattr(cfg_ret, "keyword_weight", 1.0)),
                                 float(getattr(cfg_ret, "visual_weight", 1.0))],
                    )
                    order = [(d, p) for d, p, _s in fused]
                    rank_scores = {(d, p): s for d, p, s in fused}
                top = []
                for d, p in order[:pool_k]:
                    key = (d, p)
                    if key not in words_map:
                        words_map[key] = page_words(con, d, p)
                    top.append((d, p, rank_scores[key], words_map[key]))
                ranked = top
                mode_used = mode
            except Exception as e:  # noqa: BLE001 - honest keyword fallback
                mode_used = "keyword"
                retrieval_note = (f"visual scoring failed "
                                  f"({type(e).__name__}: {str(e)[:80]}); "
                                  f"keyword fallback")
                ranked = kw_full[:pool_k]
    else:
        ranked = kw_full[:pool_k]

    # Phase 4: deterministic lexical rerank after fusion (IDF + bigram +
    # number bonus). Pure CPU, no model. Recorded for measurement runs.
    rerank_used = False
    if bool(getattr(cfg_ret, "rerank", False)) and len(ranked) > 1:
        from vvrag.retrieve.rerank import lexical_rerank

        ranked = lexical_rerank(
            question, ranked, k=max(rerank_k, reader_pages))
        rerank_used = True

    def _top_pages(scores, n: int = 3):
        if not scores:
            return []
        return [{"doc_id": d, "page": p, "score": round(float(s), 4)}
                for (d, p), s in sorted(scores.items(), key=lambda kv: kv[1],
                                        reverse=True)[:n]]

    retrieval_info = {
        "mode_requested": mode,
        "mode_used": mode_used,
        "rerank_used": rerank_used,
        "keyword_weight": float(getattr(cfg_ret, "keyword_weight", 1.0)),
        "visual_weight": float(getattr(cfg_ret, "visual_weight", 1.0)),
        "candidate_k": candidate_k,
        "rerank_k": rerank_k,
        "reader_pages": reader_pages,
        "candidate_pool_size": len(ranked),
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
    max_pages = reader_pages
    ctx_keys = [(d, p) for d, p, _s, _w in ranked[:max_pages]]
    if primary_key not in ctx_keys:
        ctx_keys.insert(0, primary_key)

    # Phase 4: bridge-page resolution for multi-hop questions. Pages covering
    # the question tokens that the primary page lacks are added to the reader
    # context (the weakest ranked page yields the slot); everything is still
    # OCR pages from the same index. Recorded per row in retrieval_debug.
    bridge_pages: list[dict] = []
    if (bool(getattr(cfg_ret, "bridge_resolution", True))
            and len(words_map) > len(ctx_keys)):
        from vvrag.retrieve.bridge import resolve_bridge

        extra = resolve_bridge(question, primary_key, ctx_keys, words_map,
                               max_extra=1)
        if extra:
            d_b, p_b, s_b = extra[0]
            ctx_keys = ctx_keys[: max_pages - 1] + [(d_b, p_b)]
            ctx_keys = [primary_key] + [k for k in ctx_keys
                                        if k != primary_key]
            bridge_pages.append({"doc_id": d_b, "page": p_b, "score": s_b})
            retrieval_info["note"] = (
                (retrieval_info["note"] + "; ") if retrieval_info["note"] else ""
            ) + f"bridge page {d_b}:{p_b} promoted into reader context"
    retrieval_info["bridge_pages"] = bridge_pages

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
                "sentences": sentences_from_page(
                    con, d, p,
                    include_tables=bool(getattr(getattr(cfg, "ingest", None),
                                                "table_cells", True)))[0],
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
        reader = create_reader(
            provider, model, DiskCache(cfg.cache_dir),
            RateLimiter(cfg.ratelimit.requests_per_minute),
            max_output_tokens=int(getattr(cfg.reader, "max_output_tokens", 512)),
            fallback_provider=getattr(cfg.reader, "fallback_provider", ""),
            fallback_model=getattr(cfg.reader, "fallback_model", ""))
        images = [img_path, crop_path]
    from vvrag.reader.client import run_reader_samples
    from vvrag.reader.prompts import (build_multipage_sentence_prompt,
                                      build_prompt, build_sentence_prompt)

    prompt_stats: dict = {}
    if reader_mode == "mock":
        prompt = build_prompt(question, crop_text if has_text else None, max_claims=5)
    elif len(blocks) > 1:
        prompt = build_multipage_sentence_prompt(
            question, blocks, crop_text if has_text else None, max_claims=5,
            stats_out=prompt_stats)
    else:
        prompt = build_sentence_prompt(question, sentences,
                                       crop_text if has_text else None, max_claims=5)
    retrieval_info["prompt_pages"] = prompt_stats
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
        derived_from_quote = False
        # P3: a reader may return a verbatim table-cell quote but omit
        # sentence_ids. If exactly one shown sentence contains that quote
        # (normalized, length >= 2 tokens), resolve it deterministically and
        # record that it was derived. Never rebind a non-empty unresolved id.
        if not refs and sample.quote and len(
                normalize_answer(sample.quote).split()) >= 2:
            candidates = []
            for key in ctx_keys:
                st = _get_state(key)
                for sent in (st["sentences"] if st else []):
                    score = quote_match_score(sample.quote, [sent.text],
                                              ocr_normalize=ocr_norm)
                    if score >= 0.95:
                        candidates.append((key, sent, score))
            uniq = {(k[0], k[1], sent.sent_id): (k, sent)
                    for k, sent, _score in candidates}
            if len(uniq) == 1:
                (key, sent) = next(iter(uniq.values()))
                resolved = [(key[0], key[1], sent.sent_id)]
                derived_from_quote = True
        texts: list[str] = []
        for d, p, sid in resolved:
            st = _get_state((d, p))
            s = next((x for x in (st["sentences"] if st else [])
                      if x.sent_id == sid), None)
            if s is not None:
                texts.append(s.text)
        qs = (quote_match_score(sample.quote, texts, ocr_normalize=ocr_norm)
              if texts else 0.0)
        if resolved and not refs:
            status = "ok"
        elif not refs:
            status = "none"
        elif not resolved:
            status = "unresolved"
        elif unresolved:
            status = "partial"
        else:
            status = "ok"
        resolution = {"status": status, "n_refs": len(refs),
                      "n_resolved": len(resolved),
                      "unresolved": unresolved[:5],
                      "derived_from_quote": derived_from_quote}
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
    # Run 9 (guideline s4.1): support checks are reported, never a veto.
    require_support = bool(getattr(cfg.verify, "require_answer_support", False))
    support_threshold = float(getattr(cfg.verify, "support_entail_threshold", 0.5))
    type_check = getattr(cfg.verify, "answer_type_check", False)
    comparison_check = getattr(cfg.verify, "comparison_check", "warn")
    support_warnings: list[str] = []
    if is_unanswerable:
        supported, support_score, support_method = False, 0.0, "empty"
    elif not cited_texts:
        supported, support_score, support_method = False, 0.0, "no_citation"
    else:
        from vvrag.verify.support import qa_support_report

        (supported, support_score, support_method,
         support_warnings) = qa_support_report(
            question, answer, cited_texts, stmt_entail, support_threshold,
            type_check=type_check, comparison_check=comparison_check)

    # Phase C: deterministic document candidates per claim (not trusted yet;
    # the examiner must judge the candidate). Multi-page when the reader cited
    # or the context includes another page.
    claim_regions: list[dict] = []
    if not is_unanswerable and str(answer).strip():
        from vvrag.ground.candidates import candidates_for_claim

        for claim in (samples[0].claims or [])[:5]:
            claim_text = str(getattr(claim, "text", "") or "").strip()
            if not claim_text:
                continue
            cands = []
            for ckey in ctx_keys:
                cst = _get_state(ckey)
                if not cst:
                    continue
                try:
                    cands.extend(candidates_for_claim(
                        claim_text, cst["sentences"], ckey[0], ckey[1],
                        cst["W"], cst["H"],
                        reader_region=(samples[0].region
                                       if ckey == primary_key else None),
                        words=cst["words"],
                        min_shared=int(getattr(cfg.verify,
                                               "citation_min_terms", 1)),
                        top_k=5, focus=str(answer)))
                except Exception as e:  # noqa: BLE001 - grounding is best-effort
                    log.debug("claim candidates failed for %s: %s", ckey, e)
            # cross-page dedupe + deterministic top-5
            seen = set()
            uniq = []
            for c in sorted(cands, key=lambda x: -x.score):
                key = (c.region.doc_id, c.region.page, c.region.box,
                       c.region.text)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(c)
            def _cand_dict(c):
                d = c.region.to_dict()
                d["rank_score"] = round(float(c.score), 4)
                d["matched_terms"] = list(c.matched_terms)
                return d

            claim_regions.append({
                "text": claim_text,
                "candidates": [_cand_dict(c) for c in uniq[:5]],
                "best": uniq[0].region.to_dict() if uniq else None,
                "best_score": round(float(uniq[0].score), 4) if uniq else 0.0,
            })

    # ---------------- Run 9 (guideline s4.3/s7): the examiner -------------
    # The examiner is shown the claim on EVERY answer (not just failed ones),
    # finds the quote itself and judges fit. A failed or unavailable examiner
    # means "not confident enough" -- it never vetoes and never silently
    # switches to the reader's own family.
    region_present = bool(samples[0].region and samples[0].region.get("box_frac"))
    ex_mode = str(getattr(cfg.verify, "mode", "examiner_shown")
                  or "examiner_shown").lower()
    ex_on = ex_mode in ("examiner_shown", "on", "each", "true") or bool(
        getattr(cfg.verify, "examiner_on_every_answer", True))
    ex_threshold = float(getattr(cfg.verify, "examiner_quote_threshold", 0.6))
    examiner: dict = {"ran": False, "served_by": "", "error": "",
                      "quote_found": False, "quote_seen": "",
                      "answer_supported": "", "answers_the_question": None,
                      "citation_specific": None, "region_supports": None,
                      "problem": "", "quote_seen_match": 0.0,
                      "confirmed": False, "reason": "not attempted",
                      "attempts": [], "errors": {}, "evidence_mode": "",
                      "citation_precheck": {}}
    independence_ok = True
    independence_required = bool(getattr(cfg.verify, "independence_required", True))

    # Evidence text for the examiner quote cross-check and the details check.
    # The cited sentence alone is too narrow: years and company names often sit
    # in the page header or deck title.
    evidence_keys = [primary_key] + [(d, p) for d, p, _s in cited_refs]
    rpage = (samples[0].region or {}).get("page")
    if rpage is not None:
        rkey = next((k for k in ctx_keys if k[1] == int(rpage)), None)
        if rkey is not None:
            evidence_keys.append(rkey)
    page_texts: list[str] = []
    for key in dict.fromkeys(evidence_keys):
        st = _get_state(key)
        if st is not None:
            page_texts.append(" ".join(w.text for w in st["words"]))
    deck_texts: list[str] = []
    try:
        first_page_words = page_words(con, doc_id, 0)
        if first_page_words:
            deck_texts.append(" ".join(w.text for w in first_page_words))
    except Exception:  # noqa: BLE001 - deck title is best-effort
        pass
    region_texts = [crop_text] if has_text and crop_text else []

    def _check_img() -> tuple[str, str]:
        """Evidence crop for the examiner; never the full page if a usable crop exists."""
        from PIL import Image as _PILImage

        def _usable(path) -> bool:
            try:
                with _PILImage.open(path) as im:
                    return im.size[0] >= 32 and im.size[1] >= 32
            except Exception:  # noqa: BLE001 - unreadable/absent crop
                return False

        # Phase C: prefer the deterministic document candidate for the first
        # claim over the reader's loose answer crop. The verifier must judge
        # the cited candidate region, not just the page.
        best = None
        if claim_regions:
            best = claim_regions[0].get("best")
        if best and best.get("box_frac"):
            try:
                bkey = (best["doc_id"], int(best["page"]))
                bst = _get_state(bkey)
                if bst is not None:
                    bf = best["box_frac"]
                    bp = _PILImage.open(bst["img_path"])
                    bw, bh = bp.size
                    pad = 0.02
                    box = (int(max(0.0, bf[0] - pad) * bw),
                           int(max(0.0, bf[1] - pad) * bh),
                           int(min(1.0, bf[2] + pad) * bw),
                           int(min(1.0, bf[3] + pad) * bh))
                    if box[2] > box[0] and box[3] > box[1]:
                        cp = Path(str(crop_path)).with_name(
                            Path(str(crop_path)).stem + "_claimcheck.png")
                        bp.crop(box).save(cp)
                        if _usable(cp):
                            return str(cp), "claim_candidate"
            except Exception:  # noqa: BLE001 - fall back to reader crop
                pass
        f = (samples[0].region or {}).get("box_frac")
        if f:
            rkey = primary_key
            if rpage is not None:
                rkey = next((k for k in ctx_keys if k[1] == int(rpage)),
                            primary_key)
            st = _get_state(rkey)
            if st is not None:
                try:
                    im = _PILImage.open(st["img_path"])
                    w, h = im.size
                    pad = 0.04
                    box = (int(max(0.0, f[0] - pad) * w),
                           int(max(0.0, f[1] - pad) * h),
                           int(min(1.0, f[2] + pad) * w),
                           int(min(1.0, f[3] + pad) * h))
                    if box[2] > box[0] and box[3] > box[1]:
                        cp = Path(str(crop_path)).with_name(
                            Path(str(crop_path)).stem + "_vcheck.png")
                        im.crop(box).save(cp)
                        if _usable(cp):
                            return str(cp), "region_crop"
                except Exception:  # noqa: BLE001 - fall back to answer crop
                    pass
        if crop_path and Path(str(crop_path)).exists() and _usable(crop_path):
            return str(crop_path), "answer_crop"
        # A tiny answer crop (often a narrow title) is rejected by Groq
        # (<32 px). Fall back to the cited sentence's box before the page.
        try:
            for d, p, sid in cited_refs:
                st = _get_state((d, p))
                sent = next((x for x in (st["sentences"] if st else [])
                             if x.sent_id == sid), None)
                if sent is None:
                    continue
                im = _PILImage.open(st["img_path"])
                w, h = im.size
                x0, y0, x1, y1 = sent.box
                pad = max(8, int(0.02 * min(w, h)))
                box = (max(0, x0 - pad), max(0, y0 - pad),
                       min(w, x1 + pad), min(h, y1 + pad))
                if box[2] - box[0] < 32 or box[3] - box[1] < 32:
                    continue
                cp = Path(str(crop_path)).with_name(
                    Path(str(crop_path)).stem + "_sentcheck.png")
                im.crop(box).save(cp)
                if _usable(cp):
                    return str(cp), "sentence_crop"
        except Exception:  # noqa: BLE001 - best effort
            pass
        return img_path, "page"

    def _crop_claim_region(region: dict | None, suffix: str) -> str | None:
        """Crop one deterministic claim candidate; None when unusable."""
        if not region or not region.get("box_frac"):
            return None
        try:
            from PIL import Image as _PILImage

            bkey = (region["doc_id"], int(region["page"]))
            bst = _get_state(bkey)
            if bst is None:
                return None
            im = _PILImage.open(bst["img_path"])
            w, h = im.size
            bf = region["box_frac"]
            pad = 0.02
            box = (int(max(0.0, bf[0] - pad) * w),
                   int(max(0.0, bf[1] - pad) * h),
                   int(min(1.0, bf[2] + pad) * w),
                   int(min(1.0, bf[3] + pad) * h))
            if box[2] - box[0] < 32 or box[3] - box[1] < 32:
                return None
            cp = Path(str(crop_path)).with_name(
                Path(str(crop_path)).stem + suffix)
            im.crop(box).save(cp)
            return str(cp)
        except Exception:  # noqa: BLE001 - candidate crop is best-effort
            return None

    # Deterministic citation-specific precheck (report-gap fix). A cited
    # region that shares no content term with the claim cannot be the
    # specific evidence, even if the claim is true elsewhere on the page.
    # For a visual region with no OCR text the independent verifier decides.
    citation_claim_text = " ".join(
        [str(answer)] + [c.text for c in samples[0].claims if c.text])
    citation_evidence = (cited_texts if cited_texts else
                         ([crop_text] if crop_text else []))
    if is_unanswerable or not str(answer).strip() or not citation_evidence:
        citation_ok, citation_score, citation_terms = None, 0.0, []
    else:
        citation_ok, citation_score, citation_terms = citation_overlap(
            citation_claim_text, citation_evidence,
            min_shared=int(getattr(cfg.verify, "citation_min_terms", 1)))
    numeric_sink = page_number_sink(citation_claim_text, citation_evidence)
    if numeric_sink:
        citation_ok = False

    verdict: dict | None = None
    claim_level_on = bool(getattr(cfg.verify, "claim_level", True))
    claim_max = max(0, int(getattr(cfg.verify, "claim_level_max", 5)))
    claim_image_max = max(0, int(getattr(cfg.verify, "claim_level_images", 2)))
    claim_prompt_texts: list[str] = []
    claim_prompt_evidence: list[str] = []
    claim_crops: list[str] = []
    claim_level_indices: list[int] = []
    claim_level_ok = True
    if not is_unanswerable and ex_on and str(answer).strip():
        from vvrag.reader.prompts import (build_examiner_prompt,
                                         build_text_examiner_prompt)

        exam_evidence = str(getattr(cfg.verify, "examiner_evidence",
                                   "region")).lower()
        text_examiner = exam_evidence in ("text", "text_only", "text-only")
        if text_examiner:
            # Text-only independent examiner: no image tokens, no crops. The
            # deterministic citation precheck remains a hard gate for VERIFIED.
            check_img, evidence_mode = "", "text"
            examiner["evidence_mode"] = "text"
            ex_imgs = []
            claim_image_max = 0
            if claim_level_on:
                for i, cr in enumerate(claim_regions[:claim_max]):
                    text = str(cr.get("text") or "").strip()
                    if not text:
                        continue
                    best = cr.get("best") or {}
                    claim_level_indices.append(i)
                    claim_prompt_texts.append(text)
                    claim_prompt_evidence.append(str(best.get("text") or ""))
        else:
            check_img, evidence_mode = _check_img()
            if claim_level_on:
                for i, cr in enumerate(claim_regions[:claim_max]):
                    text = str(cr.get("text") or "").strip()
                    if not text:
                        continue
                    best = cr.get("best") or {}
                    claim_level_indices.append(i)
                    claim_prompt_texts.append(text)
                    claim_prompt_evidence.append(str(best.get("text") or ""))
                    if len(claim_crops) < claim_image_max:
                        if i == 0 and str(check_img) != str(img_path):
                            crop = str(check_img)
                        else:
                            crop = _crop_claim_region(best, f"_claim{i}.png")
                        claim_crops.append(crop or img_path)
            if exam_evidence == "page":
                check_img, evidence_mode = img_path, "page"
                ex_imgs = [check_img]
                claim_crops = []
            elif exam_evidence in ("page_and_region", "page+region", "both",
                                   "page_region"):
                # Full page for context + one highlighted candidate per claim.
                region_imgs = claim_crops or (
                    [check_img] if str(check_img) != str(img_path) else [])
                ex_imgs = [img_path] + region_imgs
                examiner["evidence_mode"] = ("page_and_region"
                                             if len(ex_imgs) > 1 else "page")
            else:
                ex_imgs = claim_crops or [check_img]
            examiner["evidence_mode"] = (examiner.get("evidence_mode")
                                         or evidence_mode)
        if text_examiner:
            ex_prompt = build_text_examiner_prompt(
                question, answer, samples[0].quote or "",
                samples[0].found_in or "",
                cited_texts=cited_texts,
                page_texts=page_texts,
                claims=claim_prompt_texts or None,
                claim_evidence=claim_prompt_evidence)
        else:
            ex_prompt = build_examiner_prompt(
                question, answer, samples[0].quote or "",
                samples[0].found_in or "",
                claims=claim_prompt_texts or None,
                claim_evidence=claim_prompt_evidence,
                n_claim_images=len(claim_crops))
        if reader_mode == "mock":
            from vvrag.verify.examiner import MockExaminer

            raw = MockExaminer().examine(question, answer,
                                         samples[0].quote or "", cited_texts)
            verdict = parse_examiner_output(raw)
            if claim_level_on and claim_prompt_texts:
                mock_claims = []
                for pos, ci in enumerate(claim_level_indices):
                    ctext = claim_prompt_texts[pos]
                    craw = MockExaminer().examine(question, ctext, "",
                                                  cited_texts)
                    cv = parse_examiner_output(craw)
                    label = normalize_claim_label(
                        "", cv.get("answer_supported", ""),
                        cv.get("citation_specific"),
                        cv.get("region_supports"))
                    mock_claims.append({
                        "index": ci, "label": label,
                        "citation_specific": cv.get("citation_specific"),
                        "region_supports": cv.get("region_supports"),
                        "quote_found": cv.get("quote_found"),
                        "quote_seen": cv.get("quote_seen", ""),
                        "problem": cv.get("problem", ""),
                    })
                verdict["claims"] = mock_claims
            examiner.update(ran=True, served_by="mock/examiner", mock=True)
        else:
            from vvrag.cache import DiskCache as _DiskCache
            from vvrag.ratelimit import RateLimiter as _RateLimiter
            from vvrag.reader.client import create_reader as _create_reader

            provider_ex = str(getattr(cfg.verify, "examiner_provider", "groq"))
            models_ex = list(getattr(cfg.verify, "examiner_models", None)
                             or [getattr(cfg.verify, "cross_family_model", "")])
            independence_ok = (provider_ex.lower()
                               != str(provider or "").lower())
            for m_ex in models_ex:
                if not m_ex:
                    continue
                attempt = {"provider": provider_ex, "model": m_ex,
                           "ok": False, "error": ""}
                try:
                    cl = _create_reader(
                        provider_ex, m_ex, _DiskCache(cfg.cache_dir),
                        _RateLimiter(cfg.ratelimit.requests_per_minute),
                        max_output_tokens=int(getattr(
                            cfg.verify, "examiner_max_tokens", 700)))
                    raw = cl.generate(
                        ex_imgs, ex_prompt,
                        float(getattr(cfg.verify, "examiner_temperature", 0.0)),
                        1)[0]
                    api_calls += 1
                    verdict = parse_examiner_output(raw)
                    attempt["ok"] = True
                    examiner["attempts"].append(attempt)
                    examiner.update(ran=True,
                                    served_by=f"{provider_ex}/{m_ex}")
                    break
                except Exception as e:  # noqa: BLE001 - try next model
                    err = str(e)[:300]
                    attempt["error"] = err
                    examiner["attempts"].append(attempt)
                    examiner["errors"][f"{provider_ex}/{m_ex}"] = err
                    log.warning("examiner %s/%s failed: %s", provider_ex, m_ex, e)
            if verdict is None:
                if examiner["errors"]:
                    examiner["error"] = "; ".join(
                        f"{k}: {v}" for k, v in examiner["errors"].items())
                else:
                    examiner["error"] = "no examiner model available"
                verdict = {"ok": False, "quote_found": False, "quote_seen": "",
                           "answer_supported": "", "answers_the_question": None,
                           "problem": "",
                           "parse_error": examiner["error"]}
        if citation_ok is False and verdict is not None:
            # The cited text itself contradicts the citation. Keep the
            # verifier's label/quote but make the citation-specific fields
            # fail structurally; the answer remains shown as not confident.
            verdict = dict(verdict)
            verdict["citation_specific"] = False
            verdict["region_supports"] = False
            if not verdict.get("problem"):
                verdict["problem"] = (
                    "the cited text shares no content term with the claim")
        examiner.update(verdict)
        examiner["citation_precheck"] = {
            "ok": citation_ok, "score": round(float(citation_score), 4),
            "shared_terms": citation_terms[:10],
            "numeric_sink": bool(numeric_sink),
        }
        claim_verdicts = list((verdict or {}).get("claims") or [])
        claim_by_index = {}
        for _i, cv in enumerate(claim_verdicts):
            try:
                claim_by_index[int(cv.get("index", _i))] = cv
            except (TypeError, ValueError):
                claim_by_index[_i] = cv
        claim_level_ran = bool(claim_level_on and claim_level_indices
                               and claim_verdicts)
        claim_min_shared = int(getattr(cfg.verify, "citation_min_terms", 1))
        for i, claim in enumerate(samples[0].claims or []):
            ctext = str(getattr(claim, "text", "") or "").strip()
            cr = claim_regions[i] if i < len(claim_regions) else None
            best = (cr or {}).get("best") or {}
            cv = claim_by_index.get(i)
            cev = [str(best.get("text") or "")] if best.get("text") else (
                cited_texts if cited_texts else [])
            if ctext and cev:
                c_ok, c_score, c_terms = citation_overlap(
                    ctext, cev, min_shared=claim_min_shared)
            else:
                c_ok, c_score, c_terms = None, 0.0, []
            if best.get("text") and page_number_sink(
                    ctext, [str(best.get("text") or "")]):
                c_ok = False
            if cv is not None:
                label = normalize_claim_label(
                    cv.get("label", ""), cv.get("answer_supported", ""),
                    cv.get("citation_specific"), cv.get("region_supports"))
                reason = str(cv.get("problem") or "")
                citation_specific = cv.get("citation_specific")
            elif claim_level_ran:
                label = "insufficient"
                reason = "the examiner did not return this claim's verdict"
                citation_specific = None
            else:
                label = normalize_claim_label(
                    "", examiner.get("answer_supported", ""),
                    examiner.get("citation_specific"),
                    examiner.get("region_supports"))
                reason = "claim-level verdict unavailable; sample verdict used"
                citation_specific = examiner.get("citation_specific")
            if c_ok is False and label == "supported":
                label = "partially_supported"
                reason = (reason or
                          "the candidate region shares no content term with the claim")
            if best:
                claim.regions = [GroundedRegion.from_dict(best)]
            claim.label = label
            claim.reason = reason
            claim.confidence = claim_confidence(label)
            claim.citation_specific = (
                citation_specific if isinstance(citation_specific, bool)
                else examiner.get("citation_specific"))
            claim.withheld = False
            if cr is not None:
                cr.update({
                    "label": label,
                    "verified": label == "supported",
                    "partial": label == "partially_supported",
                    "citation_specific": claim.citation_specific,
                    "region_supports": (cv or {}).get(
                        "region_supports", examiner.get("region_supports")),
                    "reason": reason,
                    "confidence": claim.confidence,
                    "citation_check": {
                        "ok": c_ok, "score": round(float(c_score), 4),
                        "shared_terms": c_terms[:10]},
                })
        labels = [c.label for c in (samples[0].claims or [])]
        claim_level_ok = (not claim_level_on or not labels
                          or (claim_level_ran and all(
                              x == "supported" for x in labels)))
        examiner["claim_level"] = {
            "enabled": claim_level_on,
            "ran": claim_level_ran,
            "n_claims": len(labels),
            "n_supported": sum(1 for x in labels if x == "supported"),
            "n_partial": sum(1 for x in labels
                             if x == "partially_supported"),
            "n_failed": sum(1 for x in labels
                            if x in ("insufficient", "unsupported")),
            "labels": labels,
            "ok": bool(claim_level_ok),
        }
        confirmed, qs_match, ex_reason = examiner_confirmed(
            verdict, samples[0].quote or "", cited_texts, ex_threshold, ocr_norm,
            page_texts=page_texts, region_texts=region_texts,
            require_citation_specific=bool(getattr(
                cfg.verify, "require_citation_specific", True))
            and not text_examiner)
        examiner.update(confirmed=bool(confirmed),
                        quote_seen_match=round(float(qs_match), 4),
                        reason=ex_reason)
        if independence_required and not independence_ok:
            examiner.update(
                confirmed=False,
                reason="the examiner shares the reader's provider; "
                       "independence is required")

    # Run 9 (s4.4): question-detail grounding (years/dates/units/entities).
    detail_mode = str(getattr(cfg.verify, "question_detail_check", "warn"))
    corpus_years, corpus_entities, corpus_durations = corpus_detail_vocab(con)
    details = check_question_details(
        question,
        cited_texts if cited_texts else ([crop_text] if has_text else []),
        quote_seen=examiner.get("quote_seen", ""),
        corpus_years=corpus_years, corpus_entities=corpus_entities,
        corpus_durations=corpus_durations,
        mode=detail_mode, page_texts=page_texts, deck_texts=deck_texts)

    # ---------------- Run 9 (s7): the three-state decision ----------------
    quote_on_page = bool(quote_ok)
    citation_conflict = citation_info["status"] == "unresolved"
    rules_ok = bool(quote_on_page and details["ok"] and not citation_conflict
                    and citation_ok is not False)
    impossible_years = list(details.get("impossible_years") or [])
    impossible_durations = list(details.get("impossible_durations") or [])
    impossible_constraints = impossible_years + impossible_durations
    nothing_found = bool(is_unanswerable or samples[0].no_evidence
                         or not str(answer).strip() or impossible_constraints)
    if nothing_found:
        state = "nothing_found"
    elif (bool(examiner["confirmed"]) and rules_ok and bool(examiner["ran"])
          and (independence_ok or not independence_required)
          and claim_level_ok):
        state = "verified"
    else:
        state = "not_confident"
    verified = state == "verified"

    reasons: list[str] = []
    if state == "nothing_found":
        if impossible_constraints:
            impossible = set(impossible_constraints)
            concrete = [w for w in (details.get("warnings") or [])
                        if any(y in w for y in impossible)]
            reasons.extend(concrete or [
                "the question asks about "
                f"{', '.join(impossible_constraints)}, which does not appear "
                "anywhere in this document set"])
        else:
            reasons.append("the reader found nothing relevant on the "
                           "retrieved pages")
    elif state == "not_confident":
        if examiner["error"] and not examiner["ran"]:
            reasons.append(f"the examiner check could not be run "
                           f"({examiner['error']})")
        elif examiner["ran"] and not examiner["confirmed"] and examiner["reason"]:
            reasons.append(examiner["reason"])
        if not examiner["ran"] and not examiner["error"]:
            reasons.append("the examiner check was not run")
        if independence_required and not independence_ok:
            reasons.append("the examiner was not independent of the reader")
        if not claim_level_ok:
            cl = examiner.get("claim_level") or {}
            if not cl.get("ran"):
                reasons.append("the examiner did not return claim-level "
                               "verdicts")
            else:
                labels = cl.get("labels") or []
                bad = [f"claim {i}={label}" for i, label in enumerate(labels)
                       if label and label != "supported"]
                reasons.append("not every claim is independently supported"
                               + (f" ({', '.join(bad)})" if bad else ""))
        if citation_ok is False:
            reasons.append("the cited text shares no content term with the "
                           "claim, so it is not specific evidence")
        if not quote_on_page:
            if citation_conflict:
                reasons.append("the reader's citation did not resolve to the "
                               "retrieved pages")
            elif not citations_valid:
                reasons.append("the answer has no resolvable citation on the "
                               "retrieved pages")
            else:
                reasons.append("the quote is not verbatim on the cited page")
        for w in (details.get("warnings") or [])[:2]:
            reasons.append(w)
        for w in support_warnings[:1]:
            reasons.append(w)
    state_reason = "; ".join(dict.fromkeys(r for r in reasons if r))

    # P5: a separate visual-verification tier. The examiner looked at the
    # region crop and confirmed the claim, but the text citation/quote did not
    # resolve. Never mixed with text-verified counts.
    visual_verified = bool(
        not nothing_found and region_present and examiner["ran"]
        and examiner["confirmed"] and not quote_on_page and claim_level_ok
        and examiner["evidence_mode"] in ("region_crop", "answer_crop",
                                          "claim_candidate", "sentence_crop"))
    tier = ("text" if verified
            else "visual" if visual_verified else "none")

    # final evidence boxes = cited sentences only (#1, #4, #6; Phase 2 across pages)
    cited_sentences: list[dict] = []
    final_boxes: list[tuple[int, int, int, int]] = []
    final_frac: list[tuple[float, float, float, float]] = []
    pages_evidence: dict = {}
    loc_score = 0.0
    loc_words: list[WordBox] = []
    loc_source = "none"
    evidence_valid = False
    if not nothing_found:
        if cited_texts and citation_info["status"] in ("ok", "partial"):
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
        elif region_present:
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

    # Claim-level display: the selected document candidate is the source when
    # it is a supported claim on the primary page. Otherwise it is exposed as
    # an overlay, never silently substituted for the reader's citation.
    claim_boxes_out = []
    for _ci, cr in enumerate(claim_regions):
        best = cr.get("best")
        if not best or not best.get("box"):
            continue
        claim_boxes_out.append({
            "claim_idx": _ci,
            "text": cr.get("text", ""),
            "label": cr.get("label", ""),
            "doc_id": best.get("doc_id"),
            "page": best.get("page"),
            "box": list(best.get("box") or []),
            "box_frac": list(best.get("box_frac") or []),
            "score": cr.get("best_score", 0.0),
        })
    display_source = loc_source
    claim_display = next(
        (cr for cr in claim_regions if cr.get("label") == "supported"
         and cr.get("best")), None)
    if (state == "verified" and claim_display is not None
            and claim_display["best"].get("doc_id") == doc_id
            and int(claim_display["best"].get("page", -1)) == page
            and claim_display["best"].get("box")):
        cb = claim_display["best"]
        final_boxes = [tuple(cb["box"])]
        final_frac = [tuple(cb["box_frac"])]
        evidence_valid = True
        loc_score = max(float(loc_score), float(cb.get("score", 1.0)))
        display_source = "claim_region"

    # Phase 4 features: citation lands on the retrieved top-1 page; visual gap
    top1_key = (ranked[0][0], ranked[0][1])
    cited_is_top1 = (any((d, p) == top1_key for d, p, _s in cited_refs)
                     if cited_refs else False)
    if vis_scores:
        _vs = sorted(vis_scores.values(), reverse=True)
        visual_margin = float(_vs[0] - _vs[1]) if len(_vs) > 1 else 0.0
    else:
        visual_margin = 0.0

    # Run 9 features: citation resolution + examiner verdict + details
    citation_resolved = {"none": 1.0, "ok": 1.0,
                         "partial": 0.5}.get(citation_info["status"], 0.0)
    if support_method.endswith("+type"):
        support_type = 1.0
    elif support_method == "no_citation":
        support_type = 0.0
    else:
        support_type = 0.5
    ex_supported = verdict_supported_value(
        verdict if verdict is not None else examiner)
    details_ok = 1.0 if details.get("ok", True) else 0.0

    feats = build_features(agreement=agr, entail_min=entail_min, retrieval_margin=margin,
                           heat_concentration=heat_conc, n_claims=len(samples[0].claims),
                           crop_has_text=has_text,
                           localize_score=loc_score, evidence_valid=evidence_valid,
                           quote_match=qscore,
                           answer_supported=supported,
                           cited_is_top1=cited_is_top1,
                           visual_margin=visual_margin,
                           citation_resolved=citation_resolved,
                           support_type=support_type,
                           examiner_ran=bool(examiner["ran"]),
                           examiner_supported=ex_supported,
                           quote_seen_match=float(examiner["quote_seen_match"]),
                           details_ok=details_ok)
    from vvrag.gate.scorer import features_to_matrix, hand_weighted

    # confidence: calibrated scorer when available, else hand-weighted (#5)
    scorer, cal_tau = load_gate_artifacts(cfg)
    score = hand_weighted(agr, entail_min, heat_conc, loc_score, evidence_valid,
                          qscore, examiner_ran=bool(examiner["ran"]),
                          examiner_supported=ex_supported,
                          quote_seen_match=float(examiner["quote_seen_match"]),
                          details_ok=details_ok)
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
    # Run 9: the gate no longer decides abstention. Nothing found is the only
    # abstention; every other answer is shown with its state + reason. The
    # score/tau remain displayed confidence (s7 rules decide VERIFIED).
    abstained = state == "nothing_found"
    abstain_reason = state_reason if abstained else ""

    passages = closest_passages(question, sentences) if abstained else []

    out_cfg = getattr(cfg, "output", None)
    low_suffix = str(getattr(out_cfg, "low_confidence_suffix",
                             "(not confident enough)"))
    nothing_text = str(getattr(out_cfg, "nothing_found_text",
                               "Nothing found in this document."))
    state_labels = {
        "verified": str(getattr(out_cfg, "verified_label", "Verified")),
        "not_confident": str(getattr(out_cfg, "not_confident_label",
                                     "Not confident enough")),
        "nothing_found": str(getattr(out_cfg, "nothing_found_label",
                                     "Nothing found")),
    }
    try:
        reader_served_by = getattr(reader, "served_by", None)
    except NameError:  # pragma: no cover - reader always bound here
        reader_served_by = None

    return {
        "answer": answer,
        "abstained": abstained,
        "abstain_reason": abstain_reason,
        "not_found": abstained,
        "verified": verified,
        "visual_verified": visual_verified,
        "state": state,
        "state_label": state_labels.get(state, state),
        "state_reason": state_reason,
        "not_confident": state == "not_confident",
        "low_confidence_suffix": low_suffix,
        "nothing_found_text": nothing_text,
        "tier": tier,
        "citations": {"status": citation_info["status"],
                      "n_refs": citation_info["n_refs"],
                      "n_resolved": citation_info["n_resolved"],
                      "unresolved": citation_info["unresolved"],
                      "derived_from_quote": bool(
                          citation_info.get("derived_from_quote"))},
        "visual_confirm": {"attempted": bool(examiner["ran"]),
                           "method": examiner["served_by"],
                           "verdict": bool(examiner["confirmed"]),
                           "detail": examiner["problem"] or examiner["reason"]},
        "examiner": examiner,
        "citation_check": {"ok": citation_ok,
                           "score": round(float(citation_score), 4),
                           "shared_terms": citation_terms[:10],
                           "numeric_sink": bool(numeric_sink)},
        "index_manifest": index_manifest,
        "index_manifest_mismatches": index_manifest_mismatches,
        "claim_regions": claim_regions,
        "claim_level": examiner.get("claim_level", {}),
        "examiner_served_by": examiner["served_by"],
        "examiner_ran": bool(examiner["ran"]),
        "independence_ok": bool(independence_ok),
        "independence_required": bool(independence_required),
        "reader_served_by": reader_served_by or f"{provider}/{model or ''}",
        "details": details,
        "support": {"score": round(float(support_score), 4),
                    "method": support_method,
                    "require": require_support,
                    "warnings": support_warnings,
                    "supported": bool(supported)},
        "score": round(score, 4),
        "score_display_only": True,
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
                     "sentence_ids": [sid for _d, _p, sid in cited_refs],
                     "refs": [list(r) for r in cited_refs],
                     "pages": list(pages_evidence.values()),
                     "region_boxes": [list(b) for b in region_boxes],
                     "crop_text": crop_text, "crop_has_text": has_text,
                     "heat_concentration": round(heat_conc, 4),
                     "claim_boxes": claim_boxes_out,
                     "claim_display": claim_display,
                     "display_source": display_source,
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
                     "region": s.region,
                     "no_evidence": s.no_evidence,
                     "found_in": s.found_in,
                     "confidence": s.confidence} for s in samples],
        "features": feats.to_dict(),
        "api_calls": api_calls,
        "timings": {"total": round(time.time() - t0, 3), "reader": round(t_r, 3)},
    }
