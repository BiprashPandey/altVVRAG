"""Ad-hoc Q&A for the localhost UI (no gold answers, no query encoder).

Honest demo-path disclosure (also shown in the UI):
- Retrieval is KEYWORD overlap over SQLite word boxes (not MaxSim: the query
  encoder only runs on the GPU box; see notebooks/ingest_colab.ipynb).
- Grounding default is keyword-SPAN boxes (exact+fuzzy, same code as
  eval/gold_regions.py). The "heatmap demo" toggle seeds the REAL C1 heatmap
  code from the keyword hit cells so you can see boxes/crops/overlays work;
  real queries use ColQwen query vectors instead.
- Reader is either live (Gemini/Groq/OpenRouter, needs key) or mock-canned.
- Verification (agreement + NLI on crop text) and gates are the REAL code.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from vvrag.eval.gold_regions import derive_gold_box
from vvrag.ground.boxes import grid_to_boxes
from vvrag.ground.crop import crop_union, overlay_heatmap
from vvrag.ground.heatmap import build_heatmap, heat_concentration
from vvrag.ground.text_in_box import extract_crop_text, group_lines
from vvrag.ingest.embed import load_embedding
from vvrag.logging_utils import get_logger
from vvrag.reader.claims import normalize_answer
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
                 limit: int = 5) -> list[tuple[str, int, float, list[WordBox]]]:
    """Rank pages by keyword overlap. Returns (doc, page, score, words)."""
    toks = set(content_tokens(question))
    if not toks:
        toks = set(normalize_answer(question).split())
    pages = con.execute("SELECT doc_id, page FROM pages").fetchall()
    ranked = []
    for doc_id, page in pages:
        words = page_words(con, doc_id, page)
        hit = sum(1 for w in words if normalize_answer(w.text) in toks)
        ranked.append((doc_id, page, hit / max(1, len(toks)), words))
    ranked.sort(key=lambda r: r[2], reverse=True)
    return ranked[:limit]


def span_box(question: str, words: list[WordBox]) -> tuple[int, int, int, int] | None:
    """Evidence box from keyword word-hits: union of hit lines (full lines)."""
    toks = set(content_tokens(question))
    hits = [w for w in words if normalize_answer(w.text) in toks]
    if not hits:
        # fuzzy fallback: reuse gold-region machinery on the longest keyword
        toks_sorted = sorted(toks, key=len, reverse=True)
        for t in toks_sorted[:3]:
            box = derive_gold_box(t, words)
            if box:
                return box
        return None
    hit_ids = {id(w) for w in hits}
    boxes = []
    for line in group_lines(words):
        if any(id(w) in hit_ids for w in line):
            boxes.extend(w.box for w in line)
    if not boxes:
        return None
    # cap: keep the 3 longest lines to avoid whole-page boxes
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def heatmap_box_from_span(question: str, words: list[WordBox], page_vecs: np.ndarray,
                          grid_h: int, grid_w: int, page_w: int, page_h: int,
                          mode: str = "max") -> tuple[list[tuple], float, np.ndarray] | tuple[None, None, None]:
    """Demo C1 path: cells under keyword hits -> pseudo-query -> real heatmap/boxes."""
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
    _, norm = build_heatmap(q, labels, page_vecs, grid_h, grid_w, mode=mode)
    boxes, mask = grid_to_boxes(norm, page_w, page_h)
    return boxes, heat_concentration(norm, mask), norm


def answer_question(question: str, index_dir: str, work_dir: str,
                    reader_mode: str = "mock", provider: str = "gemini", model: str = "",
                    n_samples: int = 3, temperature: float = 0.0,
                    grounding: str = "span", gate: str = "none",
                    fixed_tau: float = 0.5) -> dict:
    """Full ad-hoc pipeline -> JSON-able result dict."""
    t0 = time.time()
    index_dir, work_dir = Path(index_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(index_dir / "meta.db")

    ranked = keyword_rank(question, con)
    if not ranked or ranked[0][2] <= 0:
        return {"answer": "UNANSWERABLE", "abstained": True, "reason": "no keyword hit",
                "score": 0.0, "timings": {"total": time.time() - t0}}
    doc_id, page, rank_score, words = ranked[0]
    meta = con.execute(
        "SELECT width,height,grid_h,grid_w,image_path FROM pages WHERE doc_id=? AND page=?",
        (doc_id, page)).fetchone()
    W, H, gh, gw, img_path = meta

    boxes, heat_conc, norm = None, 0.0, None
    if grounding == "heatmap":
        try:
            pe = load_embedding(index_dir / "embeddings", doc_id, page)
            boxes, heat_conc, norm = heatmap_box_from_span(
                question, words, pe.vectors, gh or pe.grid_h, gw or pe.grid_w, W, H)
        except Exception as e:
            log.warning("heatmap demo failed, falling back to span: %s", e)
    if not boxes:
        single = span_box(question, words)
        boxes = [single] if single else [(0, 0, W, H)]
        heat_conc = 0.0

    from vvrag.schemas import Evidence

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"q{int(time.time()*1000)%100000000}")
    crop_path = str(work_dir / f"{safe}_crop.png")
    ov_path = str(work_dir / f"{safe}_overlay.png")
    crop_union(img_path, boxes, crop_path)
    if norm is not None:
        overlay_heatmap(img_path, norm, boxes, ov_path)
    else:
        overlay_heatmap(img_path, np.zeros((gh or 8, gw or 6), dtype=np.float32), boxes, ov_path)
    crop_text, has_text = extract_crop_text(words, boxes, crop_path)

    # reader
    t_r = time.time()
    if reader_mode == "mock":
        from vvrag.reader.client import MockReader

        reader: object = MockReader()
        images: list = []
    else:
        from vvrag.cache import DiskCache
        from vvrag.config import load_config
        from vvrag.ratelimit import RateLimiter
        from vvrag.reader.client import create_reader

        cfg = load_config("config.yaml")
        model = model or cfg.reader.model
        reader = create_reader(provider, model, DiskCache(cfg.cache_dir),
                               RateLimiter(cfg.ratelimit.requests_per_minute))
        images = [img_path, crop_path]
    from vvrag.reader.client import run_reader_samples
    from vvrag.reader.prompts import build_prompt

    prompt = build_prompt(question, crop_text if has_text else None, max_claims=5)
    samples, api_calls = run_reader_samples(reader, images, prompt, n_samples,
                                           temperature if n_samples > 1 else 0.0)
    t_r = time.time() - t_r

    from vvrag.verify.consistency import agreement
    from vvrag.verify.features import build_features
    from vvrag.verify.nli import aggregate_entailment

    agr, _ = agreement(samples)
    claim_probs = []
    if has_text:
        nli = get_nli()
        for c in samples[0].claims:
            claim_probs.extend(nli.entail_probs(crop_text, [c.text]))
    emin, _ = aggregate_entailment(claim_probs) if claim_probs else (None, None)
    feats = build_features(agreement=agr, entail_min=emin, retrieval_margin=0.0,
                           heat_concentration=heat_conc, n_claims=len(samples[0].claims),
                           crop_has_text=has_text)
    from vvrag.gate.fixed import apply_fixed
    from vvrag.gate.scorer import hand_weighted

    score = hand_weighted(agr, emin, heat_conc)
    abstained = (not apply_fixed(score, fixed_tau)) if gate == "fixed" else False

    return {
        "answer": samples[0].answer,
        "abstained": abstained,
        "score": round(score, 4),
        "tau": fixed_tau if gate == "fixed" else 0.0,
        "retrieval": [{"doc_id": d, "page": p, "keyword_score": round(s, 3)}
                      for d, p, s, _ in ranked],
        "evidence": {"doc_id": doc_id, "page": page,
                     "boxes": [list(b) for b in boxes],
                     "crop_text": crop_text, "crop_has_text": has_text,
                     "heat_concentration": round(heat_conc, 4),
                     "crop_url": "/media/" + Path(crop_path).name,
                     "overlay_url": "/media/" + Path(ov_path).name},
        "claims": [c.to_dict() for c in samples[0].claims],
        "samples": [{"answer": s.answer} for s in samples],
        "features": feats.to_dict(),
        "api_calls": api_calls,
        "timings": {"total": round(time.time() - t0, 3), "reader": round(t_r, 3)},
    }
