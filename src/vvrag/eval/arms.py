"""Eval arms (Section 10.3): Baseline / Grounded / Verified-fixed / Verified-conformal.

Offline machinery: REAL retrieval (two-stage MaxSim), REAL grounding
(heatmap/boxes/crop/crop-text), REAL agreement + NLI, REAL scorer + gates.
Stand-ins (documented): FakePatchEmbedder index, queries built from gold-page
patches + noise (retrieval is easy by construction), SimulatedReader with
seeded accuracy, SQLite word boxes for crop text / gold regions.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np

from vvrag.eval.datasets import QAItem
from vvrag.eval.gold_regions import derive_gold_box
from vvrag.eval.metrics import center_inside, exact_match, f1, hit_at_k, iou, mrr
from vvrag.gate.conformal import apply_gate as conformal_apply
from vvrag.gate.conformal import select_tau
from vvrag.gate.fixed import apply_fixed
from vvrag.gate.scorer import LogisticScorer, features_to_matrix, hand_weighted
from vvrag.ground.boxes import grid_to_boxes
from vvrag.ground.crop import crop_union
from vvrag.ground.heatmap import build_heatmap, heat_concentration
from vvrag.ground.text_in_box import extract_crop_text
from vvrag.ingest import index as index_mod
from vvrag.ingest.embed import load_embedding
from vvrag.logging_utils import get_logger
from vvrag.reader.claims import Claim
from vvrag.retrieve.two_stage import TwoStageRetriever, retrieval_margin
from vvrag.schemas import Evidence, Features, ReaderSample, Result, WordBox
from vvrag.verify.consistency import agreement
from vvrag.verify.features import build_features
from vvrag.verify.nli import FakeNLI, aggregate_entailment

log = get_logger(__name__)

ARMS = ["baseline", "grounded", "verified-fixed", "verified-conformal"]


def stable_seed(*parts: str) -> int:
    """Deterministic seed (Python hash() is salted per-process: never use it)."""
    import hashlib

    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


@dataclass
class ArmConfig:
    name: str = "verified-conformal"
    grounding: bool = True
    n_samples: int = 3
    heat_mode: str = "max"
    gate: str = "conformal"  # none | fixed | conformal
    scorer: str = "learned"  # learned | hand
    alpha: float = 0.10
    delta: float = 0.10
    fixed_tau: float = 0.5
    reader_acc: float = 0.7
    prefetch_mult: int = 4
    top_k: int = 3
    use_real_nli: bool = True


@dataclass
class PredRecord:
    item: QAItem
    result: Result
    correct_f1: float
    correct_em: float
    gold_box: tuple | None
    iou: float | None
    hit3: float
    mrr: float
    rank_pages: list[tuple[str, int]]


class SimulatedReader:
    """Seeded noisy reader: gold answer w.p. reader_acc, else a distractor.

    Distractors are sibling facts from the same doc (plausible wrong values),
    so NLI must discriminate contradiction from entailment.
    """

    def __init__(self, acc: float = 0.7, pool: dict[str, list[str]] | None = None):
        self.acc = acc
        self.pool = pool or {}

    def sample(self, item: QAItem, sample_id: int) -> ReaderSample:
        rng = np.random.default_rng(stable_seed(item.question_id, str(sample_id)))
        if rng.random() < self.acc:
            ans = item.gold_answer
        else:
            alts = [a for a in self.pool.get(item.gold_doc, []) if a != item.gold_answer]
            ans = str(rng.choice(alts)) if alts else "UNANSWERABLE"
        claim = Claim(text=f"{ans}.", answer_span=ans)
        raw = f'{{"answer": {ans!r}, "claims": [{{"text": {claim.text!r}}}]}}'
        return ReaderSample(sample_id=sample_id, answer=ans, claims=[claim], raw=raw)


class OfflineSystem:
    def __init__(self, index_dir: str, cfg: ArmConfig, work_dir: str = "results/eval_work"):
        import os

        self.index_dir = index_dir
        self.cfg = cfg
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.client = index_mod.get_qdrant_client("local", f"{index_dir}/qdrant")
        self.retriever = TwoStageRetriever(self.client, top_k=cfg.top_k,
                                           prefetch_k_multiplier=cfg.prefetch_mult)
        self.con = sqlite3.connect(f"{index_dir}/meta.db")
        self.reader = SimulatedReader(acc=cfg.reader_acc)
        self.nli = None
        self.scorer = LogisticScorer()
        self.tau = 0.0
        self.pages_cache: dict[tuple[str, int], tuple] = {}  # (meta, words)

    def set_distractor_pool(self, items: list[QAItem]) -> None:
        pool: dict[str, list[str]] = {}
        for it in items:
            pool.setdefault(it.gold_doc, []).append(it.gold_answer)
        self.reader.pool = pool

    # ------------------------------------------------------------- helpers ---
    def query_vecs(self, item: QAItem, variant: int = 0, noise: float = 0.15) -> np.ndarray:
        """Query = perturbed copies of the answer-region patches (planted cells).

        Falls back to random page patches when the item carries no planting.
        """
        pe = load_embedding(f"{self.index_dir}/embeddings", item.gold_doc, item.gold_page)
        rng = np.random.default_rng(stable_seed("q" + item.question_id, str(variant)))
        if item.gold_patch_idx:
            idx = np.array(item.gold_patch_idx)
        else:
            idx = rng.choice(pe.vectors.shape[0], size=4, replace=False)
        q = pe.vectors[idx] + noise * rng.standard_normal((len(idx), pe.dim)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        return q.astype(np.float32)

    def page_data(self, doc_id: str, page: int):
        key = (doc_id, page)
        if key not in self.pages_cache:
            row = self.con.execute(
                "SELECT width,height,dpi,grid_h,grid_w,has_text_layer,image_path FROM pages "
                "WHERE doc_id=? AND page=?", (doc_id, page)).fetchone()
            wrows = self.con.execute(
                "SELECT text,x0,y0,x1,y1,block_id,line_id FROM word_boxes "
                "WHERE doc_id=? AND page=?", (doc_id, page)).fetchall()
            words = [WordBox(doc_id, page, r[0], (r[1], r[2], r[3], r[4]), r[5], r[6])
                     for r in wrows]
            self.pages_cache[key] = (row, words)
        return self.pages_cache[key]

    def ensure_nli(self):
        if self.nli is None:
            if self.cfg.use_real_nli:
                from vvrag.verify.nli import NLIEntailer

                self.nli = NLIEntailer()
                self.nli.load()
            else:
                self.nli = FakeNLI()

    # ------------------------------------------------------------------ run ---
    def answer_item(self, item: QAItem) -> PredRecord:
        t0 = time.time()
        q = self.query_vecs(item)
        cands = self.retriever.prefetch(q)
        rank_pages = [(d, p) for d, p, _ in cands]
        hits = self.retriever.rerank(q, cands)[: self.cfg.top_k]
        top = hits[0] if hits else None
        margin = retrieval_margin(hits) if len(hits) > 1 else 0.0

        evidence = Evidence(doc_id=item.gold_doc, page=item.gold_page)
        crop_text, heat_conc = "", 0.0
        if self.cfg.grounding and top is not None:
            (meta_row, words) = self.page_data(top.doc_id, top.page)
            W, H, gh, gw, img_path = meta_row[0], meta_row[1], meta_row[3], meta_row[4], meta_row[6]
            pe = load_embedding(f"{self.index_dir}/embeddings", top.doc_id, top.page)
            toks = [t for t in item.question.lower().split() if t][:8] or ["q"]
            qv = q[: len(toks)] if q.shape[0] >= len(toks) else np.tile(q, (2, 1))[: len(toks)]
            hm, norm = build_heatmap(qv, toks, pe.vectors, gh, gw,
                                     top.doc_id, top.page, mode=self.cfg.heat_mode)
            boxes, mask = grid_to_boxes(norm, W, H)
            heat_conc = heat_concentration(norm, mask)
            if boxes:
                safe_qid = "".join(c if c.isalnum() or c in "-_." else "_" for c in item.question_id)
                crop_path = f"{self.work_dir}/{safe_qid}.png"
                crop_union(img_path, boxes, crop_path)
                evidence = Evidence(top.doc_id, top.page, boxes, crop_path, "", heat_conc)
                crop_text, has = extract_crop_text(words, boxes, crop_path)
                evidence.crop_text = crop_text
            else:
                evidence = Evidence(top.doc_id, top.page, [], "", "", heat_conc)

        samples = [self.reader.sample(item, s) for s in range(self.cfg.n_samples)]
        agr, _ = agreement([s for s in samples])
        self.ensure_nli()
        claim_probs = []
        for s in samples[:1]:
            for c in s.claims:
                if evidence.crop_text:
                    claim_probs.extend(self.nli.entail_probs(evidence.crop_text, [c.text]))
        emin, emean = aggregate_entailment(claim_probs) if claim_probs else (None, None)
        feats = build_features(agreement=agr, entail_min=emin,
                               retrieval_margin=margin, heat_concentration=heat_conc,
                               n_claims=len(samples[0].claims),
                               crop_has_text=bool(evidence.crop_text))
        if self.cfg.scorer == "learned":
            try:
                score = float(self.scorer.predict_score(features_to_matrix([feats]))[0])
            except RuntimeError:
                score = hand_weighted(agr, emin, heat_conc)  # unfitted fallback
        else:
            score = hand_weighted(agr, emin, heat_conc)

        if self.cfg.gate == "none":
            abstained = False
        elif self.cfg.gate == "fixed":
            abstained = not apply_fixed(score, self.cfg.fixed_tau)
        else:
            abstained = not conformal_apply(score, self.tau)

        res = Result(question=item.question, answer=samples[0].answer, abstained=abstained,
                     score=score, tau=self.tau if self.cfg.gate == "conformal" else self.cfg.fixed_tau,
                     evidence=evidence, samples=samples, features=feats,
                     timings={"total": time.time() - t0}, api_calls=self.cfg.n_samples)
        cf1, cem = f1(res.answer, item.gold_answer), exact_match(res.answer, item.gold_answer)
        (_, gold_words) = self.page_data(item.gold_doc, item.gold_page)
        gold_box = derive_gold_box(item.gold_answer, gold_words)
        iou_v = iou(evidence.boxes[0], gold_box) if evidence.boxes and gold_box else None
        return PredRecord(item=item, result=res, correct_f1=cf1, correct_em=cem,
                          gold_box=gold_box, iou=iou_v,
                          hit3=hit_at_k(rank_pages, (item.gold_doc, item.gold_page), 3),
                          mrr=mrr(rank_pages, (item.gold_doc, item.gold_page)),
                          rank_pages=rank_pages)

    # ------------------------------------------------------- train/gate fit ---
    def fit_scorer(self, records: list[PredRecord], f1_threshold: float = 0.5) -> None:
        X = features_to_matrix([r.result.features for r in records])
        y = np.array([1 if r.correct_f1 >= f1_threshold else 0 for r in records])
        if len(set(y.tolist())) < 2:
            log.warning("only one class in scorer train; keeping hand-weighted fallback")
            return
        self.scorer.fit(X, y)

    def rescore(self, records: list[PredRecord]) -> np.ndarray:
        X = features_to_matrix([r.result.features for r in records])
        try:
            s = self.scorer.predict_score(X)
        except RuntimeError:
            s = np.array([hand_weighted(f.agreement, f.entail_prob, f.heat_concentration)
                          for f in (r.result.features for r in records)])
        for r, v in zip(records, s):
            r.result.score = float(v)
        return s

    def calibrate(self, records: list[PredRecord], min_n: int | None = None) -> float:
        scores = np.array([r.result.score for r in records])
        errors = np.array([0 if r.correct_f1 >= 0.5 else 1 for r in records])
        if min_n is None:
            # scale the certification floor with calib size (unit-test default
            # min_n=30 suits n_calib >= hundreds; tiny debug calib needs less)
            min_n = max(5, len(records) // 3)
        self.tau, _ = select_tau(scores, errors, self.cfg.alpha, self.cfg.delta,
                                 min_n=min_n)
        return self.tau

    def apply_gate_records(self, records: list[PredRecord]) -> None:
        for r in records:
            if self.cfg.gate == "none":
                r.result.abstained = False
            elif self.cfg.gate == "fixed":
                r.result.abstained = not apply_fixed(r.result.score, self.cfg.fixed_tau)
            else:
                r.result.abstained = not conformal_apply(r.result.score, self.tau)
                r.result.tau = self.tau
