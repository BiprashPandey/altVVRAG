"""Eval datasets (Section 10.1): SlideVQA primary, synthetic debug fallback.

SlideVQA download requires the GPU/key environment (see
scripts/download_data.sh); the SyntheticDebugSet below builds small PDFs with
known answer facts so the FULL machinery (retrieve -> ground -> verify ->
gate) runs offline on CPU with a noisy simulated reader (eval/arms.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class QAItem:
    question_id: str
    deck_id: str
    question: str
    gold_answer: str
    gold_doc: str
    gold_page: int
    gold_patch_idx: tuple[int, ...] = ()  # flat patch indices tied to answer location
    is_unanswerable: bool = False  # suggestion #8: gold says answer is absent
    acceptable_answers: list[str] = field(default_factory=list)  # P3: equivalent golds

    def golds(self) -> list[str]:
        """All acceptable gold strings (primary first, de-duplicated)."""
        out, seen = [], set()
        for a in [self.gold_answer, *self.acceptable_answers]:
            if a and a not in seen:
                seen.add(a)
                out.append(a)
        return out


@dataclass
class EvalSet:
    items: list[QAItem] = field(default_factory=list)
    index_dir: str = ""  # built index for this set (synthetic)


def deck_split(items: list[QAItem], train_frac: float = 0.3, calib_frac: float = 0.3,
               seed: int = 42) -> tuple[list[QAItem], list[QAItem], list[QAItem]]:
    """Split by DECK (not question) to avoid leakage."""
    rng = np.random.default_rng(seed)
    decks = sorted({it.deck_id for it in items})
    order = rng.permutation(decks)
    n = len(order)
    n_train = max(1, int(n * train_frac))
    n_calib = max(1, int(n * calib_frac))
    train_d = set(order[:n_train])
    calib_d = set(order[n_train:n_train + n_calib])
    train = [it for it in items if it.deck_id in train_d]
    calib = [it for it in items if it.deck_id in calib_d]
    test = [it for it in items if it.deck_id not in train_d | calib_d]
    return train, calib, test


_ENTITIES = ["Aurora", "Beacon", "Comet", "Drift", "Ember", "Flint", "Grove", "Harbor"]
_FACTS = ["revenue was {v} million", "budget totals {v} million",
          "load reached {v} percent", "output hit {v} units"]


def build_synthetic_set(out_dir: str | Path, n_docs: int = 10, pages_per_doc: int = 2,
                        seed: int = 42, dpi: int = 150,
                        unanswerable_per_doc: int = 1) -> EvalSet:
    """Create PDFs with planted facts + questions. Answers are exact page strings.

    Planting (ties text location to patch cells, so C1 grounding is genuinely
    tested): each fact sentence is rendered inside a known 2x2 patch-cell block
    of the FakePatchEmbedder grid; its flat patch indices are stored on the
    item and used to build the query (arms.query_vecs).

    `unanswerable_per_doc` also plants questions about facts that are NOT in
    the deck (suggestion #8: the gate must abstain on these).
    """
    import pymupdf

    from vvrag.ingest.embed import FakePatchEmbedder
    from vvrag.schemas import grid_cell_range_to_pixel_box

    fake = FakePatchEmbedder()
    gh, gw = fake.grid_h, fake.grid_w
    PW_PT, PH_PT = 600.0, 800.0
    W_PX, H_PX = int(PW_PT * dpi / 72), int(PH_PT * dpi / 72)
    # 3 fact blocks per page: (r0, c0, r1, c1); header top, fillers bottom
    BLOCKS = [(1, 1, 3, 3), (3, 3, 5, 5), (5, 1, 7, 3)]

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    items: list[QAItem] = []
    for d in range(n_docs):
        doc_id = f"deck{d:02d}"
        pdf = raw / f"{doc_id}.pdf"
        with pymupdf.open() as doc:
            for p in range(pages_per_doc):
                page = doc.new_page(width=PW_PT, height=PH_PT)
                page.insert_text((50, 40), f"Quarterly report {doc_id} page {p}.")
                for f_i in range(3):
                    ent = rng.choice(_ENTITIES)
                    fact = rng.choice(_FACTS).format(v=int(rng.integers(10, 99)))
                    sent = f"The {ent} {fact} in this quarter."
                    r0, c0, r1, c1 = BLOCKS[f_i]
                    x0, y0, _, _ = grid_cell_range_to_pixel_box(
                        r0, c0, r1, c1, W_PX, H_PX, gh, gw)
                    page.insert_text((x0 * 72 / dpi + 4, y0 * 72 / dpi + 16), sent)
                    gold = sent.split("The ", 1)[1].split(" in this", 1)[0]
                    idx = tuple(r * gw + c for r in range(r0, r1) for c in range(c0, c1))
                    items.append(QAItem(
                        question_id=f"{doc_id}-p{p}-f{f_i}", deck_id=doc_id,
                        question=f"What is reported for {ent} on {doc_id} page {p} fact {f_i}?",
                        gold_answer=gold, gold_doc=doc_id, gold_page=p,
                        gold_patch_idx=idx))
                for j in range(4):  # filler at bottom, clear of fact blocks
                    page.insert_text((50, 730 + j * 18), "Supplementary notes follow.")
            for u in range(unanswerable_per_doc):
                # a plausible-but-absent fact: the gate must abstain (#8)
                items.append(QAItem(
                    question_id=f"{doc_id}-una{u}", deck_id=doc_id,
                    question=f"What was the {doc_id} liability figure on page 0?",
                    gold_answer="UNANSWERABLE", gold_doc=doc_id, gold_page=0,
                    is_unanswerable=True))
            doc.save(pdf)
    return EvalSet(items=items)


def _slidevqa_item(row, i: int) -> QAItem:
    """Map one SlideVQA row -> QAItem (P3: equivalent golds, 0-indexed page)."""
    q = str(row.get("question", "")).strip()
    ans_raw = row.get("answer", [])
    variants = [str(a).strip() for a in (ans_raw if isinstance(ans_raw, list) else [ans_raw])
                if str(a).strip()]
    ev = row.get("evidence_pages")
    if ev is None and isinstance(row.get("evidence"), dict):
        ev = row["evidence"].get("pages")
    ev = ev or [1]
    ev_pages = [int(p) for p in ev] if isinstance(ev, (list, tuple)) else [int(ev)]
    gold_page = max(0, (ev_pages[0] if ev_pages else 1) - 1)  # 1-indexed -> 0-indexed
    deck = str(row.get("deck_name")
               or (row.get("document", {}) or {}).get("id") or f"d{i}")
    is_unans = (not variants) or any(
        v.lower() in ("unanswerable", "not answerable", "n/a") for v in variants)
    return QAItem(
        question_id=f"{deck}-{row.get('qa_id', i)}", deck_id=deck, question=q,
        gold_answer=variants[0] if variants else "UNANSWERABLE",
        gold_doc=deck, gold_page=gold_page,
        acceptable_answers=variants[1:], is_unanswerable=is_unans)


def load_slidevqa(index_dir: str = "data/slidevqa/index", split: str = "test",
                  n_debug: int = 45, seed: int = 42, streaming: bool = True,
                  max_scan: int = 200) -> EvalSet:
    """SlideVQA loader (P3/C): real rows -> QAItems with equivalent golds.

    Tries the official gated repo first, then public mirrors. Each row is one
    deck (`deck_name`) with up to 20 slide images (`page_1..page_20`), a
    `question`, an `answer` LIST of acceptable variants, and 1-indexed
    `evidence_pages`. We map to 0-indexed gold_page and keep every answer
    variant as an acceptable gold (P3: multiple equivalent answers).

    Only the first `max_scan` rows are scanned (each row decodes 20 images,
    so full-iteration streaming is expensive); `seed` picks a sparse stride
    for deck diversity.
    """
    from datasets import load_dataset  # type: ignore

    repos = ("NTT-hil-insight/SlideVQA", "Ahren09/SlideVQA", "AHS-uni/slidevqa-qa")
    ds = None
    last_err = None
    for cand in repos:
        try:
            ds = load_dataset(cand, split=split, streaming=streaming)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if ds is None:
        raise RuntimeError(
            "SlideVQA unavailable on the Hub (tried "
            f"{', '.join(repos)}). Run scripts/prepare_slidevqa.py or fall back "
            "to build_synthetic_set(). Last error: %s" % last_err)

    stride = max(1, max_scan // max(1, n_debug))
    items: list[QAItem] = []
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        if i % stride != 0:
            continue
        item = _slidevqa_item(row, i)
        if not item.question:
            continue
        items.append(item)
        if len(items) >= n_debug:
            break
    return EvalSet(items=items, index_dir=index_dir)
