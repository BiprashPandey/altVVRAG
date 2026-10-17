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


def load_slidevqa(index_dir: str = "data/index", split: str = "debug",
                  n_debug: int = 45, seed: int = 42) -> EvalSet:
    """Primary dataset loader. Tries HF mirror, else raises with instructions.

    Set VVRAG_DATASET=slidevqa and place files per scripts/download_data.sh.
    Falls back to build_synthetic_set() when unavailable (documented in runs).
    """
    from datasets import load_dataset  # type: ignore

    for cand in ("lmms-lab/SlideVQA", "SlideVQA/SlideVQA"):
        try:
            ds = load_dataset(cand, split="test")
            break
        except Exception:
            continue
    else:
        raise RuntimeError(
            "SlideVQA unavailable on the Hub tried {lmms-lab/SlideVQA, SlideVQA/SlideVQA}. "
            "Run scripts/download_data.sh or use build_synthetic_set() fallback.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_debug, len(ds)), replace=False)
    items = []
    for i in idx:
        row = ds[int(i)]
        items.append(QAItem(question_id=str(row.get("question_id", i)),
                            deck_id=str(row.get("deck_id", row.get("deck", "d0"))),
                            question=str(row["question"]),
                            gold_answer=str(row["answer"]),
                            gold_doc=str(row.get("doc_id", "")),
                            gold_page=int(row.get("page", 0))))
    return EvalSet(items=items, index_dir=index_dir)
