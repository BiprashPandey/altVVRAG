"""Qdrant + SQLite writers (Section 6.5). Idempotent: re-running skips done pages.

- Qdrant local mode (default): two collections —
  `pages_full` (one multivector point per page, MaxSim comparator),
  `pages_pooled` (one point per pooled vector, payload {doc_id, page, kind, idx}).
- SQLite: `pages` (PageMeta) + `word_boxes` tables.
Verified against qdrant-client 1.19.1 (:memory: multivector upsert+query tested).
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import numpy as np

from vvrag.logging_utils import get_logger
from vvrag.schemas import PageMeta, WordBox

log = get_logger(__name__)


# ---------------------------------------------------------------- SQLite ---

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages(
  doc_id TEXT, page INTEGER,
  width INTEGER, height INTEGER, dpi INTEGER,
  grid_h INTEGER, grid_w INTEGER,
  has_text_layer INTEGER, image_path TEXT,
  PRIMARY KEY (doc_id, page));
CREATE TABLE IF NOT EXISTS word_boxes(
  doc_id TEXT, page INTEGER, text TEXT,
  x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,
  block_id INTEGER, line_id INTEGER);
CREATE INDEX IF NOT EXISTS idx_wb_docpage ON word_boxes(doc_id, page);
"""


def connect_sqlite(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con


def upsert_page(con: sqlite3.Connection, meta: PageMeta) -> None:
    con.execute(
        "INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?,?,?,?)",
        (meta.doc_id, meta.page, meta.width, meta.height, meta.dpi,
         meta.grid_h, meta.grid_w, int(meta.has_text_layer), meta.image_path),
    )
    con.commit()


def replace_word_boxes(con: sqlite3.Connection, boxes: list[WordBox]) -> None:
    if not boxes:
        return
    keys = {(b.doc_id, b.page) for b in boxes}
    for doc_id, page in keys:
        con.execute("DELETE FROM word_boxes WHERE doc_id=? AND page=?", (doc_id, page))
    con.executemany(
        "INSERT INTO word_boxes VALUES (?,?,?,?,?,?,?,?,?)",
        [(b.doc_id, b.page, b.text, *b.box, b.block_id, b.line_id) for b in boxes],
    )
    con.commit()


def count_pages(con: sqlite3.Connection) -> int:
    return con.execute("SELECT COUNT(*) FROM pages").fetchone()[0]


def count_word_boxes(con: sqlite3.Connection) -> int:
    return con.execute("SELECT COUNT(*) FROM word_boxes").fetchone()[0]


def indexed_pages(con: sqlite3.Connection) -> set[tuple[str, int]]:
    return {(r[0], r[1]) for r in con.execute("SELECT doc_id, page FROM pages")}


# ---------------------------------------------------------------- Qdrant ---

def _point_id(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join(parts)))


def get_qdrant_client(mode: str, path: str = "data/qdrant",
                      url: str = "", api_key: str = ""):
    from qdrant_client import QdrantClient

    if mode == "local":
        Path(path).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=path)
    if mode == "cloud":
        if not url:
            raise ValueError("qdrant cloud mode needs QDRANT_URL")
        return QdrantClient(url=url, api_key=api_key or None)
    raise ValueError(f"unknown qdrant mode: {mode}")


def ensure_collections(client, dim: int, full: str, pooled: str) -> None:
    from qdrant_client import models

    if not client.collection_exists(full):
        client.create_collection(
            full,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )
    if not client.collection_exists(pooled):
        client.create_collection(
            pooled,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )


def upsert_page_vectors(
    client,
    doc_id: str,
    page: int,
    patch_vectors: np.ndarray,
    pooled: np.ndarray,
    pooled_kinds: list[str],
    full: str,
    pooled_name: str,
) -> None:
    """Idempotent upsert (deterministic point ids)."""
    from qdrant_client import models

    patch_vectors = np.asarray(patch_vectors, dtype=np.float32)
    pooled = np.asarray(pooled, dtype=np.float32)
    client.upsert(
        full,
        [models.PointStruct(id=_point_id(doc_id, str(page), "full"),
                            vector=patch_vectors.tolist(),
                            payload={"doc_id": doc_id, "page": page})],
    )
    pts = [
        models.PointStruct(
            id=_point_id(doc_id, str(page), kind, str(i)),
            vector=pooled[i].tolist(),
            payload={"doc_id": doc_id, "page": page, "kind": kind, "idx": i},
        )
        for i, kind in enumerate(pooled_kinds)
    ]
    client.upsert(pooled_name, pts)


def pooled_kind_labels(grid_h: int, grid_w: int, include_global: bool = True) -> list[str]:
    labels = [f"row{i}" for i in range(grid_h)] + [f"col{i}" for i in range(grid_w)]
    if include_global:
        labels.append("global")
    return labels


def count_points(client, collection: str) -> int:
    return client.count(collection).count
