"""Retrieval-never-abstains + closest-passages tests (suggestions #3, #9)."""
from vvrag.ingest.index import connect_sqlite, replace_word_boxes, upsert_page
from vvrag.schemas import PageMeta, SentenceSpan, WordBox
from vvrag.ui.qa import closest_passages, keyword_rank


def _db(tmp_path):
    con = connect_sqlite(tmp_path / "meta.db")
    upsert_page(con, PageMeta("d", 0, 600, 800, grid_h=4, grid_w=4, image_path="p0.png"))
    upsert_page(con, PageMeta("d", 1, 600, 800, grid_h=4, grid_w=4, image_path="p1.png"))
    replace_word_boxes(con, [
        WordBox("d", 0, "Aurora", (0, 0, 50, 20)),
        WordBox("d", 0, "revenue", (55, 0, 120, 20)),
        WordBox("d", 0, "42", (125, 0, 150, 20)),
        WordBox("d", 1, "unrelated", (0, 0, 80, 20)),
    ])
    return con


def test_retrieval_ranks_hits_first(tmp_path):
    con = _db(tmp_path)
    ranked = keyword_rank("What is Aurora revenue?", con)
    assert ranked[0][:2] == ("d", 0) and ranked[0][2] > 0


def test_retrieval_never_abstains_zero_overlap(tmp_path):
    con = _db(tmp_path)
    # no query token appears anywhere: retrieval must still return pages
    ranked = keyword_rank("zzzqqq nonexistent", con)
    assert len(ranked) == 2
    assert all(score == 0.0 for _, _, score, _ in ranked)


def test_retrieval_doc_filter(tmp_path):
    con = _db(tmp_path)
    ranked = keyword_rank("Aurora", con, doc_id="d")
    assert {d for d, _, _, _ in ranked} == {"d"}


def test_empty_index_returns_empty(tmp_path):
    con = connect_sqlite(tmp_path / "empty.db")
    assert keyword_rank("anything", con) == []


def test_closest_passages_ranked(tmp_path):
    sents = [
        SentenceSpan("d", 0, 0, "The Aurora revenue was 42 million.", (0, 0, 10, 10)),
        SentenceSpan("d", 0, 1, "Completely unrelated note.", (0, 20, 10, 30)),
    ]
    passages = closest_passages("What was the Aurora revenue?", sents)
    assert passages and passages[0]["sent_id"] == 0
    assert passages[0]["score"] > 0 and "text" in passages[0]
