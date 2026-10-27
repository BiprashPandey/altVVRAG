"""Telemetry for multipage reader prompts (Phase 0)."""
from vvrag.reader.prompts import build_multipage_sentence_prompt
from vvrag.schemas import SentenceSpan


def _sent(page: int, sid: int, text: str) -> SentenceSpan:
    return SentenceSpan("d", page, sid, text, (0, 0, 10, 10))


def test_multipage_prompt_stats_track_all_pages():
    blocks = [
        {"label": 0, "page": 0, "doc_id": "d", "is_primary": True,
         "sentences": [_sent(0, 1, "primary evidence")]},
        {"label": 1, "page": 1, "doc_id": "d", "is_primary": False,
         "sentences": [_sent(1, 2, "secondary evidence")]},
    ]
    stats: dict = {}
    prompt = build_multipage_sentence_prompt("q?", blocks, stats_out=stats)
    assert "secondary evidence" in prompt
    assert stats["truncated"] is False
    assert [p["page"] for p in stats["pages"]] == [0, 1]
    assert all(p["included"] and not p["truncated"] for p in stats["pages"])
    assert stats["included_chars"] == stats["body_chars"]


def test_multipage_prompt_stats_mark_truncated_pages():
    blocks = [
        {"label": 0, "page": 0, "doc_id": "d", "is_primary": True,
         "sentences": [_sent(0, 1, "A" * 300)]},
        {"label": 1, "page": 1, "doc_id": "d", "is_primary": False,
         "sentences": [_sent(1, 2, "B" * 300)]},
    ]
    stats: dict = {}
    prompt = build_multipage_sentence_prompt("q?", blocks, max_chars=260,
                                              stats_out=stats)
    assert stats["truncated"] is True
    assert stats["body_chars"] > stats["included_chars"]
    assert stats["pages"][0]["included"] is True
    assert stats["pages"][0]["truncated"] is True
    assert stats["pages"][1]["included"] is False
    assert "..." in prompt
