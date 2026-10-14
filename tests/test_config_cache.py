"""Config loads + cache hit on second call (Phase 1 acceptance)."""
from vvrag.cache import DiskCache
from vvrag.config import load_config


def test_config_loads(tmp_path):
    cfg = load_config("config.yaml")
    assert cfg.render.dpi == 150
    assert cfg.retriever.top_k == 3
    assert cfg.gate.alpha == 0.10


def test_cache_hit_on_second_call(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"v": 42}

    v1, hit1 = cache.get_or_compute({"q": "x"}, compute)
    v2, hit2 = cache.get_or_compute({"q": "x"}, compute)
    assert (v1, hit1) == ({"v": 42}, False)
    assert (v2, hit2) == ({"v": 42}, True)
    assert calls["n"] == 1
