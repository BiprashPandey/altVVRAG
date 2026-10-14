"""Disk cache keyed by hash of request (Section 13: resumable, no repeated API calls)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


def hash_request(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class DiskCache:
    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def contains(self, key_obj: Any) -> bool:
        return self._path(hash_request(key_obj)).exists()

    def get(self, key_obj: Any) -> Any | None:
        p = self._path(hash_request(key_obj))
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def set(self, key_obj: Any, value: Any) -> str:
        key = hash_request(key_obj)
        p = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=str(self.cache_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return key

    def get_or_compute(self, key_obj: Any, compute: Callable[[], Any]) -> tuple[Any, bool]:
        """Return (value, hit). Compute + store on miss."""
        p = self._path(hash_request(key_obj))
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), True
        value = compute()
        self.set(key_obj, value)
        return value, False
