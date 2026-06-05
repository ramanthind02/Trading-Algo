from __future__ import annotations

from pathlib import Path

import pytest

from lib.cache.runtime.central_cache import CentralCacheStore


@pytest.fixture
def isolated_central_cache(tmp_path: Path) -> None:
    CentralCacheStore.reset()
    CentralCacheStore._instance = CentralCacheStore(cache_dir=tmp_path / "central_cache")  # type: ignore[attr-defined]
    yield
    CentralCacheStore.reset()
