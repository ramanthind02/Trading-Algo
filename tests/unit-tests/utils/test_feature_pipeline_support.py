from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from utils.cache.central_cache import CentralCacheStore
from utils.cache.central_cache_models import ArtifactDescriptor, ArtifactScope, CacheRequest
from utils.cache.feature_pipeline_support import read_aligned_feature_artifact
from utils.core.enums import Ticker, TimeFrame


@pytest.fixture
def central_cache(tmp_path: Path) -> CentralCacheStore:
    CentralCacheStore.reset()
    store = CentralCacheStore(cache_dir=tmp_path)
    CentralCacheStore._instance = store  # type: ignore[attr-defined]
    yield store
    CentralCacheStore.reset()


def test_read_aligned_feature_artifact_clamps_calendar_boundaries_to_coverage(
    central_cache: CentralCacheStore,
) -> None:
    descriptor = ArtifactDescriptor(
        family="bias",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        module_name="rsi",
        params={"lookback": 5},
        scope=ArtifactScope.LIVE,
    )
    artifact_index = pd.DatetimeIndex(
        [
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
            datetime(2024, 1, 5),
        ]
    )
    central_cache.write_artifact(
        descriptor,
        pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=artifact_index),
        depends_on=[(Ticker.ES, TimeFrame.D)],
    )

    aligned = read_aligned_feature_artifact(
        central_cache,
        descriptor,
        CacheRequest(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        ),
        price_index=artifact_index,
    )

    assert list(aligned.index) == list(artifact_index)
    assert aligned["value"].tolist() == [1.0, 2.0, 3.0]
