from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from lib.cache.runtime.central_cache import CentralCacheStore
from lib.cache.runtime.central_cache_models import ArtifactDescriptor, ArtifactScope, CacheRequest
from lib.cache.runtime.feature_pipeline_support import expand_param_grid, read_aligned_feature_artifact
from lib.core.enums import Ticker, TimeFrame


@pytest.fixture
def central_cache(tmp_path: Path) -> CentralCacheStore:
    CentralCacheStore.reset()
    store = CentralCacheStore(cache_dir=tmp_path)
    CentralCacheStore._instance = store  # type: ignore[attr-defined]
    yield store
    CentralCacheStore.reset()


def test_expand_param_grid_nested_signal_params_filter_gate() -> None:
    params = {
        "filter_module": "adx_filter",
        "filter_params": {"length": 20, "threshold": 20.0},
        "signal_module": "cyclical_rsi",
        "signal_params": {"short_period": [2, 3], "long_period": [80], "rsi_period": [2]},
    }
    combos = expand_param_grid(params)
    assert len(combos) == 2
    assert combos[0]["signal_params"] == {"short_period": 2, "long_period": 80, "rsi_period": 2}
    assert combos[1]["signal_params"] == {"short_period": 3, "long_period": 80, "rsi_period": 2}
    assert all(c["filter_params"] == {"length": 20, "threshold": 20.0} for c in combos)


def test_expand_param_grid_dual_signal_nested_grids() -> None:
    params = {
        "moduleA": "rsi_signal",
        "moduleB": "ewmac",
        "paramsA": {"lookback": [10, 14]},
        "paramsB": {"span_fast": [16]},
    }
    combos = expand_param_grid(params)
    assert len(combos) == 2
    assert {c["paramsA"]["lookback"] for c in combos} == {10, 14}


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
