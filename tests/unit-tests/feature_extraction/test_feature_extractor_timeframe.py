from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import feature_extraction.feature_extractor as feature_extractor
from utils.cache.runtime.central_cache import CentralCacheStore
from utils.cache.runtime.central_cache_models import ArtifactDescriptor, ArtifactScope
from utils.core.enums import Ticker, TimeFrame


def _sample_price_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=4, freq="D"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000, 1000, 1000, 1000],
        }
    )


def _sample_override_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "ticker": ["ES", "ES", "ES", "ES"],
        }
    )


def test_extract_features_single_ticker_uses_requested_timeframe_for_loading_and_streaming(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {"loaded_timeframe": None, "last_candle_tf": None, "calls": 0}

    class _DummyNode:
        def __init__(self, tf: TimeFrame) -> None:
            self.tf = tf
            self.module_name = "dummy"
            self.params = {"lookback": 2}

        def get_column_names(self) -> list[str]:
            return [f"dummy_{self.tf.name}"]

        def add_candle(self, candle: Any) -> list[float]:
            captured["last_candle_tf"] = candle.tf
            captured["calls"] += 1
            return [1.0]

    def _fake_load_data(
        ticker: Ticker,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        _ = ticker
        _ = start
        _ = end
        captured["loaded_timeframe"] = timeframe
        return _sample_price_df()

    def _fake_create_fresh_bias_node(
        module_name: str,
        ticker: Ticker,
        tf: TimeFrame,
        params: dict[str, Any],
    ) -> _DummyNode:
        _ = module_name
        _ = ticker
        _ = params
        return _DummyNode(tf)

    monkeypatch.setattr(feature_extractor.helpers, "load_data", _fake_load_data)
    monkeypatch.setattr(
        feature_extractor.helpers,
        "create_fresh_bias_node",
        _fake_create_fresh_bias_node,
    )

    features_df, targets_df = feature_extractor._extract_features_single_ticker(
        module_name="dummy",
        params={"lookback": 2},
        ticker=Ticker.ES,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 5),
        timeframes=[TimeFrame.W],
        use_cache=False,
    )

    assert captured["loaded_timeframe"] == TimeFrame.W
    assert captured["last_candle_tf"] == TimeFrame.W
    assert captured["calls"] == len(_sample_price_df())
    assert not features_df.empty
    assert not targets_df.empty


def test_extract_features_with_forward_returns_uses_timeframe_scaled_ewsd_aux_params(
    monkeypatch: Any,
) -> None:
    captured_calls: list[tuple[str, dict[str, Any]]] = []
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")

    def _fake_extract_features(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        module_name = kwargs["module_name"]
        params = kwargs["params"]
        captured_calls.append((module_name, params))

        if module_name == "ewsd":
            return (
                pd.DataFrame(
                    {
                        "ewsd_signal_W_long_run_window_520": [1.0, 1.0, 1.0, 1.0],
                        "ticker": ["ES"] * 4,
                    },
                    index=idx,
                ),
                pd.DataFrame(),
            )
        return (
            pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0], "ticker": ["ES"] * 4}, index=idx),
            pd.DataFrame(),
        )

    def _fake_compute_forward_returns(
        candles_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        _ = candles_df
        _ = features_df
        return pd.DataFrame(
            {
                "raw_return": [0.01, 0.01, 0.01, 0.01],
                "log_return": [0.01, 0.01, 0.01, 0.01],
                "log_return_ewsd": [1.0, 1.0, 1.0, 1.0],
                "ticker": ["ES"] * 4,
            },
            index=idx,
        )

    monkeypatch.setattr(feature_extractor, "extract_features", _fake_extract_features)
    monkeypatch.setattr(feature_extractor, "compute_forward_returns", _fake_compute_forward_returns)

    feature_extractor.extract_features_with_forward_returns(
        module_name="rsi",
        params={"lookback": 14},
        ticker=Ticker.ES,
        timeframes=[TimeFrame.W],
        candles_override=_sample_override_candles(),
    )

    call_map = {module_name: params for module_name, params in captured_calls}
    assert call_map["ewsd"] == {"long_run_window": 520}


def test_compute_forward_returns_accepts_non_252_ewsd_column() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    candles_df = pd.DataFrame(
        {
            "datetime": idx,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "ticker": ["ES"] * 4,
        }
    )
    features_df = pd.DataFrame(
        {
            "ewsd_signal_W_long_run_window_520": [1.0, 1.0, 1.0, 1.0],
            "ticker": ["ES"] * 4,
        },
        index=idx,
    )

    targets_df = feature_extractor.compute_forward_returns(candles_df, features_df=features_df)

    assert "log_return_ewsd" in targets_df.columns
    assert not targets_df.empty


def test_extract_features_single_ticker_populate_on_miss_writes_cache(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    CentralCacheStore.reset()
    store = CentralCacheStore(cache_dir=str(tmp_path))
    CentralCacheStore._instance = store  # type: ignore[attr-defined]

    captured: dict[str, Any] = {"calls": 0}
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")

    class _DummyNode:
        def __init__(self, tf: TimeFrame) -> None:
            self.tf = tf
            self.module_name = "dummy"
            self.params = {"lookback": 2}

        def get_column_names(self) -> list[str]:
            return [f"dummy_{self.tf.name}"]

        def add_candle(self, candle: Any) -> list[float]:
            captured["calls"] += 1
            return [1.0]

    def _fake_load_data(
        ticker: Ticker,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        _ = ticker
        _ = timeframe
        _ = start
        _ = end
        return _sample_price_df()

    def _fake_create_fresh_bias_node(
        module_name: str,
        ticker: Ticker,
        tf: TimeFrame,
        params: dict[str, Any],
    ) -> _DummyNode:
        _ = module_name
        _ = ticker
        _ = params
        return _DummyNode(tf)

    monkeypatch.setattr(feature_extractor.helpers, "load_data", _fake_load_data)
    monkeypatch.setattr(feature_extractor.helpers, "create_fresh_bias_node", _fake_create_fresh_bias_node)

    features_df, targets_df = feature_extractor._extract_features_single_ticker(
        module_name="dummy",
        params={"lookback": 2},
        ticker=Ticker.ES,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 5),
        timeframes=[TimeFrame.D],
        use_cache=True,
        populate_on_miss=True,
    )

    descriptor = ArtifactDescriptor(
        family="bias",
        module_name="dummy",
        ticker=Ticker.ES,
        timeframe=TimeFrame.D,
        params={"lookback": 2},
        scope=ArtifactScope.LIVE,
    )

    assert captured["calls"] == len(_sample_price_df())
    assert not features_df.empty
    assert not targets_df.empty
    assert store.describe_artifact(descriptor) is not None
    assert store.read_artifact(descriptor).iloc[0]["value"] == pytest.approx(1.0)
    CentralCacheStore.reset()
