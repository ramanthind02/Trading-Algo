from __future__ import annotations

import feature_extraction.feature_extractor as feature_extractor
from feature_extraction.backtest import Backtest as LegacyBacktest
from utils.cache.runtime.backtest import Backtest as CentralBacktest
from utils.cache.runtime.cross_ticker_store import (
    CrossTickerDataStore as CentralCrossTickerDataStore,
)
from utils.cache.runtime.cross_ticker_store import extract_cross_ticker_names as central_extract_cross_ticker_names
from utils.cache.runtime.feature_pipeline_support import (
    build_bias_node_descriptor,
    preload_cross_ticker_data,
    preload_cross_ticker_override_data,
)
from utils.data.cross_ticker_store import (
    CrossTickerDataStore as LegacyCrossTickerDataStore,
    extract_cross_ticker_names as legacy_extract_cross_ticker_names,
)


def test_backtest_wrapper_reexports_canonical_cache_runtime() -> None:
    assert LegacyBacktest is CentralBacktest


def test_cross_ticker_wrapper_reexports_canonical_cache_store() -> None:
    assert LegacyCrossTickerDataStore is CentralCrossTickerDataStore
    assert legacy_extract_cross_ticker_names is central_extract_cross_ticker_names


def test_feature_extractor_uses_centralized_cache_helpers() -> None:
    assert feature_extractor._build_bias_node_descriptor is build_bias_node_descriptor
    assert feature_extractor._preload_cross_ticker_data is preload_cross_ticker_data
    assert feature_extractor._preload_cross_ticker_override_data is preload_cross_ticker_override_data
