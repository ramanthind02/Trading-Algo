from __future__ import annotations

import features.extraction.feature_extractor as feature_extractor
from cache.runtime.feature_pipeline_support import (
    build_bias_node_descriptor,
    preload_cross_ticker_data,
    preload_cross_ticker_override_data,
)


def test_feature_extractor_uses_centralized_cache_helpers() -> None:
    assert feature_extractor._build_bias_node_descriptor is build_bias_node_descriptor
    assert feature_extractor._preload_cross_ticker_data is preload_cross_ticker_data
    assert feature_extractor._preload_cross_ticker_override_data is preload_cross_ticker_override_data
