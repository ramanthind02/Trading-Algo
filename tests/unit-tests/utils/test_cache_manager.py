"""Tests for CacheManager initialization and default cache/candle directories.

The legacy ``populate_cache`` / ``_populate_single_cache`` surface (and its CLI)
was retired in favour of the central-cache path
(``bootstrap_source_candles`` + ``ensure_bias_cache_coverage`` /
``ensure_vault_cache_coverage``); those paths are covered by the integration
suites ``tests/integration/test_portfolio_cache_cutover.py`` and
``tests/integration/test_live_cache_refresh.py``.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cache.runtime.cache_manager import CacheManager
from cache.runtime.cache_paths import default_source_candle_dir
from lib.core.enums import Ticker


@pytest.fixture
def temp_candle_dir():
    """Create a temporary candles directory with sample data."""
    temp_dir = tempfile.mkdtemp()

    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    for ticker in [Ticker.ES, Ticker.NQ]:
        ticker_dir = os.path.join(temp_dir, ticker.name)
        os.makedirs(ticker_dir, exist_ok=True)

        candles = pd.DataFrame({
            'datetime': dates,
            'open': 3000 + np.random.randn(500).cumsum(),
            'high': 3005 + np.random.randn(500).cumsum(),
            'low': 2995 + np.random.randn(500).cumsum(),
            'close': 3000 + np.random.randn(500).cumsum(),
            'volume': np.random.randint(1000, 10000, 500),
            'ticker': ticker.name,
            'timeframe': 'D'
        })

        candles.to_parquet(os.path.join(ticker_dir, 'D.parquet'))

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCacheManagerInit:
    """Tests for CacheManager initialization."""

    def test_init_creates_cache_dir(self, temp_candle_dir):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = os.path.join(tempfile.gettempdir(), 'test_cache_init')
        try:
            manager = CacheManager(
                cache_dir=cache_dir,
                candle_dir=temp_candle_dir,
            )
            # Directory should be created during init
            assert manager.cache_dir == cache_dir
            assert os.path.exists(cache_dir)
        finally:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

    def test_init_uses_runtime_cache_and_read_only_source_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        expected_cache_dir = tmp_path / ".cache" / "trading_algo" / "central_cache" / "artifacts" / "live"
        expected_candle_dir = tmp_path / "data" / "ohlc_data"
        monkeypatch.setattr(
            "cache.runtime.cache_manager.default_live_artifact_cache_dir",
            lambda: expected_cache_dir,
        )
        monkeypatch.setattr(
            "cache.runtime.cache_manager.default_source_candle_dir",
            lambda: expected_candle_dir,
        )

        manager = CacheManager()

        assert Path(manager.cache_dir) == expected_cache_dir
        assert Path(manager.candle_dir) == expected_candle_dir
        assert Path(manager.cache_dir) != default_source_candle_dir()
        assert Path(manager.cache_dir).is_relative_to(tmp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
