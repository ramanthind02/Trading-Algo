"""
Tests for CacheManager - orchestrates cache population.

Tests cover:
- Concurrent population
- Overwrite behavior
- Error handling for missing candle files
"""

import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from utils.cache_manager import CacheManager
from utils.enums import Ticker, TimeFrame


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_candle_dir():
    """Create a temporary candles directory with sample data."""
    temp_dir = tempfile.mkdtemp()

    # Create sample candles files
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


@pytest.fixture
def cache_manager(temp_cache_dir, temp_candle_dir):
    """Create a CacheManager instance."""
    return CacheManager(
        cache_dir=temp_cache_dir,
        candle_dir=temp_candle_dir  # Fixed: use candle_dir not candles_dir
    )


class TestCacheManagerInit:
    """Tests for CacheManager initialization."""

    def test_init_creates_cache_dir(self, temp_candle_dir):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = os.path.join(tempfile.gettempdir(), 'test_cache_init')
        try:
            manager = CacheManager(
                cache_dir=cache_dir,
                candle_dir=temp_candle_dir  # Fixed
            )
            # Directory should be created during init
            assert manager.cache_dir == cache_dir
            assert os.path.exists(cache_dir)
        finally:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)


class TestPopulateSingleCache:
    """Tests for single cache population."""

    def test_populate_single_cache(self, cache_manager, temp_cache_dir):
        """Test populating a single cache."""
        result = cache_manager._populate_single_cache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31)
        )

        assert result['status'] == 'success'
        assert result['module_name'] == 'rsi'
        assert result['ticker'] == 'ES'
        assert result['row_count'] > 0

        # Verify cache file exists
        cache_path = result['cache_path']
        assert os.path.exists(cache_path)

    def test_populate_single_cache_with_complex_params(self, cache_manager):
        """Test populating cache with complex params."""
        result = cache_manager._populate_single_cache(
            module_name='ewmac',
            params={'spanFast': 16, 'spanSlow': 64},
            ticker=Ticker.NQ,
            tf=TimeFrame.D,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 6, 30)
        )

        assert result['status'] == 'success'
        assert result['row_count'] > 0


class TestPopulateCache:
    """Tests for bulk cache population."""

    def test_populate_cache_single_spec(self, cache_manager):
        """Test populating cache for single spec."""
        specs = [{
            'module_name': 'rsi',
            'params': {'lookback': 14},
            'timeframes': [TimeFrame.D]
        }]

        result = cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            show_progress=False
        )

        assert result['total'] == 1
        assert result['success'] == 1
        assert result['failed'] == 0

    def test_populate_cache_multiple_tickers(self, cache_manager):
        """Test populating cache for multiple tickers."""
        specs = [{
            'module_name': 'momentum',
            'params': {'lookback': 20},
            'timeframes': [TimeFrame.D]
        }]

        result = cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES, Ticker.NQ],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            show_progress=False
        )

        assert result['total'] == 2
        assert result['success'] == 2

    def test_populate_cache_multiple_specs(self, cache_manager):
        """Test populating cache for multiple specs."""
        specs = [
            {'module_name': 'rsi', 'params': {'lookback': 14}, 'timeframes': [TimeFrame.D]},
            {'module_name': 'rsi', 'params': {'lookback': 21}, 'timeframes': [TimeFrame.D]},
        ]

        result = cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            show_progress=False
        )

        assert result['total'] == 2
        assert result['success'] == 2

    def test_populate_cache_overwrite_existing(self, cache_manager):
        """Test overwriting existing cache."""
        specs = [{
            'module_name': 'rsi',
            'params': {'lookback': 14},
            'timeframes': [TimeFrame.D]
        }]

        # First population
        result1 = cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            overwrite_existing=True,
            show_progress=False
        )

        # Second population (overwrite)
        result2 = cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            overwrite_existing=True,
            show_progress=False
        )

        assert result1['success'] == 1
        assert result2['success'] == 1

    def test_populate_cache_skip_existing(self, cache_manager):
        """Test skipping existing cache when overwrite_existing=False."""
        specs = [{
            'module_name': 'rsi',
            'params': {'lookback': 14},
            'timeframes': [TimeFrame.D]
        }]

        # First population
        cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            overwrite_existing=True,
            show_progress=False
        )

        # Second population (should skip)
        result = cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            overwrite_existing=False,
            show_progress=False
        )

        assert result['skipped'] == 1


class TestCacheManagement:
    """Tests for cache management utilities."""

    def test_list_caches(self, cache_manager):
        """Test listing cached files."""
        specs = [{
            'module_name': 'rsi',
            'params': {'lookback': 14},
            'timeframes': [TimeFrame.D]
        }]

        cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            show_progress=False
        )

        caches = cache_manager.list_caches()
        assert len(caches) > 0
        # list_caches returns list of dicts
        assert any(c['module_name'] == 'rsi' for c in caches)

    def test_clear_caches(self, cache_manager, temp_cache_dir):
        """Test clearing all caches."""
        specs = [{
            'module_name': 'rsi',
            'params': {'lookback': 14},
            'timeframes': [TimeFrame.D]
        }]

        cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            show_progress=False
        )

        # Verify cache exists
        caches_before = cache_manager.list_caches()
        assert len(caches_before) > 0

        # Clear caches (must pass confirm=True)
        deleted = cache_manager.clear_caches(confirm=True)

        # Verify caches are cleared
        caches_after = cache_manager.list_caches()
        assert len(caches_after) == 0
        assert deleted > 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_candle_file(self, cache_manager):
        """Test handling of missing candle files."""
        # Try to populate for a ticker without candle data
        result = cache_manager._populate_single_cache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.CL,  # Not in our test candles
            tf=TimeFrame.D,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31)
        )

        assert result['status'] == 'failed'
        assert result.get('message') is not None

    def test_invalid_bias_node_spec(self, cache_manager):
        """Test handling of invalid bias node spec."""
        result = cache_manager._populate_single_cache(
            module_name='nonexistent_module',
            params={},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31)
        )

        assert result['status'] == 'failed'


class TestConcurrency:
    """Tests for concurrent cache population."""

    def test_concurrent_population(self, cache_manager):
        """Test that concurrent population works correctly."""
        specs = [
            {'module_name': 'rsi', 'params': {'lookback': 14}, 'timeframes': [TimeFrame.D]},
            {'module_name': 'momentum', 'params': {'lookback': 20}, 'timeframes': [TimeFrame.D]},
        ]

        result = cache_manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES, Ticker.NQ],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            max_workers=2,
            show_progress=False
        )

        # Should populate 2 specs x 2 tickers = 4 caches
        assert result['total'] == 4
        assert result['success'] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
