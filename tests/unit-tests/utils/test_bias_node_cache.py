"""
Tests for BiasNodeCache - cache infrastructure for bias node outputs.

Tests cover:
- Cache path generation and params hashing
- Parquet save/load roundtrip
- Datetime range filtering
- Cache miss handling
"""

import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.cache.bias_node_cache import BiasNodeCache, CacheMissError
from utils.cache.cache_paths import default_live_artifact_cache_dir
from utils.core.enums import Ticker, TimeFrame


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_cache(temp_cache_dir):
    """Create a sample BiasNodeCache instance."""
    cache = BiasNodeCache(
        module_name='rsi',
        params={'lookback': 14},
        ticker=Ticker.ES,
        tf=TimeFrame.D,
        cache_dir=temp_cache_dir
    )
    return cache


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.random.randn(100)
    return pd.Series(values, index=dates, name='signal')


class TestBiasNodeCache:
    """Tests for BiasNodeCache class."""

    def test_init(self, sample_cache):
        """Test cache initialization."""
        assert sample_cache.module_name == 'rsi'
        assert sample_cache.params == {'lookback': 14}
        assert sample_cache.ticker == Ticker.ES
        assert sample_cache.tf == TimeFrame.D

    def test_default_cache_dir_uses_runtime_cache_tree(self) -> None:
        cache = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
        )

        assert Path(cache.cache_path).parent.parent == default_live_artifact_cache_dir()

    def test_cache_path_generation(self, sample_cache):
        """Test that cache path is generated correctly."""
        path = sample_cache.cache_path
        assert 'rsi' in path
        assert 'ES' in path
        assert 'D' in path
        assert 'lookback_14' in path
        assert path.endswith('.parquet')

    def test_params_hashing_deterministic(self, temp_cache_dir):
        """Test that params hashing is deterministic."""
        cache1 = BiasNodeCache(
            module_name='ewmac',
            params={'spanFast': 16, 'spanSlow': 64},
            ticker=Ticker.NQ,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache2 = BiasNodeCache(
            module_name='ewmac',
            params={'spanFast': 16, 'spanSlow': 64},
            ticker=Ticker.NQ,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        assert cache1.cache_path == cache2.cache_path

    def test_params_hashing_order_independent(self, temp_cache_dir):
        """Test that params order doesn't affect hash."""
        cache1 = BiasNodeCache(
            module_name='ewmac',
            params={'spanFast': 16, 'spanSlow': 64},
            ticker=Ticker.NQ,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache2 = BiasNodeCache(
            module_name='ewmac',
            params={'spanSlow': 64, 'spanFast': 16},  # Different order
            ticker=Ticker.NQ,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        assert cache1.cache_path == cache2.cache_path

    def test_save_and_load(self, sample_cache, sample_data):
        """Test save and load roundtrip."""
        # Save
        sample_cache.save(sample_data)
        assert sample_cache.exists()

        # Load returns DataFrame, get the 'value' column as series
        loaded_df = sample_cache.load()
        # The loaded data should have same values
        if isinstance(loaded_df, pd.DataFrame):
            loaded_values = loaded_df.iloc[:, 0] if loaded_df.shape[1] == 1 else loaded_df['value']
        else:
            loaded_values = loaded_df
        np.testing.assert_array_almost_equal(loaded_values.values, sample_data.values)

    def test_get_values_full_range(self, sample_cache, sample_data):
        """Test get_values returns all data when full range requested."""
        sample_cache.save(sample_data)

        start = sample_data.index[0]
        end = sample_data.index[-1]
        values = sample_cache.get_values(start, end)

        assert len(values) == len(sample_data)
        pd.testing.assert_series_equal(values, sample_data, check_names=False)

    def test_get_values_partial_range(self, sample_cache, sample_data):
        """Test get_values with partial date range."""
        sample_cache.save(sample_data)

        # Get middle 50 values
        start = sample_data.index[25]
        end = sample_data.index[74]
        values = sample_cache.get_values(start, end)

        expected = sample_data.loc[start:end]
        assert len(values) == len(expected)

    def test_get_values_cache_miss_required(self, sample_cache):
        """Test CacheMissError is raised when cache required but missing."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 4, 10)

        with pytest.raises(CacheMissError) as exc_info:
            sample_cache.get_values(start, end, require_cache=True)

        assert 'rsi' in str(exc_info.value)
        assert 'ES' in str(exc_info.value)

    def test_get_values_cache_miss_optional(self, sample_cache):
        """Test None is returned when cache optional and missing."""
        start = datetime(2020, 1, 1)
        end = datetime(2020, 4, 10)

        result = sample_cache.get_values(start, end, require_cache=False)
        assert result is None

    def test_exists_false_initially(self, sample_cache):
        """Test exists() returns False before save."""
        assert not sample_cache.exists()

    def test_exists_true_after_save(self, sample_cache, sample_data):
        """Test exists() returns True after save."""
        sample_cache.save(sample_data)
        assert sample_cache.exists()

    def test_invalidate(self, sample_cache, sample_data):
        """Test invalidate removes cache file."""
        sample_cache.save(sample_data)
        assert sample_cache.exists()

        sample_cache.invalidate()
        assert not sample_cache.exists()

    def test_save_dataframe_multi_output(self, sample_cache):
        """Test saving DataFrame for multi-output bias nodes."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'signal': np.random.randn(50),
            'signalBool': np.random.randint(0, 2, 50)
        }, index=dates)

        sample_cache.save(df)
        assert sample_cache.exists()

        # Load as DataFrame
        loaded = sample_cache.get_dataframe(
            start=dates[0],
            end=dates[-1]
        )
        assert isinstance(loaded, pd.DataFrame)
        assert 'signal' in loaded.columns
        assert 'signalBool' in loaded.columns


class TestCacheMissError:
    """Tests for CacheMissError exception."""

    def test_error_message_contains_context(self):
        """Test error message contains module, ticker, timeframe info."""
        error = CacheMissError(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            date_range=(datetime(2020, 1, 1), datetime(2020, 12, 31))
        )
        msg = str(error)
        assert 'rsi' in msg
        assert 'ES' in msg
        assert 'D' in msg
        assert '2020' in msg

    def test_error_attributes(self):
        """Test CacheMissError attributes are accessible."""
        error = CacheMissError(
            module_name='ewmac',
            params={'spanFast': 16, 'spanSlow': 64},
            ticker=Ticker.NQ,
            tf=TimeFrame.W,
            date_range=(datetime(2019, 1, 1), datetime(2023, 12, 31))
        )
        assert error.module_name == 'ewmac'
        assert error.params == {'spanFast': 16, 'spanSlow': 64}
        assert error.ticker == Ticker.NQ
        assert error.tf == TimeFrame.W


class TestCachePathStructure:
    """Tests for cache directory structure."""

    def test_creates_module_directory(self, temp_cache_dir, sample_data):
        """Test that module directory is created."""
        cache = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache.save(sample_data)

        module_dir = os.path.join(temp_cache_dir, 'rsi')
        assert os.path.isdir(module_dir)

    def test_different_params_different_files(self, temp_cache_dir, sample_data):
        """Test that different params create different cache files."""
        cache1 = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache2 = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 21},  # Different lookback
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )

        cache1.save(sample_data)
        cache2.save(sample_data * 2)

        assert cache1.cache_path != cache2.cache_path
        assert cache1.exists()
        assert cache2.exists()

        # Verify data is different
        data1 = cache1.load()
        data2 = cache2.load()
        assert not np.allclose(data1.values, data2.values)

    def test_complex_params_no_collision(self, temp_cache_dir, sample_data):
        """Test that complex params (lists/dicts) use hash to avoid filename collision.

        This tests the fix for the bug where params with only complex values
        (lists, dicts) would result in empty suffix and filename collision.
        """
        # Both params have only complex values (lists)
        cache1 = BiasNodeCache(
            module_name='custom_node',
            params={'thresholds': [0.1, 0.2, 0.3]},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache2 = BiasNodeCache(
            module_name='custom_node',
            params={'thresholds': [0.5, 0.6, 0.7]},  # Different list values
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )

        # Cache paths should be different (both should use hash)
        assert cache1.cache_path != cache2.cache_path

        # Verify both paths contain a hash (not empty suffix)
        # Path.stem gives filename without extension
        from pathlib import Path
        assert len(Path(cache1.cache_path).stem) > len('ES_D')
        assert len(Path(cache2.cache_path).stem) > len('ES_D')

        # Save different data and verify no collision
        cache1.save(sample_data)
        cache2.save(sample_data * 2)

        assert cache1.exists()
        assert cache2.exists()

        data1 = cache1.load()
        data2 = cache2.load()
        assert not np.allclose(data1.values, data2.values)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
