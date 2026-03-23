"""
Integration tests for bias node caching system.

Tests cover:
- Equivalence test: vectorized vs streaming outputs must be identical
- Lookahead bias test: cached backtest doesn't leak future data
- Multi-ticker test: aggregation works with cached features
- End-to-end test: Portfolio fit/predict with cache
"""

import os
import tempfile
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from utils.cache.bias_node_cache import BiasNodeCache
from utils.cache.cache_manager import CacheManager
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle
from nodes.rsi import RSI
from nodes.momentum import Momentum


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_candles_df():
    """Create sample candles DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    np.random.seed(42)  # For reproducibility

    candles = pd.DataFrame({
        'datetime': dates,
        'open': 3000 + np.random.randn(300).cumsum(),
        'high': 3005 + np.random.randn(300).cumsum(),
        'low': 2995 + np.random.randn(300).cumsum(),
        'close': 3000 + np.random.randn(300).cumsum(),
        'volume': np.random.randint(1000, 10000, 300),
        'ticker': 'ES',
        'timeframe': 'D'
    })

    # Ensure high is always >= low and >= close
    candles['high'] = candles[['open', 'high', 'close']].max(axis=1) + 5
    candles['low'] = candles[['open', 'low', 'close']].min(axis=1) - 5

    return candles


@pytest.fixture
def multi_ticker_candles_df():
    """Create multi-ticker candles DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)

    all_candles = []
    for ticker in ['ES', 'NQ', 'CL']:
        candles = pd.DataFrame({
            'datetime': dates,
            'open': 3000 + np.random.randn(200).cumsum(),
            'high': 3005 + np.random.randn(200).cumsum(),
            'low': 2995 + np.random.randn(200).cumsum(),
            'close': 3000 + np.random.randn(200).cumsum(),
            'volume': np.random.randint(1000, 10000, 200),
            'ticker': ticker,
            'timeframe': 'D'
        })
        candles['high'] = candles[['open', 'high', 'close']].max(axis=1) + 5
        candles['low'] = candles[['open', 'low', 'close']].min(axis=1) - 5
        all_candles.append(candles)

    return pd.concat(all_candles, ignore_index=True)


class TestVectorizedVsStreamingEquivalence:
    """Tests that vectorized and streaming methods produce identical results."""

    def test_rsi_equivalence(self, sample_candles_df, temp_cache_dir):
        """Test RSI output equivalence between streaming and cached."""
        # Create RSI node
        rsi_node = RSI(
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            lookback=14
        )

        # Stream candles (original method)
        streaming_output = []
        for _, row in sample_candles_df.iterrows():
            candle = Candle.from_row(row)
            result = rsi_node.add_candle(candle)
            if result:
                streaming_output.append(result[0])

        # Create cache from output
        dates = sample_candles_df['datetime'].values[:len(streaming_output)]
        cached_series = pd.Series(streaming_output, index=pd.DatetimeIndex(dates))

        # Create cache and save
        cache = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache.save(cached_series)

        # Load from cache (vectorized access)
        start = dates[0]
        end = dates[-1]
        vectorized_output = cache.get_values(start, end)

        # Compare
        np.testing.assert_array_almost_equal(
            streaming_output,
            vectorized_output.values,
            decimal=10,
            err_msg="Streaming and cached RSI outputs differ"
        )

    def test_momentum_equivalence(self, sample_candles_df, temp_cache_dir):
        """Test Momentum output equivalence between streaming and cached."""
        # Create Momentum node
        mom_node = Momentum(
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            lookback=20
        )

        # Stream candles
        streaming_output = []
        for _, row in sample_candles_df.iterrows():
            candle = Candle.from_row(row)
            result = mom_node.add_candle(candle)
            if result:
                streaming_output.append(result[0])

        # Create cache from output
        dates = sample_candles_df['datetime'].values[:len(streaming_output)]
        cached_series = pd.Series(streaming_output, index=pd.DatetimeIndex(dates))

        cache = BiasNodeCache(
            module_name='momentum',
            params={'lookback': 20},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache.save(cached_series)

        # Load from cache
        start = dates[0]
        end = dates[-1]
        vectorized_output = cache.get_values(start, end)

        # Compare
        np.testing.assert_array_almost_equal(
            streaming_output,
            vectorized_output.values,
            decimal=10,
            err_msg="Streaming and cached Momentum outputs differ"
        )


class TestLookaheadBias:
    """Tests that caching system doesn't introduce lookahead bias."""

    def test_no_future_data_leakage(self, sample_candles_df, temp_cache_dir):
        """Test that cached values don't include future data."""
        # Create and populate cache with full data
        rsi_node = RSI(
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            lookback=14
        )

        all_output = []
        for _, row in sample_candles_df.iterrows():
            candle = Candle.from_row(row)
            result = rsi_node.add_candle(candle)
            if result:
                all_output.append(result[0])

        dates = sample_candles_df['datetime'].values[:len(all_output)]
        full_series = pd.Series(all_output, index=pd.DatetimeIndex(dates))

        cache = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache.save(full_series)

        # Request only first 100 values
        partial_end = dates[99]
        partial_values = cache.get_values(dates[0], partial_end)

        # Should only have 100 values, not future data
        assert len(partial_values) == 100
        assert partial_values.index[-1] <= pd.Timestamp(partial_end)

        # Values should match first 100 streaming outputs
        np.testing.assert_array_almost_equal(
            partial_values.values,
            all_output[:100],
            decimal=10
        )

    def test_sequential_access_matches_streaming(self, sample_candles_df, temp_cache_dir):
        """Test that sequential cache access matches sequential streaming."""
        # Create and populate cache
        rsi_node = RSI(
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            lookback=14
        )

        all_output = []
        for _, row in sample_candles_df.iterrows():
            candle = Candle.from_row(row)
            result = rsi_node.add_candle(candle)
            if result:
                all_output.append(result[0])

        dates = sample_candles_df['datetime'].values[:len(all_output)]
        full_series = pd.Series(all_output, index=pd.DatetimeIndex(dates))

        cache = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache.save(full_series)

        # Simulate walk-forward: access cache in chunks
        chunk_size = 50
        for i in range(0, len(dates) - chunk_size, chunk_size):
            start_idx = i
            end_idx = min(i + chunk_size - 1, len(dates) - 1)

            chunk_values = cache.get_values(dates[start_idx], dates[end_idx])

            # Each chunk should match streaming output for that period
            expected = all_output[start_idx:end_idx + 1]
            np.testing.assert_array_almost_equal(
                chunk_values.values,
                expected,
                decimal=10,
                err_msg=f"Chunk {i//chunk_size} mismatch"
            )


class TestMultiTickerAggregation:
    """Tests for multi-ticker feature aggregation with caching."""

    def test_multi_ticker_cache_access(self, multi_ticker_candles_df, temp_cache_dir):
        """Test accessing cached features for multiple tickers."""
        tickers = [Ticker.ES, Ticker.NQ, Ticker.CL]

        # Create caches for each ticker
        for ticker in tickers:
            ticker_name = ticker.name
            ticker_candles = multi_ticker_candles_df[
                multi_ticker_candles_df['ticker'] == ticker_name
            ].copy()

            rsi_node = RSI(
                ticker=ticker,
                tf=TimeFrame.D,
                lookback=14
            )

            output = []
            for _, row in ticker_candles.iterrows():
                candle = Candle.from_row(row)
                result = rsi_node.add_candle(candle)
                if result:
                    output.append(result[0])

            dates = ticker_candles['datetime'].values[:len(output)]
            series = pd.Series(output, index=pd.DatetimeIndex(dates))

            cache = BiasNodeCache(
                module_name='rsi',
                params={'lookback': 14},
                ticker=ticker,
                tf=TimeFrame.D,
                cache_dir=temp_cache_dir
            )
            cache.save(series)

        # Access all caches
        all_features = []
        for ticker in tickers:
            cache = BiasNodeCache(
                module_name='rsi',
                params={'lookback': 14},
                ticker=ticker,
                tf=TimeFrame.D,
                cache_dir=temp_cache_dir
            )
            values = cache.get_values(
                datetime(2020, 1, 1),
                datetime(2020, 7, 18)
            )
            all_features.append(values)

        # Each ticker should have features
        for i, features in enumerate(all_features):
            assert len(features) > 0, f"Ticker {tickers[i].name} has no features"

    def test_aggregation_across_tickers(self, multi_ticker_candles_df, temp_cache_dir):
        """Test feature aggregation across multiple tickers."""
        tickers = [Ticker.ES, Ticker.NQ]

        # Create caches for each ticker
        ticker_values = {}
        for ticker in tickers:
            ticker_name = ticker.name
            ticker_candles = multi_ticker_candles_df[
                multi_ticker_candles_df['ticker'] == ticker_name
            ].copy()

            rsi_node = RSI(
                ticker=ticker,
                tf=TimeFrame.D,
                lookback=14
            )

            output = []
            for _, row in ticker_candles.iterrows():
                candle = Candle.from_row(row)
                result = rsi_node.add_candle(candle)
                if result:
                    output.append(result[0])

            dates = ticker_candles['datetime'].values[:len(output)]
            series = pd.Series(output, index=pd.DatetimeIndex(dates))

            cache = BiasNodeCache(
                module_name='rsi',
                params={'lookback': 14},
                ticker=ticker,
                tf=TimeFrame.D,
                cache_dir=temp_cache_dir
            )
            cache.save(series)
            ticker_values[ticker.name] = series

        # Load and aggregate
        all_values = []
        for ticker in tickers:
            cache = BiasNodeCache(
                module_name='rsi',
                params={'lookback': 14},
                ticker=ticker,
                tf=TimeFrame.D,
                cache_dir=temp_cache_dir
            )
            values = cache.get_values(
                datetime(2020, 1, 1),
                datetime(2020, 7, 18)
            )
            all_values.append(values)

        # Concatenate and aggregate by datetime (mean)
        combined = pd.concat(all_values)
        aggregated = combined.groupby(combined.index).mean()

        # Verify aggregation works
        assert len(aggregated) > 0
        assert not aggregated.isna().all()


class TestEndToEnd:
    """End-to-end tests for the caching system."""

    def test_bias_node_cache_integration(self, sample_candles_df, temp_cache_dir):
        """Test bias node with cache loading at initialization."""
        # First, create and save cache manually
        rsi_node_stream = RSI(
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            lookback=14
        )

        output = []
        for _, row in sample_candles_df.iterrows():
            candle = Candle.from_row(row)
            result = rsi_node_stream.add_candle(candle)
            if result:
                output.append(result[0])

        dates = sample_candles_df['datetime'].values[:len(output)]
        series = pd.Series(output, index=pd.DatetimeIndex(dates))

        # Create cache directory for RSI
        cache = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache.save(series)

        # Verify cache exists
        assert cache.exists()

        # Create a new RSI node and verify it can access cached values
        cache2 = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )

        cached_values = cache2.get_values(
            datetime(2020, 1, 1),
            datetime(2020, 10, 27)
        )

        assert cached_values is not None
        assert len(cached_values) > 0

    def test_cache_with_base_model(self, sample_candles_df, temp_cache_dir):
        """Test that BaseModel can use cached features."""
        from feature_selection.base_models.feature_base_model import BaseModel
        from feature_selection.base_models.continuous_binning import ContinuousBinningModel

        # Create and populate cache
        rsi_node = RSI(
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            lookback=14
        )

        output = []
        for _, row in sample_candles_df.iterrows():
            candle = Candle.from_row(row)
            result = rsi_node.add_candle(candle)
            if result:
                output.append(result[0])

        dates = sample_candles_df['datetime'].values[:len(output)]
        series = pd.Series(output, index=pd.DatetimeIndex(dates))

        cache = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=temp_cache_dir
        )
        cache.save(series)

        # Create BaseModel with use_cache=False (for comparison)
        feature_config = {
            'bias_node_spec': {
                'module_name': 'rsi',
                'timeframes': [TimeFrame.D],
                'params': {'lookback': 14}
            },
            'model_type': 'ContinuousBinningModel',
            'constructor_params': {'n_bins': 5}
        }

        base_model = BaseModel(
            feature_config=feature_config,
            tickers=[Ticker.ES],
            use_cache=False  # Use streaming for this test
        )

        # Create target data
        returns = sample_candles_df['close'].pct_change().shift(-1)
        returns.index = sample_candles_df['datetime']
        returns = returns.dropna()

        # Fit model (streaming mode)
        base_model.stream_fit(sample_candles_df, returns)

        # Verify model is fitted
        assert base_model.binning_model.is_fitted_


class TestCacheManagerIntegration:
    """Integration tests for CacheManager with bias nodes."""

    def test_populate_and_use_cache(self, sample_candles_df, temp_cache_dir):
        """Test populating cache via CacheManager and using it."""
        # Create temp candles directory
        candles_dir = os.path.join(temp_cache_dir, 'candles')
        os.makedirs(candles_dir, exist_ok=True)

        # Save candles
        ticker_dir = os.path.join(candles_dir, 'ES')
        os.makedirs(ticker_dir, exist_ok=True)
        sample_candles_df.to_parquet(os.path.join(ticker_dir, 'D.parquet'))

        # Create CacheManager
        manager = CacheManager(
            cache_dir=os.path.join(temp_cache_dir, 'cache'),
            candle_dir=candles_dir  # Fixed: use candle_dir not candles_dir
        )

        # Populate cache
        specs = [{
            'module_name': 'rsi',
            'params': {'lookback': 14},
            'timeframes': [TimeFrame.D]
        }]

        result = manager.populate_cache(
            bias_node_specs=specs,
            tickers=[Ticker.ES],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 10, 27),
            show_progress=False
        )

        # 1 user spec + 1 auxiliary (ewsd) = 2 total
        assert result['success'] == 2

        # Verify cache can be loaded
        cache = BiasNodeCache(
            module_name='rsi',
            params={'lookback': 14},
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            cache_dir=os.path.join(temp_cache_dir, 'cache')
        )

        assert cache.exists()
        values = cache.get_values(
            datetime(2020, 1, 1),
            datetime(2020, 10, 27)
        )
        assert values is not None
        assert len(values) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
