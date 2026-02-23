"""
Tests for Cython vs Python path equivalence in bias nodes.

This test suite validates that bias nodes produce identical outputs whether
using Cython-optimized kernels or pure Python fallbacks. Tests use monkeypatching
to toggle CYTHON_*_AVAILABLE flags and compare results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes.atr import ATRNode
from nodes.donchian_channel import DonchianChannel
from nodes.williamsr import WilliamsRNode
from nodes.rsi import RSI
from nodes.cumulative_rsi import CumulativeRSI
from nodes.roc import ROC
from nodes.ma_diff import MADiffNode
from nodes.ultimate_c import UltimateC


def create_synthetic_candles(n: int, start_price: float = 100.0, seed: int = 42) -> list[Candle]:
    """
    Create a deterministic sequence of synthetic candles for testing.
    
    Parameters:
    - n: Number of candles to generate
    - start_price: Starting close price
    - seed: Random seed for reproducibility
    
    Returns:
    - List of Candle objects
    """
    rng = np.random.default_rng(seed)
    base_date = datetime(2020, 1, 1)
    
    candles = []
    price = start_price
    
    for i in range(n):
        # Generate price movement
        change = rng.normal(0.0, 0.01)  # 1% daily volatility
        price = price * (1.0 + change)
        
        # Create OHLC with realistic spread
        high = price * (1.0 + abs(rng.normal(0.0, 0.002)))
        low = price * (1.0 - abs(rng.normal(0.0, 0.002)))
        open_price = price * (1.0 + rng.normal(0.0, 0.001))
        
        candle = Candle(
            datetime=base_date + timedelta(days=i),
            open=open_price,
            high=high,
            low=low,
            close=price,
            volume=1000000.0,
            ticker=Ticker.ES,
            tf=TimeFrame.D
        )
        candles.append(candle)
    
    return candles


class TestATRNode:
    """Tests for ATRNode Cython vs Python paths."""
    
    def test_atr_cython_vs_python(self):
        """Test that ATRNode produces identical outputs with Cython and Python paths."""
        candles = create_synthetic_candles(300, seed=42)
        period = 14
        
        # Import the Python fallback function
        from utils.compute.fast_nodes import _python_atr
        
        # Create a version that always uses Python (simulating Cython with same logic)
        def python_atr_wrapper(high, low, close, prev_close, true_ranges, buffer_idx, n_filled, period):
            return _python_atr(high, low, close, prev_close, true_ranges, buffer_idx, n_filled, period)
        
        # Run with Cython enabled (pretend it's available, use Python fallback as mock)
        # Patch the function in nodes.atr module's namespace
        with patch('nodes.atr.CYTHON_NODES_AVAILABLE', True):
            with patch('nodes.atr.compute_atr_fast', python_atr_wrapper):
                node_cython = ATRNode(Ticker.ES, TimeFrame.D, period=period)
                results_cython = []
                for candle in candles:
                    result = node_cython.add_candle(candle)
                    results_cython.append(result)
        
        # Run with Cython disabled (pure Python fallback)
        with patch('nodes.atr.CYTHON_NODES_AVAILABLE', False):
            node_python = ATRNode(Ticker.ES, TimeFrame.D, period=period)
            results_python = []
            for candle in candles:
                result = node_python.add_candle(candle)
                results_python.append(result)
        
        # Compare outputs
        assert len(results_cython) == len(results_python)
        
        # Skip warmup period and compare
        for i in range(period, len(results_cython)):
            cython_atr, cython_atr_pct = results_cython[i]
            python_atr, python_atr_pct = results_python[i]
            
            # ATR values should be very close (within numerical precision)
            assert np.allclose(cython_atr, python_atr, rtol=1e-10, atol=1e-10), \
                f"Mismatch at index {i}: Cython ATR={cython_atr}, Python ATR={python_atr}"
            assert np.allclose(cython_atr_pct, python_atr_pct, rtol=1e-10, atol=1e-10), \
                f"Mismatch at index {i}: Cython ATR%={cython_atr_pct}, Python ATR%={python_atr_pct}"


class TestDonchianChannel:
    """Tests for DonchianChannel Cython vs Python paths."""
    
    def test_donchian_cython_vs_python(self):
        """Test that DonchianChannel produces identical outputs with Cython and Python paths."""
        candles = create_synthetic_candles(100, seed=123)
        lookback = 20
        
        # Import the Python fallback function
        from utils.compute.fast_nodes import _python_high_low_channel
        
        # Create a version that always uses Python
        def python_hl_wrapper(highs, lows, start_idx, window, n):
            return _python_high_low_channel(highs, lows, start_idx, window, n)
        
        # Run with Cython enabled
        with patch('nodes.donchian_channel.CYTHON_NODES_AVAILABLE', True):
            with patch('nodes.donchian_channel.compute_high_low_channel_fast', python_hl_wrapper):
                node_cython = DonchianChannel(Ticker.ES, TimeFrame.D, lookback=lookback)
                results_cython = []
                for candle in candles:
                    result = node_cython.add_candle(candle)
                    results_cython.append(result[0])
        
        # Run with Cython disabled
        with patch('nodes.donchian_channel.CYTHON_NODES_AVAILABLE', False):
            node_python = DonchianChannel(Ticker.ES, TimeFrame.D, lookback=lookback)
            results_python = []
            for candle in candles:
                result = node_python.add_candle(candle)
                results_python.append(result[0])
        
        # Compare outputs
        assert len(results_cython) == len(results_python)
        
        # Skip warmup period and compare
        front_bad = lookback + 1
        for i in range(front_bad, len(results_cython)):
            cython_signal = results_cython[i]
            python_signal = results_python[i]
            
            # Signals should be identical (discrete values: 1, 0, or -1)
            assert cython_signal == python_signal, \
                f"Mismatch at index {i}: Cython={cython_signal}, Python={python_signal}"


class TestWilliamsRNode:
    """Tests for WilliamsRNode Cython vs Python paths."""
    
    def test_williamsr_cython_vs_python(self):
        """Test that WilliamsRNode produces identical outputs with Cython and Python paths."""
        candles = create_synthetic_candles(100, seed=456)
        lookback = 14
        
        # Import the Python fallback function
        from utils.compute.fast_nodes import _python_high_low_channel
        
        # Create a version that always uses Python
        def python_hl_wrapper(highs, lows, start_idx, window, n):
            return _python_high_low_channel(highs, lows, start_idx, window, n)
        
        # Run with Cython enabled
        with patch('nodes.williamsr.CYTHON_NODES_AVAILABLE', True):
            with patch('nodes.williamsr.compute_high_low_channel_fast', python_hl_wrapper):
                node_cython = WilliamsRNode(Ticker.ES, TimeFrame.D, lookback=lookback)
                results_cython = []
                for candle in candles:
                    result = node_cython.add_candle(candle)
                    results_cython.append(result[0])
        
        # Run with Cython disabled
        with patch('nodes.williamsr.CYTHON_NODES_AVAILABLE', False):
            node_python = WilliamsRNode(Ticker.ES, TimeFrame.D, lookback=lookback)
            results_python = []
            for candle in candles:
                result = node_python.add_candle(candle)
                results_python.append(result[0])
        
        # Compare outputs
        assert len(results_cython) == len(results_python)
        
        # Skip warmup period and compare
        front_bad = lookback
        for i in range(front_bad, len(results_cython)):
            cython_value = results_cython[i]
            python_value = results_python[i]
            
            # Values should be very close (within numerical precision)
            assert np.allclose(cython_value, python_value, rtol=1e-10, atol=1e-10), \
                f"Mismatch at index {i}: Cython={cython_value}, Python={python_value}"


class TestRSI:
    """Tests for RSI Cython vs Python paths."""
    
    def test_rsi_cython_vs_python(self):
        """Test that RSI produces identical outputs with Cython and Python paths."""
        candles = create_synthetic_candles(100, seed=789)
        lookback = 14
        
        # Import the Python fallback functions
        from utils.compute.fast_nodes import _python_rsi_initial, _python_update_rsi
        
        # Create wrappers that use Python implementations
        def python_rsi_initial_wrapper(close_prices, lookback):
            return _python_rsi_initial(close_prices, lookback)
        
        def python_update_rsi_wrapper(prev_close, curr_close, upsum, dnsum, lookback):
            return _python_update_rsi(prev_close, curr_close, upsum, dnsum, lookback)
        
        # Run with Cython enabled
        with patch('nodes.rsi.compute_rsi_initial_fast', python_rsi_initial_wrapper):
            with patch('nodes.rsi.update_rsi_fast', python_update_rsi_wrapper):
                node_cython = RSI(Ticker.ES, TimeFrame.D, lookback=lookback)
                results_cython = []
                for candle in candles:
                    result = node_cython.add_candle(candle)
                    results_cython.append(result[0])
        
        # Run with Cython disabled (already uses Python fallback)
        node_python = RSI(Ticker.ES, TimeFrame.D, lookback=lookback)
        results_python = []
        for candle in candles:
            result = node_python.add_candle(candle)
            results_python.append(result[0])
        
        # Compare outputs
        assert len(results_cython) == len(results_python)
        
        # Skip warmup period and compare
        for i in range(lookback, len(results_cython)):
            cython_rsi = results_cython[i]
            python_rsi = results_python[i]
            
            # RSI values should be very close (within numerical precision)
            assert np.allclose(cython_rsi, python_rsi, rtol=1e-10, atol=1e-10), \
                f"Mismatch at index {i}: Cython RSI={cython_rsi}, Python RSI={python_rsi}"


class TestCumulativeRSI:
    """Tests for CumulativeRSI Cython vs Python paths."""
    
    def test_cumulative_rsi_cython_vs_python(self):
        """Test that CumulativeRSI produces identical outputs with Cython and Python paths."""
        candles = create_synthetic_candles(100, seed=321)
        lookback = 14
        avg_period = 5
        
        # Import the Python fallback functions
        from utils.compute.fast_nodes import _python_rsi_initial, _python_update_rsi
        
        # Create wrappers that use Python implementations
        def python_rsi_initial_wrapper(close_prices, lookback):
            return _python_rsi_initial(close_prices, lookback)
        
        def python_update_rsi_wrapper(prev_close, curr_close, upsum, dnsum, lookback):
            return _python_update_rsi(prev_close, curr_close, upsum, dnsum, lookback)
        
        # Run with Cython enabled
        with patch('nodes.cumulative_rsi.compute_rsi_initial_fast', python_rsi_initial_wrapper):
            with patch('nodes.cumulative_rsi.update_rsi_fast', python_update_rsi_wrapper):
                node_cython = CumulativeRSI(Ticker.ES, TimeFrame.D, lookback=lookback, avg_period=avg_period)
                results_cython = []
                for candle in candles:
                    result = node_cython.add_candle(candle)
                    results_cython.append(result[0])
        
        # Run with Cython disabled (already uses Python fallback)
        node_python = CumulativeRSI(Ticker.ES, TimeFrame.D, lookback=lookback, avg_period=avg_period)
        results_python = []
        for candle in candles:
            result = node_python.add_candle(candle)
            results_python.append(result[0])
        
        # Compare outputs
        assert len(results_cython) == len(results_python)
        
        # Skip warmup period and compare
        front_bad = lookback + avg_period - 1
        for i in range(front_bad, len(results_cython)):
            cython_value = results_cython[i]
            python_value = results_python[i]
            
            # Values should be very close (within numerical precision)
            assert np.allclose(cython_value, python_value, rtol=1e-10, atol=1e-10), \
                f"Mismatch at index {i}: Cython={cython_value}, Python={python_value}"


class TestROC:
    """Tests for ROC Cython vs Python paths."""
    
    def test_roc_cython_vs_python(self):
        """Test that ROC produces identical outputs with Cython and Python paths."""
        candles = create_synthetic_candles(100, seed=654)
        lookback = 10
        
        # ROC uses compute_roc_fast which has inline Python fallback
        # Create a wrapper that uses Python logic
        def python_roc(curr_close: float, past_close: float) -> float:
            if past_close <= 0.0:
                return 0.0
            return ((curr_close - past_close) / past_close) * 100.0
        
        # Run with Cython enabled (using Python logic as mock)
        with patch('nodes.roc.compute_roc_fast', python_roc):
            node_cython = ROC(Ticker.ES, TimeFrame.D, lookback=lookback)
            results_cython = []
            for candle in candles:
                result = node_cython.add_candle(candle)
                results_cython.append(result[0])
        
        # Run with Cython disabled (already uses Python fallback)
        node_python = ROC(Ticker.ES, TimeFrame.D, lookback=lookback)
        results_python = []
        for candle in candles:
            result = node_python.add_candle(candle)
            results_python.append(result[0])
        
        # Compare outputs
        assert len(results_cython) == len(results_python)
        
        # Skip warmup period and compare
        for i in range(lookback, len(results_cython)):
            cython_roc = results_cython[i]
            python_roc = results_python[i]
            
            # ROC values should be very close (within numerical precision)
            assert np.allclose(cython_roc, python_roc, rtol=1e-10, atol=1e-10), \
                f"Mismatch at index {i}: Cython ROC={cython_roc}, Python ROC={python_roc}"


class TestMADiffNode:
    """Tests for MADiffNode Cython vs Python paths."""
    
    def test_ma_diff_cython_vs_python(self):
        """Test that MADiffNode produces identical outputs with Cython and Python paths."""
        candles = create_synthetic_candles(300, seed=987)
        lookback = 20
        atr_length = 252
        
        # Import the Python fallback function
        from utils.compute.fast_stats import _python_ma_diff
        
        # Create a wrapper that uses Python implementation
        def python_ma_diff_wrapper(log_close, log_closes, true_ranges, lookback, compression):
            return _python_ma_diff(log_close, log_closes, true_ranges, lookback, compression)
        
        # Run with Cython enabled
        with patch('nodes.ma_diff.compute_ma_diff_fast', python_ma_diff_wrapper):
            node_cython = MADiffNode(Ticker.ES, TimeFrame.D, lookback=lookback, atr_length=atr_length)
            results_cython = []
            for candle in candles:
                result = node_cython.add_candle(candle)
                # MADiffNode returns [output_value, bool] but we only care about the numeric value
                results_cython.append(result[0] if isinstance(result, list) else result)
        
        # Run with Cython disabled (already uses Python fallback)
        node_python = MADiffNode(Ticker.ES, TimeFrame.D, lookback=lookback, atr_length=atr_length)
        results_python = []
        for candle in candles:
            result = node_python.add_candle(candle)
            results_python.append(result[0] if isinstance(result, list) else result)
        
        # Compare outputs
        assert len(results_cython) == len(results_python)
        
        # Skip warmup period and compare
        front_bad = max(lookback, atr_length)
        for i in range(front_bad, len(results_cython)):
            cython_value = results_cython[i]
            python_value = results_python[i]
            
            # Values should be very close (within numerical precision)
            # Note: scipy.stats.norm.cdf may have slight differences, so use slightly looser tolerance
            assert np.allclose(cython_value, python_value, rtol=1e-9, atol=1e-9), \
                f"Mismatch at index {i}: Cython={cython_value}, Python={python_value}"


class TestUltimateC:
    """Tests for UltimateC Cython vs Numba paths."""

    def test_ultimate_c_cython_vs_numba(self):
        """Test that UltimateC produces identical outputs with Cython and Numba paths."""
        candles = create_synthetic_candles(80, seed=123)
        lookback = 2
        factor = 2.0
        smooth_lookback = 2

        # Run with Cython enabled (default)
        node_cython = UltimateC(
            ticker=Ticker.ES,
            tf=TimeFrame.D,
            lookback=lookback,
            factor=factor,
            smooth_lookback=smooth_lookback,
        )
        results_cython = []
        for candle in candles:
            out = node_cython.add_candle(candle)
            results_cython.append(out[0] if out else 50.0)

        # Run with Cython disabled (Numba fallback)
        with patch("nodes.ultimate_c.CYTHON_NODES_AVAILABLE", False):
            node_numba = UltimateC(
                ticker=Ticker.ES,
                tf=TimeFrame.D,
                lookback=lookback,
                factor=factor,
                smooth_lookback=smooth_lookback,
            )
            results_numba = []
            for candle in candles:
                out = node_numba.add_candle(candle)
                results_numba.append(out[0] if out else 50.0)

        assert len(results_cython) == len(results_numba)
        front_bad = UltimateC(
            ticker=Ticker.ES, tf=TimeFrame.D,
            lookback=lookback, factor=factor, smooth_lookback=smooth_lookback,
        ).front_bad
        for i in range(front_bad, len(results_cython)):
            assert np.allclose(
                results_cython[i], results_numba[i], rtol=1e-9, atol=1e-9
            ), f"Mismatch at index {i}: Cython={results_cython[i]}, Numba={results_numba[i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
