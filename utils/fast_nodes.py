"""
Fast node computation utilities with automatic Cython fallback.

This module provides optimized implementations of node computation functions.
It automatically uses Cython if available, otherwise falls back to pure Python.

Usage:
    from utils.fast_nodes import compute_atr_fast, compute_ema_fast
    
    # These will use Cython if compiled, otherwise pure Python
    atr, atr_pct, idx, n = compute_atr_fast(high, low, close, prev_close, ...)
"""

import numpy as np

# Try to import Cython optimized versions
try:
    from utils.cython_nodes import (
        compute_atr_fast as _cython_atr,
        compute_ema_fast as _cython_ema,
        compute_sma_fast as _cython_sma,
        compute_stddev_fast as _cython_stddev,
        compute_stddev_sample_fast as _cython_stddev_sample,
        compute_zscore_fast as _cython_zscore,
        compute_profit_factor_fast as _cython_pf,
        rolling_sum_update as _cython_rolling_sum,
        batch_compute_returns as _cython_batch_returns,
        compute_high_low_channel_fast as _cython_high_low_channel,
        compute_momentum_fast as _cython_momentum,
        compute_roc_fast as _cython_roc,
        compute_rsi_initial_fast as _cython_rsi_initial,
        update_rsi_fast as _cython_update_rsi,
        compute_pct_change_fast as _cython_pct_change,
        rolling_max_1d_fast as _cython_rolling_max,
        rolling_min_1d_fast as _cython_rolling_min,
        rolling_mean_1d_fast as _cython_rolling_mean,
        compute_ultimate_c_fast as _cython_ultimate_c,
        smooth_ultimate_c_fast as _cython_smooth_ultimate_c,
        normal_cdf_fast as _cython_normal_cdf,
        compute_return_fast as _cython_return,
        compute_atr_from_slice_fast as _cython_atr_from_slice,
    )
    CYTHON_NODES_AVAILABLE = True
    print("✓ Cython node optimizations loaded (5-10x speedup for ATR, EMA, etc.)")
except ImportError:
    CYTHON_NODES_AVAILABLE = False
    _cython_batch_returns = None  # type: ignore[assignment]
    _cython_high_low_channel = None  # type: ignore[assignment]
    _cython_momentum = None  # type: ignore[assignment]
    _cython_roc = None  # type: ignore[assignment]
    _cython_rsi_initial = None  # type: ignore[assignment]
    _cython_update_rsi = None  # type: ignore[assignment]
    _cython_stddev_sample = None  # type: ignore[assignment]
    _cython_pct_change = None  # type: ignore[assignment]
    _cython_rolling_max = None  # type: ignore[assignment]
    _cython_rolling_min = None  # type: ignore[assignment]
    _cython_rolling_mean = None  # type: ignore[assignment]
    _cython_ultimate_c = None  # type: ignore[assignment]
    _cython_smooth_ultimate_c = None  # type: ignore[assignment]
    _cython_normal_cdf = None  # type: ignore[assignment]
    _cython_return = None  # type: ignore[assignment]
    _cython_atr_from_slice = None  # type: ignore[assignment]
    print("⚠ Cython node optimizations not available (compile with: python utils/setup_cython.py build_ext --inplace)")


def compute_atr_fast(high, low, close, prev_close, true_ranges, buffer_idx, n_filled, period):
    """
    Fast ATR computation with automatic Cython fallback.
    
    Parameters:
    - high: Current candle high
    - low: Current candle low
    - close: Current candle close
    - prev_close: Previous candle close (or -1 if first)
    - true_ranges: Circular buffer of true ranges
    - buffer_idx: Current position in buffer
    - n_filled: Number of elements filled
    - period: ATR period
    
    Returns:
    - tuple: (atr, atr_pct, new_buffer_idx, new_n_filled)
    """
    if CYTHON_NODES_AVAILABLE:
        return _cython_atr(high, low, close, prev_close, true_ranges, buffer_idx, n_filled, period)
    else:
        # Pure Python fallback
        return _python_atr(high, low, close, prev_close, true_ranges, buffer_idx, n_filled, period)


def compute_ema_fast(value, prev_ema, alpha, is_first):
    """Fast EMA computation with automatic fallback."""
    if CYTHON_NODES_AVAILABLE:
        return _cython_ema(value, prev_ema, alpha, is_first)
    else:
        if is_first:
            return value
        else:
            return alpha * value + (1.0 - alpha) * prev_ema


def compute_sma_fast(values, n):
    """Fast SMA computation with automatic fallback."""
    if CYTHON_NODES_AVAILABLE:
        return _cython_sma(values, n)
    else:
        return np.mean(values[:n])


def compute_stddev_fast(values, n, mean_val=0.0):
    """Fast standard deviation with automatic fallback."""
    if CYTHON_NODES_AVAILABLE:
        return _cython_stddev(values, n, mean_val)
    else:
        if mean_val == 0.0:
            return np.std(values[:n])
        else:
            return np.std(values[:n])


def compute_zscore_fast(value, mean_val, stddev_val):
    """Fast z-score computation with automatic fallback."""
    if CYTHON_NODES_AVAILABLE:
        return _cython_zscore(value, mean_val, stddev_val)
    else:
        if stddev_val < 1e-10:
            return 0.0
        return (value - mean_val) / stddev_val


def compute_profit_factor_fast(returns, n):
    """Fast profit factor computation with automatic fallback."""
    if CYTHON_NODES_AVAILABLE:
        return _cython_pf(returns, n)
    else:
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = -np.sum(returns[returns < 0])
        
        if gross_loss < 1e-10:
            return 999.0 if gross_profit > 1e-10 else 1.0
        
        return gross_profit / gross_loss


def rolling_sum_update(new_value, old_value, current_sum):
    """Fast rolling sum update with automatic fallback."""
    if CYTHON_NODES_AVAILABLE:
        return _cython_rolling_sum(new_value, old_value, current_sum)
    else:
        return current_sum + new_value - old_value


def batch_compute_log_returns(close_prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns for a 1D array of close prices.

    Uses the Cython implementation from ``utils.cython_nodes`` when available,
    otherwise falls back to a vectorized NumPy implementation.

    Parameters
    ----------
    close_prices : np.ndarray
        1D array of close prices (length >= 2).

    Returns
    -------
    np.ndarray
        1D array of log returns with length ``len(close_prices) - 1``.
    """
    if close_prices.size < 2:
        return np.empty(0, dtype=np.float64)

    prices = np.asarray(close_prices, dtype=np.float64)

    if CYTHON_NODES_AVAILABLE and _cython_batch_returns is not None:
        return _cython_batch_returns(prices, prices.size)

    log_prices = np.log(prices)
    return np.diff(log_prices)


def compute_high_low_channel_fast(
    highs: np.ndarray,
    lows: np.ndarray,
    start_idx: int,
    window: int,
    n: int
) -> tuple[float, float]:
    """
    Fast computation of highest high and lowest low over a rolling window.
    
    Efficiently computes (highest_high, lowest_low) over the last `window` valid
    elements ending at `start_idx`. Designed for circular-buffer friendly usage.
    
    Parameters:
    - highs: Array of high prices (circular buffer or array)
    - lows: Array of low prices (circular buffer or array)
    - start_idx: Current index position (end of window)
    - window: Number of elements to look back
    - n: Total number of valid elements in buffer
    
    Returns:
    - tuple: (highest_high, lowest_low) over the window
    """
    if CYTHON_NODES_AVAILABLE and _cython_high_low_channel is not None:
        highs_arr = np.asarray(highs, dtype=np.float64)
        lows_arr = np.asarray(lows, dtype=np.float64)
        return _cython_high_low_channel(highs_arr, lows_arr, start_idx, window, n)
    else:
        return _python_high_low_channel(highs, lows, start_idx, window, n)


def compute_momentum_fast(curr_close: float, past_close: float) -> float:
    """
    Fast momentum computation (price difference).
    
    Computes the simple momentum as the difference between current and past close.
    
    Parameters:
    - curr_close: Current close price
    - past_close: Past close price (from lookback periods ago)
    
    Returns:
    - momentum: curr_close - past_close
    """
    if CYTHON_NODES_AVAILABLE and _cython_momentum is not None:
        return _cython_momentum(curr_close, past_close)
    else:
        return curr_close - past_close


def compute_roc_fast(curr_close: float, past_close: float) -> float:
    """
    Fast Rate of Change (ROC) computation as percentage.
    
    Computes ROC as: ((curr_close - past_close) / past_close) * 100.0
    Returns 0.0 safely when past_close <= 0 to avoid division by zero.
    
    Parameters:
    - curr_close: Current close price
    - past_close: Past close price (from lookback periods ago)
    
    Returns:
    - roc: Percentage ROC, or 0.0 if past_close <= 0
    """
    if CYTHON_NODES_AVAILABLE and _cython_roc is not None:
        return _cython_roc(curr_close, past_close)
    else:
        if past_close <= 0.0:
            return 0.0
        return ((curr_close - past_close) / past_close) * 100.0


def compute_rsi_initial_fast(
    close_prices: np.ndarray,
    lookback: int
) -> tuple[float, float]:
    """
    Fast RSI initial computation.
    
    Initializes RSI computation for the first valid period by computing average gains
    and losses over the initial lookback period. Uses 1e-60 guards to prevent division
    by zero, matching the Numba implementation exactly.
    
    Parameters:
    - close_prices: Array of close prices (must have at least lookback elements)
    - lookback: RSI period
    
    Returns:
    - tuple: (upsum, dnsum) as averages, where:
      - upsum: Average of positive price changes
      - dnsum: Average of negative price changes (as positive values)
    """
    if CYTHON_NODES_AVAILABLE and _cython_rsi_initial is not None:
        prices_arr = np.asarray(close_prices, dtype=np.float64)
        return _cython_rsi_initial(prices_arr, lookback)
    else:
        return _python_rsi_initial(close_prices, lookback)


def update_rsi_fast(
    prev_close: float,
    curr_close: float,
    upsum: float,
    dnsum: float,
    lookback: int
) -> tuple[float, float, float]:
    """
    Fast RSI update computation.
    
    Updates RSI using exponential moving average approach. Computes new average gains
    and losses, then calculates RSI value. Matches the Numba implementation exactly.
    
    Parameters:
    - prev_close: Previous close price
    - curr_close: Current close price
    - upsum: Current average gain
    - dnsum: Current average loss (as positive value)
    - lookback: RSI period
    
    Returns:
    - tuple: (new_upsum, new_dnsum, rsi_value) where:
      - new_upsum: Updated average gain
      - new_dnsum: Updated average loss (as positive value)
      - rsi_value: RSI value (0-100)
    """
    if CYTHON_NODES_AVAILABLE and _cython_update_rsi is not None:
        return _cython_update_rsi(prev_close, curr_close, upsum, dnsum, lookback)
    else:
        return _python_update_rsi(prev_close, curr_close, upsum, dnsum, lookback)


def compute_pct_change_fast(prev_close: float, curr_close: float) -> float:
    """One-period percent change. Returns 0.0 if prev_close <= 0."""
    if CYTHON_NODES_AVAILABLE and _cython_pct_change is not None:
        return _cython_pct_change(prev_close, curr_close)
    if prev_close <= 0.0:
        return 0.0
    return ((curr_close - prev_close) / prev_close) * 100.0


def rolling_max_1d_fast(values: np.ndarray, n: int, window: int) -> float:
    """Rolling max over last `window` elements of `values` (valid length `n`)."""
    if CYTHON_NODES_AVAILABLE and _cython_rolling_max is not None:
        arr = np.asarray(values, dtype=np.float64)
        return _cython_rolling_max(arr, n, window)
    if n <= 0:
        return 0.0
    start = max(0, n - window)
    return float(np.max(values[start:n]))


def rolling_min_1d_fast(values: np.ndarray, n: int, window: int) -> float:
    """Rolling min over last `window` elements of `values` (valid length `n`)."""
    if CYTHON_NODES_AVAILABLE and _cython_rolling_min is not None:
        arr = np.asarray(values, dtype=np.float64)
        return _cython_rolling_min(arr, n, window)
    if n <= 0:
        return 0.0
    start = max(0, n - window)
    return float(np.min(values[start:n]))


def rolling_mean_1d_fast(values: np.ndarray, n: int, window: int) -> float:
    """Rolling mean over last `window` elements of `values` (valid length `n`)."""
    if CYTHON_NODES_AVAILABLE and _cython_rolling_mean is not None:
        arr = np.asarray(values, dtype=np.float64)
        return _cython_rolling_mean(arr, n, window)
    if n <= 0:
        return 0.0
    start = max(0, n - window)
    return float(np.mean(values[start:n]))


def compute_ultimate_c_fast(
    roc_values: np.ndarray, lookback: int, factor: float
) -> float:
    """Ultimate C% from ROC array. Returns 50.0 if empty."""
    if CYTHON_NODES_AVAILABLE and _cython_ultimate_c is not None:
        arr = np.asarray(roc_values, dtype=np.float64)
        return _cython_ultimate_c(arr, len(arr), lookback, factor)
    return _python_compute_ultimate_c(roc_values, lookback, factor)


def smooth_ultimate_c_fast(
    ultimate_c_values: np.ndarray, smooth_lookback: int
) -> float:
    """Rolling mean of ultimate_c_values over last smooth_lookback elements."""
    n = len(ultimate_c_values)
    if CYTHON_NODES_AVAILABLE and _cython_smooth_ultimate_c is not None:
        arr = np.asarray(ultimate_c_values, dtype=np.float64)
        return _cython_smooth_ultimate_c(arr, n, smooth_lookback)
    if n <= 0:
        return 0.0
    start = max(0, n - smooth_lookback)
    return float(np.mean(ultimate_c_values[start:n]))


def compute_stddev_sample_fast(values: np.ndarray, n: int) -> float:
    """Sample standard deviation (ddof=1). Returns 0.0 if n < 2."""
    if CYTHON_NODES_AVAILABLE and _cython_stddev_sample is not None:
        arr = np.asarray(values, dtype=np.float64)
        return _cython_stddev_sample(arr, n)
    if n < 2:
        return 0.0
    return float(np.std(values[:n], ddof=1))


def normal_cdf_fast(x: float) -> float:
    """Standard normal CDF: 0.5 * (1 + erf(x/sqrt(2)))."""
    if CYTHON_NODES_AVAILABLE and _cython_normal_cdf is not None:
        return _cython_normal_cdf(x)
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_return_fast(open_px: float, close_px: float) -> tuple[float, float]:
    """Returns (pct_return, log_return). If open_px <= 0 returns (0.0, 0.0)."""
    if CYTHON_NODES_AVAILABLE and _cython_return is not None:
        return _cython_return(open_px, close_px)
    if open_px <= 0.0:
        return 0.0, 0.0
    pct = ((close_px - open_px) / open_px) * 100.0
    log_ret = np.log(close_px / open_px) if close_px > 0.0 else 0.0
    return pct, log_ret


def compute_atr_from_slice_fast(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    end_idx: int,
    period: int,
) -> float:
    """ATR over slice [end_idx - period + 1, end_idx]. Returns 0.0 if invalid."""
    if CYTHON_NODES_AVAILABLE and _cython_atr_from_slice is not None:
        h = np.asarray(highs, dtype=np.float64)
        l = np.asarray(lows, dtype=np.float64)
        c = np.asarray(closes, dtype=np.float64)
        return _cython_atr_from_slice(h, l, c, end_idx, period)
    return _python_atr_from_slice(highs, lows, closes, end_idx, period)


# ============================================================================
# PURE PYTHON FALLBACK IMPLEMENTATIONS
# ============================================================================

def _python_atr(high, low, close, prev_close, true_ranges, buffer_idx, n_filled, period):
    """Pure Python implementation of ATR computation."""
    # Calculate True Range
    if prev_close >= 0:
        hl = high - low
        hc = abs(high - prev_close)
        lc = abs(low - prev_close)
        true_range = max(hl, hc, lc)
    else:
        true_range = high - low
    
    # Store in circular buffer
    true_ranges[buffer_idx] = true_range
    new_buffer_idx = (buffer_idx + 1) % period
    new_n_filled = min(n_filled + 1, period)
    
    # Calculate ATR
    if new_n_filled > 0:
        atr = np.mean(true_ranges[:new_n_filled])
    else:
        atr = 0.0
    
    # Calculate ATR percentage
    if close > 0:
        atr_pct = (atr / close) * 100.0
    else:
        atr_pct = 0.0
    
    return atr, atr_pct, new_buffer_idx, new_n_filled


def _python_compute_ultimate_c(
    roc_values: np.ndarray, lookback: int, factor: float
) -> float:
    """Pure Python Ultimate C% (mirrors ultimate_c.compute_ultimate_c)."""
    n = len(roc_values)
    if n == 0:
        return 50.0
    current_roc = float(roc_values[-1])
    short_w = lookback
    med_w = int(lookback * factor)
    long_w = int(lookback * factor * factor)

    def roll_max(v, w):
        start = max(0, n - w)
        return float(np.max(v[start:n]))

    def roll_min(v, w):
        start = max(0, n - w)
        return float(np.min(v[start:n]))

    casey_short = 50.0
    if n >= short_w:
        hi, lo = roll_max(roc_values, short_w), roll_min(roc_values, short_w)
        if hi - lo != 0.0:
            casey_short = ((current_roc - lo) / (hi - lo)) * 100.0
    casey_med = 50.0
    if n >= med_w:
        hi, lo = roll_max(roc_values, med_w), roll_min(roc_values, med_w)
        if hi - lo != 0.0:
            casey_med = ((current_roc - lo) / (hi - lo)) * 100.0
    casey_long = 50.0
    if n >= long_w:
        hi, lo = roll_max(roc_values, long_w), roll_min(roc_values, long_w)
        if hi - lo != 0.0:
            casey_long = ((current_roc - lo) / (hi - lo)) * 100.0
    factor_sq = factor * factor
    denom = factor_sq + factor + 1.0
    return ((casey_short * factor_sq) + (casey_med * factor) + casey_long) / denom


def _python_atr_from_slice(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    end_idx: int,
    period: int,
) -> float:
    """Pure Python ATR over slice [end_idx - period + 1, end_idx]."""
    if period <= 0 or end_idx < period - 1:
        return 0.0
    start = end_idx - period + 1
    tr_sum = 0.0
    for i in range(start, end_idx + 1):
        if i == 0:
            tr = highs[0] - lows[0]
        else:
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr = max(hl, hc, lc)
        tr_sum += tr
    return tr_sum / period


def _python_high_low_channel(
    highs: np.ndarray,
    lows: np.ndarray,
    start_idx: int,
    window: int,
    n: int
) -> tuple[float, float]:
    """Pure Python implementation of high/low channel computation."""
    actual_window = min(window, n)
    
    if actual_window <= 0 or n <= 0:
        return 0.0, 0.0
    
    highest_high = -1e300
    lowest_low = 1e300
    
    # Scan backwards from start_idx, wrapping around if needed
    for i in range(actual_window):
        idx = (start_idx - i + n) % n
        
        if highs[idx] > highest_high:
            highest_high = highs[idx]
        if lows[idx] < lowest_low:
            lowest_low = lows[idx]
    
    return highest_high, lowest_low


def _python_rsi_initial(
    close_prices: np.ndarray,
    lookback: int
) -> tuple[float, float]:
    """Pure Python implementation of RSI initial computation."""
    # Import from existing RSI helpers for consistency
    try:
        from utils.rsi_helpers import compute_rsi_initial
        return compute_rsi_initial(close_prices, lookback)
    except ImportError:
        # Fallback if rsi_helpers not available
        upsum = 1e-60
        dnsum = 1e-60
        
        prices = np.asarray(close_prices, dtype=np.float64)
        for i in range(1, lookback):
            diff = prices[i] - prices[i - 1]
            if diff > 0.0:
                upsum += diff
            else:
                dnsum -= diff
        
        upsum /= (lookback - 1)
        dnsum /= (lookback - 1)
        
        return upsum, dnsum


def _python_update_rsi(
    prev_close: float,
    curr_close: float,
    upsum: float,
    dnsum: float,
    lookback: int
) -> tuple[float, float, float]:
    """Pure Python implementation of RSI update computation."""
    # Import from existing RSI helpers for consistency
    try:
        from utils.rsi_helpers import update_rsi
        return update_rsi(prev_close, curr_close, upsum, dnsum, lookback)
    except ImportError:
        # Fallback if rsi_helpers not available
        diff = curr_close - prev_close
        
        if diff > 0.0:
            new_upsum = ((lookback - 1) * upsum + diff) / lookback
            new_dnsum = dnsum * (lookback - 1.0) / lookback
        else:
            new_dnsum = ((lookback - 1) * dnsum - diff) / lookback
            new_upsum = upsum * (lookback - 1.0) / lookback
        
        rsi = 100.0 * new_upsum / (new_upsum + new_dnsum)
        
        return new_upsum, new_dnsum, rsi
