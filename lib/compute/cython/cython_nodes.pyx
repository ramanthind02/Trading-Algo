# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized node computation functions.

This module provides ultra-fast implementations of node computation logic
that are called tens of thousands of times during backtesting.

Expected speedups:
- ATR computation: 5-10x faster
- Moving average computations: 3-5x faster
- General array operations: 2-5x faster
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, fmax, log, erf
cimport cython

# Initialize numpy C API
cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double fast_max3(double a, double b, double c) nogil:
    """Fast maximum of 3 values without Python overhead."""
    return fmax(fmax(a, b), c)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double fast_mean(double[::1] arr, Py_ssize_t n) nogil:
    """
    Fast mean calculation without numpy overhead.
    
    Parameters:
    - arr: Array to compute mean of
    - n: Number of elements to use
    
    Returns:
    - mean: Mean value
    """
    cdef double sum_val = 0.0
    cdef Py_ssize_t i
    
    for i in range(n):
        sum_val += arr[i]
    
    return sum_val / <double>n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_atr_fast(
    double high,
    double low,
    double close,
    double prev_close,
    double[::1] true_ranges,
    Py_ssize_t buffer_idx,
    Py_ssize_t n_filled,
    Py_ssize_t period
):
    """
    Fast ATR computation using Cython.
    
    This is 5-10x faster than the pure Python version because:
    - No numpy overhead for mean calculation
    - No deque overhead
    - Direct C-level computation
    - No Python object creation
    
    Parameters:
    - high: Current candle high
    - low: Current candle low
    - close: Current candle close
    - prev_close: Previous candle close (or -1 if first candle)
    - true_ranges: Circular buffer of true ranges
    - buffer_idx: Current position in buffer
    - n_filled: Number of elements filled in buffer
    - period: ATR period
    
    Returns:
    - tuple: (atr, atr_pct, new_buffer_idx, new_n_filled)
    """
    cdef double true_range
    cdef double hl, hc, lc
    cdef double atr, atr_pct
    cdef Py_ssize_t new_buffer_idx, new_n_filled
    
    # Calculate True Range
    if prev_close >= 0:
        hl = high - low
        hc = fabs(high - prev_close)
        lc = fabs(low - prev_close)
        true_range = fast_max3(hl, hc, lc)
    else:
        true_range = high - low
    
    # Store in circular buffer
    true_ranges[buffer_idx] = true_range
    new_buffer_idx = (buffer_idx + 1) % period
    new_n_filled = min(n_filled + 1, period)
    
    # Calculate ATR as mean of true ranges
    if new_n_filled > 0:
        atr = fast_mean(true_ranges, new_n_filled)
    else:
        atr = 0.0
    
    # Calculate ATR as percentage of close
    if close > 0:
        atr_pct = (atr / close) * 100.0
    else:
        atr_pct = 0.0
    
    return atr, atr_pct, new_buffer_idx, new_n_filled


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_ema_fast(
    double value,
    double prev_ema,
    double alpha,
    bint is_first
):
    """
    Fast EMA computation.
    
    Parameters:
    - value: Current value
    - prev_ema: Previous EMA value
    - alpha: Smoothing factor (2 / (period + 1))
    - is_first: Whether this is the first value
    
    Returns:
    - ema: New EMA value
    """
    if is_first:
        return value
    else:
        return alpha * value + (1.0 - alpha) * prev_ema


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_sma_fast(
    double[::1] values,
    Py_ssize_t n
):
    """
    Fast SMA computation.
    
    Parameters:
    - values: Array of values
    - n: Number of values to use
    
    Returns:
    - sma: Simple moving average
    """
    return fast_mean(values, n)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_stddev_fast(
    double[::1] values,
    Py_ssize_t n,
    double mean_val
):
    """
    Fast standard deviation computation.
    
    Parameters:
    - values: Array of values
    - n: Number of values
    - mean_val: Pre-computed mean (pass 0 to compute internally)
    
    Returns:
    - stddev: Standard deviation
    """
    cdef double sum_sq = 0.0
    cdef double diff
    cdef Py_ssize_t i
    cdef double computed_mean
    
    # Compute mean if not provided
    if mean_val == 0.0:
        computed_mean = fast_mean(values, n)
    else:
        computed_mean = mean_val
    
    # Compute sum of squared differences
    for i in range(n):
        diff = values[i] - computed_mean
        sum_sq += diff * diff
    
    return sqrt(sum_sq / <double>n)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_percentile_fast(
    double[::1] values,
    Py_ssize_t n,
    double percentile
):
    """
    Fast percentile computation using linear interpolation.
    
    Parameters:
    - values: Sorted array of values
    - n: Number of values
    - percentile: Percentile to compute (0-100)
    
    Returns:
    - value: Percentile value
    """
    cdef double idx_float
    cdef Py_ssize_t idx_low, idx_high
    cdef double frac
    
    if n == 0:
        return 0.0
    
    if n == 1:
        return values[0]
    
    # Compute index
    idx_float = (percentile / 100.0) * <double>(n - 1)
    idx_low = <Py_ssize_t>idx_float
    idx_high = idx_low + 1
    
    if idx_high >= n:
        return values[n - 1]
    
    # Linear interpolation
    frac = idx_float - <double>idx_low
    return values[idx_low] * (1.0 - frac) + values[idx_high] * frac


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_zscore_fast(
    double value,
    double mean_val,
    double stddev_val
):
    """
    Fast z-score computation.
    
    Parameters:
    - value: Value to compute z-score for
    - mean_val: Mean of distribution
    - stddev_val: Standard deviation of distribution
    
    Returns:
    - zscore: Z-score value
    """
    if stddev_val < 1e-10:
        return 0.0
    
    return (value - mean_val) / stddev_val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def batch_compute_returns(
    double[::1] close_prices,
    Py_ssize_t n
):
    """
    Fast batch computation of log returns.
    
    Parameters:
    - close_prices: Array of close prices
    - n: Number of prices
    
    Returns:
    - returns: Array of log returns (length n-1)
    """
    cdef cnp.ndarray[double, ndim=1] returns = np.empty(n - 1, dtype=np.float64)
    cdef Py_ssize_t i
    
    for i in range(n - 1):
        if close_prices[i] > 0:
            returns[i] = np.log(close_prices[i + 1] / close_prices[i])
        else:
            returns[i] = 0.0
    
    return returns


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rolling_sum_update(
    double new_value,
    double old_value,
    double current_sum
):
    """
    Fast rolling sum update (O(1) instead of O(n)).
    
    Parameters:
    - new_value: New value entering the window
    - old_value: Old value leaving the window
    - current_sum: Current sum
    
    Returns:
    - new_sum: Updated sum
    """
    return current_sum + new_value - old_value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_profit_factor_fast(
    double[::1] returns,
    Py_ssize_t n
):
    """
    Fast profit factor computation.
    
    Parameters:
    - returns: Array of returns
    - n: Number of returns
    
    Returns:
    - profit_factor: Gross profit / gross loss
    """
    cdef double gross_profit = 0.0
    cdef double gross_loss = 0.0
    cdef Py_ssize_t i
    cdef double ret
    
    for i in range(n):
        ret = returns[i]
        if ret > 0:
            gross_profit += ret
        else:
            gross_loss -= ret  # Make positive
    
    if gross_loss < 1e-10:
        return 999.0 if gross_profit > 1e-10 else 1.0
    
    return gross_profit / gross_loss


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_ma_from_deque(
    object deque_obj,
    Py_ssize_t required_length
):
    """
    Fast moving average computation from a deque object.
    
    This is optimized for computing moving averages from collections.deque
    which are commonly used in streaming/online computations.
    
    Parameters:
    - deque_obj: A deque object containing numeric values
    - required_length: Minimum required length to compute MA
    
    Returns:
    - ma: Moving average value (0.0 if insufficient data)
    """
    cdef Py_ssize_t n = len(deque_obj)
    
    if n < required_length:
        return 0.0
    
    # Convert deque to numpy array for fast computation
    cdef cnp.ndarray[double, ndim=1] arr = np.array(deque_obj, dtype=np.float64)
    cdef double sum_val = 0.0
    cdef Py_ssize_t i
    
    for i in range(n):
        sum_val += arr[i]
    
    return sum_val / <double>n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_high_low_channel_fast(
    double[::1] highs,
    double[::1] lows,
    Py_ssize_t start_idx,
    Py_ssize_t window,
    Py_ssize_t n
):
    """
    Fast computation of highest high and lowest low over a rolling window.
    
    Efficiently computes (highest_high, lowest_low) over the last `window` valid
    elements ending at `start_idx`. Designed for circular-buffer friendly usage
    where `start_idx` points to the current position and we look back `window`
    elements (wrapping around if needed).
    
    This function is optimized for Donchian Channel and Williams %R calculations
    where we need to find the highest high and lowest low over a lookback period.
    
    Parameters:
    - highs: Memoryview of high prices (circular buffer or array)
    - lows: Memoryview of low prices (circular buffer or array)
    - start_idx: Current index position (end of window, 0-based)
    - window: Number of elements to look back
    - n: Total number of valid elements in buffer (may be less than buffer size)
    
    Returns:
    - tuple: (highest_high, lowest_low) over the window
    
    Usage example:
        # For a circular buffer with period=20, current position at idx=15:
        # Look back 20 elements: indices wrapping around from 15 backwards
        highest, lowest = compute_high_low_channel_fast(highs, lows, start_idx=15, window=20, n=20)
    """
    cdef double highest_high = -1e300
    cdef double lowest_low = 1e300
    cdef Py_ssize_t i, idx
    cdef Py_ssize_t actual_window = min(window, n)
    
    if actual_window <= 0 or n <= 0:
        return 0.0, 0.0
    
    # Scan backwards from start_idx, wrapping around if needed
    for i in range(actual_window):
        # Calculate index going backwards, wrapping around
        # Use modulo arithmetic: (start_idx - i) % n handles wrapping
        idx = (start_idx - i + n) % n
        
        # Update highest and lowest
        if highs[idx] > highest_high:
            highest_high = highs[idx]
        if lows[idx] < lowest_low:
            lowest_low = lows[idx]
    
    return highest_high, lowest_low


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_momentum_fast(
    double curr_close,
    double past_close
):
    """
    Fast momentum computation (price difference).
    
    Computes the simple momentum as the difference between current and past close.
    This is a trivial but typed and optimized function for Momentum node calculations.
    
    Parameters:
    - curr_close: Current close price
    - past_close: Past close price (from lookback periods ago)
    
    Returns:
    - momentum: curr_close - past_close
    """
    return curr_close - past_close


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_roc_fast(
    double curr_close,
    double past_close
):
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
    if past_close <= 0.0:
        return 0.0
    
    return ((curr_close - past_close) / past_close) * 100.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_rsi_initial_fast(
    double[::1] close_prices,
    Py_ssize_t lookback
):
    """
    Fast RSI initial computation (Cython equivalent of utils.compute.rsi_helpers.compute_rsi_initial).
    
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
    
    Note:
    - This matches the logic in utils/rsi_helpers.py compute_rsi_initial exactly
    - Initializes with 1e-60 to prevent division issues
    - Computes averages over (lookback - 1) price differences
    """
    cdef double upsum = 1e-60
    cdef double dnsum = 1e-60
    cdef double diff
    cdef Py_ssize_t i
    
    # Compute sum of gains and losses over initial period
    for i in range(1, lookback):
        diff = close_prices[i] - close_prices[i - 1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff  # Make positive
    
    # Convert to averages
    upsum /= (lookback - 1)
    dnsum /= (lookback - 1)
    
    return upsum, dnsum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def update_rsi_fast(
    double prev_close,
    double curr_close,
    double upsum,
    double dnsum,
    Py_ssize_t lookback
):
    """
    Fast RSI update computation (Cython equivalent of utils.compute.rsi_helpers.update_rsi).
    
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
    
    Note:
    - This matches the logic in utils/rsi_helpers.py update_rsi exactly
    - Uses exponential moving average: new_avg = ((period-1) * old_avg + new_value) / period
    - RSI = 100 * upsum / (upsum + dnsum)
    """
    cdef double diff = curr_close - prev_close
    cdef double new_upsum, new_dnsum, rsi
    
    if diff > 0.0:
        # Price went up
        new_upsum = ((lookback - 1) * upsum + diff) / lookback
        new_dnsum = dnsum * (lookback - 1.0) / lookback
    else:
        # Price went down
        new_dnsum = ((lookback - 1) * dnsum - diff) / lookback
        new_upsum = upsum * (lookback - 1.0) / lookback
    
    # Compute RSI
    rsi = 100.0 * new_upsum / (new_upsum + new_dnsum)
    
    return new_upsum, new_dnsum, rsi


# =============================================================================
# Ultimate C% and rolling helpers (pct_change, rolling max/min/mean)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_pct_change_fast(double prev_close, double curr_close):
    """
    One-period percent change: ((curr - prev) / prev) * 100.0.
    Returns 0.0 if prev_close <= 0.
    """
    if prev_close <= 0.0:
        return 0.0
    return ((curr_close - prev_close) / prev_close) * 100.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rolling_max_1d_fast(double[::1] values, Py_ssize_t n, Py_ssize_t window):
    """
    Rolling max over the last `window` elements of `values` (valid length `n`).
    Returns max of values[n-window:n] or 0.0 if n == 0.
    """
    if n <= 0:
        return 0.0
    cdef Py_ssize_t start = max(0, n - window)
    cdef double m = values[start]
    cdef Py_ssize_t i
    for i in range(start + 1, n):
        if values[i] > m:
            m = values[i]
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rolling_min_1d_fast(double[::1] values, Py_ssize_t n, Py_ssize_t window):
    """
    Rolling min over the last `window` elements of `values` (valid length `n`).
    """
    if n <= 0:
        return 0.0
    cdef Py_ssize_t start = max(0, n - window)
    cdef double m = values[start]
    cdef Py_ssize_t i
    for i in range(start + 1, n):
        if values[i] < m:
            m = values[i]
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rolling_mean_1d_fast(double[::1] values, Py_ssize_t n, Py_ssize_t window):
    """
    Rolling mean over the last `window` elements of `values` (valid length `n`).
    """
    if n <= 0:
        return 0.0
    cdef Py_ssize_t start = max(0, n - window)
    cdef Py_ssize_t count = n - start
    cdef double s = 0.0
    cdef Py_ssize_t i
    for i in range(start, n):
        s += values[i]
    return s / <double>count


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_ultimate_c_fast(
    double[::1] roc_values,
    Py_ssize_t n,
    Py_ssize_t lookback,
    double factor
):
    """
    Ultimate C% from ROC array. Same formula as ultimate_c.compute_ultimate_c.
    Returns 50.0 if n == 0.
    """
    if n == 0:
        return 50.0
    cdef double current_roc = roc_values[n - 1]
    cdef double high_short, low_short, high_med, low_med, high_long, low_long
    cdef double casey_c_short = 50.0, casey_c_med = 50.0, casey_c_long = 50.0
    cdef Py_ssize_t short_window = lookback
    cdef Py_ssize_t med_window = <Py_ssize_t>(lookback * factor)
    cdef Py_ssize_t long_window = <Py_ssize_t>(lookback * factor * factor)
    cdef double factor_sq = factor * factor
    cdef double denominator = factor_sq + factor + 1.0

    if n >= short_window:
        high_short = rolling_max_1d_fast(roc_values, n, short_window)
        low_short = rolling_min_1d_fast(roc_values, n, short_window)
        if high_short - low_short != 0.0:
            casey_c_short = ((current_roc - low_short) / (high_short - low_short)) * 100.0
    if n >= med_window:
        high_med = rolling_max_1d_fast(roc_values, n, med_window)
        low_med = rolling_min_1d_fast(roc_values, n, med_window)
        if high_med - low_med != 0.0:
            casey_c_med = ((current_roc - low_med) / (high_med - low_med)) * 100.0
    if n >= long_window:
        high_long = rolling_max_1d_fast(roc_values, n, long_window)
        low_long = rolling_min_1d_fast(roc_values, n, long_window)
        if high_long - low_long != 0.0:
            casey_c_long = ((current_roc - low_long) / (high_long - low_long)) * 100.0

    return ((casey_c_short * factor_sq) + (casey_c_med * factor) + casey_c_long) / denominator


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smooth_ultimate_c_fast(
    double[::1] ultimate_c_values,
    Py_ssize_t n,
    Py_ssize_t smooth_lookback
):
    """Rolling mean of ultimate_c_values over last smooth_lookback elements."""
    return rolling_mean_1d_fast(ultimate_c_values, n, smooth_lookback)


# =============================================================================
# Sample stddev (ddof=1) for EWSD long-run
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_stddev_sample_fast(double[::1] values, Py_ssize_t n):
    """
    Sample standard deviation (ddof=1). Returns 0.0 if n < 2.
    """
    if n < 2:
        return 0.0
    cdef double mean_val = fast_mean(values, n)
    cdef double sum_sq = 0.0
    cdef double diff
    cdef Py_ssize_t i
    for i in range(n):
        diff = values[i] - mean_val
        sum_sq += diff * diff
    return sqrt(sum_sq / <double>(n - 1))


# =============================================================================
# Normal CDF (for CMMA), return and ATR-from-slice
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def normal_cdf_fast(double x):
    """Standard normal CDF: 0.5 * (1 + erf(x / sqrt(2)))."""
    return 0.5 * (1.0 + erf(x / 1.4142135623730951))  # sqrt(2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_return_fast(double open_px, double close_px):
    """
    Returns (pct_return, log_return): ((close-open)/open)*100 and log(close/open).
    If open_px <= 0 returns (0.0, 0.0).
    """
    if open_px <= 0.0:
        return 0.0, 0.0
    cdef double pct = ((close_px - open_px) / open_px) * 100.0
    cdef double log_ret = 0.0
    if close_px > 0.0:
        log_ret = log(close_px / open_px)
    return pct, log_ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_atr_from_slice_fast(
    double[::1] highs,
    double[::1] lows,
    double[::1] closes,
    Py_ssize_t end_idx,
    Py_ssize_t period
):
    """
    ATR over the slice [end_idx - period + 1, end_idx] (inclusive).
    Uses contiguous arrays; end_idx is the last index. Returns 0.0 if period <= 0
    or slice would go before 0.
    """
    if period <= 0 or end_idx < period - 1:
        return 0.0
    cdef Py_ssize_t start = end_idx - period + 1
    cdef double tr_sum = 0.0
    cdef double hl, hc, lc, tr
    cdef Py_ssize_t i
    for i in range(start, end_idx + 1):
        if i == 0:
            tr = highs[0] - lows[0]
        else:
            hl = highs[i] - lows[i]
            hc = fabs(highs[i] - closes[i - 1])
            lc = fabs(lows[i] - closes[i - 1])
            tr = fast_max3(hl, hc, lc)
        tr_sum += tr
    return tr_sum / <double>period
