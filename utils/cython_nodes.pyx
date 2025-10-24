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
from libc.math cimport sqrt, fabs, fmax
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
