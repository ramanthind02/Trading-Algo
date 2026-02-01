"""
Fast statistics module with automatic fallback.

This module provides optimized implementations of statistical functions.
It will automatically use Cython implementations if available, otherwise
falls back to pure Python/NumPy implementations.

Usage:
    from utils.fast_stats import spearman_rho, rank_with_tie_correction
    
    # These will use Cython if compiled, otherwise pure Python
    rho = spearman_rho(x, y)
    ranks, tie_corr = rank_with_tie_correction(arr)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Deque
import collections

# Try to import Cython optimized versions
try:
    from utils.cython_optimized import (
        spearman_rho as _cython_spearman_rho,
        rank_with_tie_correction as _cython_rank_with_tie_correction,
        optimize_threshold_fast as _cython_optimize_threshold,
        compute_ma_diff_fast as _cython_ma_diff,
    )
    CYTHON_AVAILABLE = True
    print("✓ Cython optimizations loaded successfully (20-50x speedup)")
except ImportError:
    CYTHON_AVAILABLE = False
    _cython_ma_diff = None  # type: ignore[assignment]
    print("⚠ Cython optimizations not available, using pure Python (compile with: python utils/setup_cython.py build_ext --inplace)")


def rank_with_tie_correction(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Rank array and compute tie correction factor.
    
    Automatically uses Cython implementation if available (20-50x faster),
    otherwise falls back to pure Python.
    
    Parameters:
    - arr: Input array to rank
    
    Returns:
    - ranks: Array of ranks
    - tie_correction: Tie correction sum for Spearman correlation
    """
    if CYTHON_AVAILABLE:
        return _cython_rank_with_tie_correction(np.asarray(arr, dtype=np.float64))
    else:
        # Pure Python fallback
        return _python_rank_with_tie_correction(arr)


def spearman_rho(var1, var2) -> float:
    """
    Compute Spearman Rho correlation coefficient.
    
    Automatically uses Cython implementation if available (15-30x faster),
    otherwise falls back to pure Python.
    
    Parameters:
    - var1: First variable (pandas Series or numpy array)
    - var2: Second variable (pandas Series or numpy array)
    
    Returns:
    - rho: Spearman Rho correlation coefficient in range [-1, 1]
    """
    # Convert to numpy arrays
    if isinstance(var1, pd.Series):
        x_vals = var1.values
    else:
        x_vals = np.asarray(var1)
    
    if isinstance(var2, pd.Series):
        y_vals = var2.values
    else:
        y_vals = np.asarray(var2)
    
    # Ensure float64
    x_vals = x_vals.astype(np.float64)
    y_vals = y_vals.astype(np.float64)
    
    if CYTHON_AVAILABLE:
        return _cython_spearman_rho(x_vals, y_vals)
    else:
        # Pure Python fallback
        return _python_spearman_rho(x_vals, y_vals)


def optimize_threshold_fast(feature_vals, target_vals, floor: float, n_thresholds: int = 100):
    """
    Fast threshold optimization.
    
    Automatically uses Cython implementation if available (5-10x faster),
    otherwise falls back to pure Python.
    
    Parameters:
    - feature_vals: Feature values
    - target_vals: Target values
    - floor: Minimum fraction of cases that must trade
    - n_thresholds: Number of thresholds to test
    
    Returns:
    - best_threshold: Optimal threshold value
    - best_pf: Best profit factor achieved
    - best_direction: 'long' or 'short'
    """
    # Convert to numpy arrays
    if isinstance(feature_vals, pd.Series):
        feature_vals = feature_vals.values
    if isinstance(target_vals, pd.Series):
        target_vals = target_vals.values
    
    feature_vals = np.asarray(feature_vals, dtype=np.float64)
    target_vals = np.asarray(target_vals, dtype=np.float64)
    
    if CYTHON_AVAILABLE:
        return _cython_optimize_threshold(feature_vals, target_vals, floor, n_thresholds)
    else:
        # Import and use existing Python implementation
        from feature_selection.opt_thresh import optimize_threshold
        return optimize_threshold(
            pd.Series(feature_vals),
            pd.Series(target_vals),
            floor
        )


def compute_ma_diff_fast(
    log_close: float,
    log_closes: Deque[float],
    true_ranges: Deque[float],
    lookback: int,
    compression: float
) -> float:
    """
    Fast computation of MA diff feature.
    
    Computes the difference between the current log close price and a moving average,
    normalized by ATR and scaled to be centered around 0. Automatically uses Cython
    implementation if available (15-25x faster), otherwise falls back to pure Python.
    
    Parameters:
    - log_close: Current log(close) value
    - log_closes: Deque of historical log closes
    - true_ranges: Deque of historical true ranges
    - lookback: Lookback period for MA
    - compression: Compression factor for output
    
    Returns:
    - output_value: The MA diff feature value in range [-50, 50]
    """
    if CYTHON_AVAILABLE and _cython_ma_diff is not None:
        return _cython_ma_diff(log_close, log_closes, true_ranges, lookback, compression)
    else:
        return _python_ma_diff(log_close, log_closes, true_ranges, lookback, compression)


# ============================================================================
# PURE PYTHON FALLBACK IMPLEMENTATIONS
# ============================================================================

def _python_rank_with_tie_correction(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Pure Python implementation of rank_with_tie_correction."""
    n = len(arr)
    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]
    
    ranks = np.empty(n, dtype=np.float64)
    tie_correction = 0.0
    
    j = 0
    while j < n:
        val = sorted_arr[j]
        k = j + 1
        while k < n and sorted_arr[k] == val:
            k += 1
        
        ntied = k - j
        tie_correction += ntied * ntied * ntied - ntied
        rank = 0.5 * (j + k + 1.0)
        
        for idx in range(j, k):
            ranks[sorted_indices[idx]] = rank
        
        j = k
    
    return ranks, tie_correction


def _python_spearman_rho(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    """Pure Python implementation of spearman_rho."""
    n = len(x_vals)
    
    if len(y_vals) != n:
        raise ValueError("Arrays must have the same length")
    
    if n < 2:
        raise ValueError("Need at least 2 valid data points to compute correlation")
    
    x_ranks, x_tie_correc = _python_rank_with_tie_correction(x_vals)
    y_ranks, y_tie_correc = _python_rank_with_tie_correction(y_vals)
    
    dn = float(n)
    ssx = (dn * dn * dn - dn - x_tie_correc) / 12.0
    ssy = (dn * dn * dn - dn - y_tie_correc) / 12.0
    
    rank_diff = x_ranks - y_ranks
    rankerr = np.sum(rank_diff * rank_diff)
    
    denominator = np.sqrt(ssx * ssy + 1.0e-20)
    rho = 0.5 * (ssx + ssy - rankerr) / denominator
    
    return rho


def _python_ma_diff(
    log_close: float,
    log_closes: Deque[float],
    true_ranges: Deque[float],
    lookback: int,
    compression: float
) -> float:
    """Pure Python implementation of MA diff computation."""
    # Compute moving average of log closes (excluding current candle)
    log_closes_list = list(log_closes)[:-1]  # Use all but the last (current) value
    log_ma = np.mean(log_closes_list) if log_closes_list else 0.0
    
    # Compute ATR
    atr = np.mean(list(true_ranges)) if len(true_ranges) > 0 else 0.0
    
    # Compute normalized difference
    if atr > 0.0:
        denom = atr * np.sqrt(lookback + 1.0)
        diff = (log_close - log_ma) / denom
        
        # Transform through normal CDF and scale to [-50, 50]
        from scipy.stats import norm
        output_value = 100.0 * norm.cdf(compression * diff) - 50.0
    else:
        output_value = 0.0
    
    return output_value
