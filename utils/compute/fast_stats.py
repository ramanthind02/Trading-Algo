"""
Fast statistics module with automatic fallback.

Provides optimized MA-diff computation for bias nodes. Uses Cython when
compiled, otherwise falls back to pure Python.
"""

from __future__ import annotations

from typing import Deque

import numpy as np

try:
    from utils.compute.cython.cython_optimized import (
        compute_ma_diff_fast as _cython_ma_diff,
    )

    CYTHON_AVAILABLE = True
    print("✓ Cython optimizations loaded successfully (20-50x speedup)")
except ImportError:
    CYTHON_AVAILABLE = False
    _cython_ma_diff = None  # type: ignore[assignment]
    print(
        "⚠ Cython optimizations not available, using pure Python "
        "(compile with: python utils/compute/cython/setup_cython.py build_ext --inplace)"
    )


def compute_ma_diff_fast(
    log_close: float,
    log_closes: Deque[float],
    true_ranges: Deque[float],
    lookback: int,
    compression: float,
) -> float:
    """
    Fast computation of MA diff feature.

    Computes the difference between the current log close price and a moving average,
    normalized by ATR and scaled to be centered around 0.
    """
    if CYTHON_AVAILABLE and _cython_ma_diff is not None:
        return _cython_ma_diff(log_close, log_closes, true_ranges, lookback, compression)
    return _python_ma_diff(log_close, log_closes, true_ranges, lookback, compression)


def _python_ma_diff(
    log_close: float,
    log_closes: Deque[float],
    true_ranges: Deque[float],
    lookback: int,
    compression: float,
) -> float:
    """Pure Python implementation of MA diff computation."""
    log_closes_list = list(log_closes)[:-1]
    log_ma = np.mean(log_closes_list) if log_closes_list else 0.0
    atr = np.mean(list(true_ranges)) if len(true_ranges) > 0 else 0.0

    if atr > 0.0:
        denom = atr * np.sqrt(lookback + 1.0)
        diff = (log_close - log_ma) / denom
        from scipy.stats import norm

        return 100.0 * norm.cdf(compression * diff) - 50.0
    return 0.0
