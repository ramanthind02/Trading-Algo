"""
Fast volatility utilities with optional Cython acceleration.

This module provides array-based helpers for computing volatility measures
used at the portfolio level (e.g., EWSD-style blended volatility).

It is intentionally independent of candles/nodes – call it with NumPy arrays
of prices or returns from higher-level components such as Portfolio.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from .fast_nodes import batch_compute_log_returns


LAMBDA_SHORT_DEFAULT: Final[float] = 0.06061  # 32-day span (Carver)
LONG_RUN_WINDOW_DEFAULT: Final[int] = 2520    # 10 years of daily data
BLEND_SHORT_WEIGHT_DEFAULT: Final[float] = 0.7
BLEND_LONG_WEIGHT_DEFAULT: Final[float] = 0.3


def _ewma_variance(returns_sq: np.ndarray, lambda_short: float) -> float:
    """
    Compute EWMA variance of squared returns.

    Parameters
    ----------
    returns_sq : np.ndarray
        1D array of squared returns.
    lambda_short : float
        EWMA smoothing parameter.

    Returns
    -------
    float
        Final EWMA variance.
    """
    if returns_sq.size == 0:
        return 0.0

    var = float(returns_sq[0])
    one_minus_lambda = 1.0 - lambda_short

    for value in returns_sq[1:]:
        var = lambda_short * var + one_minus_lambda * float(value)

    return var


def compute_ewsd_annualized_from_closes(
    closes: np.ndarray,
    lambda_short: float = LAMBDA_SHORT_DEFAULT,
    long_run_window: int = LONG_RUN_WINDOW_DEFAULT,
    blend_short_weight: float = BLEND_SHORT_WEIGHT_DEFAULT,
    blend_long_weight: float = BLEND_LONG_WEIGHT_DEFAULT,
) -> float:
    """
    Compute Carver-style blended EWSD volatility from close prices.

    This mirrors the EWSDNode behaviour conceptually:
    - short-run EWMA of squared returns
    - long-run standard deviation of returns
    - blended daily sigma = w_short * sigma_short + w_long * sigma_long
    - annualized by multiplying daily sigma by 16 (approx sqrt(256))

    Parameters
    ----------
    closes : np.ndarray
        1D array of close prices for a single instrument, ordered by time.
    lambda_short : float, optional
        Smoothing parameter for short-run EWMA variance.
    long_run_window : int, optional
        Lookback window for long-run historical volatility.
    blend_short_weight : float, optional
        Weight for the short-run estimate.
    blend_long_weight : float, optional
        Weight for the long-run estimate.

    Returns
    -------
    float
        Annualized blended volatility as a decimal (e.g., 0.20 for 20%).
    """
    prices = np.asarray(closes, dtype=np.float64)
    if prices.size < 2:
        return 0.20  # conservative default

    returns = batch_compute_log_returns(prices)
    if returns.size == 0:
        return 0.20

    # Short-run EWMA variance of squared returns
    returns_sq = returns * returns
    ewma_var = _ewma_variance(returns_sq, lambda_short=lambda_short)
    sigma_short = float(np.sqrt(ewma_var)) if ewma_var > 0.0 else 0.0

    # Long-run standard deviation over the chosen window
    if returns.size >= 2:
        window = min(long_run_window, returns.size)
        long_slice = returns[-window:]
        sigma_long = float(np.std(long_slice, ddof=1)) if window > 1 else float(np.std(long_slice))
    else:
        sigma_long = sigma_short

    # Blend short and long estimates
    blended_daily = blend_short_weight * sigma_short + blend_long_weight * sigma_long

    if not np.isfinite(blended_daily) or blended_daily <= 0.0:
        return 0.20

    # EWSDNode annualizes daily volatility by multiplying by 16
    annual_sigma = blended_daily * 16.0
    return float(annual_sigma)

