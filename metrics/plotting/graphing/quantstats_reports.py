"""
QuantStats Tearsheet Generation for Walk-Forward Analysis

Simple wrapper to generate QuantStats tearsheets from walk-forward results.

Author: Trading Research Team
Date: 2025-10-24
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.cache.runtime.cache_paths import win32_extended_path
from utils.core.enums import TimeFrame

# Import QuantStats
try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False
    warnings.warn("QuantStats not installed. Install with: pip install quantstats")


def _resample_to_daily_if_needed(returns: pd.Series, tf: TimeFrame) -> pd.Series:
    """For sub-daily timeframes, sum returns within each day to produce a daily series."""
    if tf in (TimeFrame.H1, TimeFrame.H4):
        return returns.resample("D").sum().dropna(how="all")
    return returns


def _normalize_returns_series(returns: pd.Series, timeframe: TimeFrame) -> pd.Series:
    """Normalize a returns series for QuantStats consumption."""
    normalized = returns.copy()
    if hasattr(normalized.index, "tz") and normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized = _resample_to_daily_if_needed(normalized, timeframe)
    normalized = normalized.dropna().sort_index()
    return normalized


def vol_scale_returns_to_target_annualized_volatility(
    returns: pd.Series,
    *,
    target_annual_volatility: float,
    bars_per_year: int,
    min_observations: int = 5,
) -> pd.Series:
    """Linearly scale a per-bar return series so sample annualized vol matches ``target_annual_volatility``.

    Uses ``std(ddof=1) * sqrt(bars_per_year)`` as realized annual volatility. Sharpe ratio
    (mean / std) is unchanged; mean return and volatility scale together with the strategy series.
    """
    if target_annual_volatility <= 0 or not math.isfinite(target_annual_volatility):
        raise ValueError("target_annual_volatility must be a finite positive number")
    if bars_per_year <= 0:
        raise ValueError("bars_per_year must be positive")

    clean = returns.dropna()
    if len(clean) < min_observations:
        warnings.warn(
            f"Skipping vol scale: need at least {min_observations} non-null returns, got {len(clean)}.",
            UserWarning,
            stacklevel=2,
        )
        return returns.copy()

    per_bar_std = float(clean.std(ddof=1))
    if per_bar_std <= 0 or not math.isfinite(per_bar_std):
        warnings.warn(
            "Skipping vol scale: per-bar standard deviation is zero or non-finite.",
            UserWarning,
            stacklevel=2,
        )
        return returns.copy()

    realized_annual = per_bar_std * math.sqrt(float(bars_per_year))
    if realized_annual <= 0 or not math.isfinite(realized_annual):
        warnings.warn(
            "Skipping vol scale: realized annual volatility is non-finite.",
            UserWarning,
            stacklevel=2,
        )
        return returns.copy()

    scale = target_annual_volatility / realized_annual
    return returns * scale


def generate_tearsheet(
    strategy_returns: pd.Series,
    baseline_returns: Optional[pd.Series] = None,
    feature_name: str = "Strategy",
    output_file: Optional[str] = None,
    mode: str = "full",
    timeframe: TimeFrame = TimeFrame.D,
    target_annual_volatility: float | None = None,
):
    """
    Generate QuantStats tearsheet for walk-forward analysis results.
    
    Uses QuantStats to create professional tearsheets with comprehensive metrics
    and visualizations. Compares strategy returns against baseline (always-in).
    
    IMPORTANT: This function expects DAILY returns, not aggregated walk-forward results!

    When ``target_annual_volatility`` is set (e.g. ``0.10`` for 10% annual vol), strategy
    returns are linearly scaled after normalization so sample annualized volatility matches
    that target. The benchmark series is not scaled.

    Parameters
    ----------
    strategy_returns : pd.Series
        DAILY returns series for the strategy (datetime index, daily frequency)
    baseline_returns : Optional[pd.Series], default=None
        DAILY returns series for baseline (always-in strategy). If None, no benchmark.
    feature_name : str, default="Strategy"
        Name of the strategy for report title
    output_file : Optional[str], default=None
        If provided, saves HTML report to this file. Otherwise displays in notebook.
    mode : str, default="full"
        Tearsheet mode:
        - 'html': Generate HTML report
        - 'full': Display full tearsheet in notebook
        - 'basic': Display basic tearsheet in notebook
        - 'metrics': Display metrics only
    timeframe : TimeFrame, default=TimeFrame.D
        Bar frequency for resampling (when needed) and for ``bars_per_year`` when vol-scaling.
    target_annual_volatility : float or None, default=None
        If set to a positive value, linearly scale strategy returns to this annualized volatility.

    Returns
    -------
    None
        Displays or saves QuantStats tearsheet
        
    Examples
    --------
    >>> # Display full tearsheet in notebook
    >>> generate_tearsheet(
    ...     results_df=strategy_results,
    ...     baseline_df=baseline_results,
    ...     feature_name='EWMAC 16/64'
    ... )
    >>> 
    >>> # Save HTML report
    >>> generate_tearsheet(
    ...     results_df=strategy_results,
    ...     baseline_df=baseline_results,
    ...     feature_name='EWMAC 16/64',
    ...     output_file='reports/ewmac_tearsheet.html',
    ...     mode='html'
    ... )
    """
    if not HAS_QUANTSTATS:
        raise ImportError(
            "QuantStats is required for tearsheet generation. "
            "Install with: pip install quantstats"
        )

    # Validate inputs - must be daily returns series
    if not isinstance(strategy_returns, pd.Series):
        raise TypeError("strategy_returns must be a pandas Series of daily returns")
    
    if not isinstance(strategy_returns.index, pd.DatetimeIndex):
        raise TypeError("strategy_returns must have a DatetimeIndex")
    
    strategy_returns = _normalize_returns_series(strategy_returns, timeframe)
    if strategy_returns.empty:
        warnings.warn(
            f"Skipping tearsheet for '{feature_name}' because strategy_returns is empty."
        )
        return

    resolved_vol_target: float | None = None
    if target_annual_volatility is not None:
        try:
            _v = float(target_annual_volatility)
        except (TypeError, ValueError):
            _v = 0.0
        resolved_vol_target = _v if _v > 0.0 else None

    if resolved_vol_target is not None:
        strategy_returns = vol_scale_returns_to_target_annualized_volatility(
            strategy_returns,
            target_annual_volatility=resolved_vol_target,
            bars_per_year=timeframe.bars_per_year,
        )

    strategy_returns.name = feature_name

    # Process baseline if provided
    benchmark = None
    if baseline_returns is not None:
        if not isinstance(baseline_returns, pd.Series):
            raise TypeError("baseline_returns must be a pandas Series of daily returns")
        
        baseline_returns = _normalize_returns_series(baseline_returns, timeframe)
        if not baseline_returns.empty:
            baseline_returns.name = "Baseline (Always-In)"
            benchmark = baseline_returns

    # Generate tearsheet based on mode
    # IMPORTANT: Use match_dates=False to prevent timezone comparison errors
    # IMPORTANT: Use compounded=False to follow Robert Carver's methodology
    #            (non-compounded/summed percentage returns, not geometric compounding)
    if mode == 'html':
        # Generate HTML tearsheet
        if output_file is None:
            output_file = f"{feature_name.replace(' ', '_')}_tearsheet.html"

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_out = out_path.resolve()
        qs_output_path = (
            win32_extended_path(resolved_out) if os.name == "nt" else str(resolved_out)
        )

        qs.reports.html(
            strategy_returns,
            benchmark=benchmark,
            output=qs_output_path,
            title=f"{feature_name} - Walk-Forward Analysis",
            match_dates=False,  # Prevent timezone comparison issues
            compounded=False    # Use non-compounded returns (Carver methodology)
        )
        print(f"\n[OK] HTML tearsheet saved to: {resolved_out}")
        print("   Note: Using non-compounded returns (Robert Carver methodology)")
        if resolved_vol_target is not None:
            print(
                f"   Strategy returns vol-scaled to ~{resolved_vol_target:.0%} annualized "
                f"({timeframe.name}, {timeframe.bars_per_year} bars/year)."
            )
        
    elif mode == 'full':
        # Display full tearsheet in notebook
        qs.reports.full(
            strategy_returns,
            benchmark=benchmark,
            match_dates=False,  # Prevent timezone comparison issues
            compounded=False    # Use non-compounded returns (Carver methodology)
        )
        
    elif mode == 'basic':
        # Display basic tearsheet in notebook
        qs.reports.basic(
            strategy_returns,
            benchmark=benchmark,
            match_dates=False,  # Prevent timezone comparison issues
            compounded=False    # Use non-compounded returns (Carver methodology)
        )
        
    elif mode == 'metrics':
        # Display metrics only
        qs.reports.metrics(
            strategy_returns,
            benchmark=benchmark,
            mode='full',
            match_dates=False,  # Prevent timezone comparison issues
            compounded=False    # Use non-compounded returns (Carver methodology)
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            f"Use 'html', 'full', 'basic', or 'metrics'"
        )


def compute_baseline_results(
    results_df: pd.DataFrame,
    all_returns: pd.Series,
    objective_metric: 'ObjectiveMetric'
) -> pd.DataFrame:
    """
    Compute baseline "always-in" results for comparison.
    
    The baseline strategy predicts 1 for all rows (always taking the trade),
    representing a simple buy-and-hold approach.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Original strategy results with columns: 'step', 'test_start', 'test_end'
    all_returns : pd.Series
        All returns data (indexed by datetime)
    objective_metric : ObjectiveMetric
        Metric instance to compute (e.g., SortinoRatio, SharpeRatio)
        
    Returns
    -------
    pd.DataFrame
        Baseline results with same structure as results_df
        
    Examples
    --------
    >>> from metrics.performance import SortinoRatio
    >>> 
    >>> baseline_df = compute_baseline_results(
    ...     results_df=strategy_results,
    ...     all_returns=target_data,
    ...     objective_metric=SortinoRatio(annualization_factor=252)
    ... )
    """
    baseline_results = []
    
    for _, row in results_df.iterrows():
        # Get test period returns
        test_start = row['test_start']
        test_end = row['test_end']
        
        # Filter returns for this test period
        test_returns = all_returns[(all_returns.index >= test_start) & 
                                   (all_returns.index <= test_end)]
        
        # Baseline: all returns (always in the market)
        if len(test_returns) > 0:
            test_metric = objective_metric.compute(test_returns)
            n_trades = len(test_returns)
            mean_return = test_returns.mean()
        else:
            test_metric = 0.0
            n_trades = 0
            mean_return = 0.0
        
        baseline_results.append({
            'step': row['step'],
            'test_start': test_start,
            'test_end': test_end,
            'test_metric': test_metric,
            'n_trades': n_trades,
            'mean_return': mean_return,
            'strategy': 'baseline'
        })
    
    return pd.DataFrame(baseline_results)
