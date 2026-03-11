"""
QuantStats Tearsheet Generation for Walk-Forward Analysis

Simple wrapper to generate QuantStats tearsheets from walk-forward results.

Author: Trading Research Team
Date: 2025-10-24
"""

import pandas as pd
from typing import Optional
import warnings
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


def generate_tearsheet(
    strategy_returns: pd.Series,
    baseline_returns: Optional[pd.Series] = None,
    feature_name: str = "Strategy",
    output_file: Optional[str] = None,
    mode: str = "full",
    timeframe: TimeFrame = TimeFrame.D,
):
    """
    Generate QuantStats tearsheet for walk-forward analysis results.
    
    Uses QuantStats to create professional tearsheets with comprehensive metrics
    and visualizations. Compares strategy returns against baseline (always-in).
    
    IMPORTANT: This function expects DAILY returns, not aggregated walk-forward results!
    
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
    
    # #region agent log
    try:
        _n = len(strategy_returns)
        _zero = (strategy_returns == 0.0).sum()
        _bm_n = len(baseline_returns) if baseline_returns is not None else 0
        with open("/home/raman/repos/Trading-Algo/.cursor/debug.log", "a") as _f:
            import json
            _f.write(
                json.dumps(
                    {
                        "hypothesisId": "A,C",
                        "location": "quantstats_reports.generate_tearsheet",
                        "message": "strategy_returns passed to QuantStats",
                        "data": {"strategy_n": _n, "pct_strategy_zero": float(_zero) / _n if _n else 0, "baseline_n": _bm_n},
                        "timestamp": __import__("time").time() * 1000,
                    },
                    default=str,
                )
                + "\n"
            )
    except Exception:  # noqa: S110
        pass
    # #endregion

    # Generate tearsheet based on mode
    # IMPORTANT: Use match_dates=False to prevent timezone comparison errors
    # IMPORTANT: Use compounded=False to follow Robert Carver's methodology
    #            (non-compounded/summed percentage returns, not geometric compounding)
    if mode == 'html':
        # Generate HTML tearsheet
        if output_file is None:
            output_file = f"{feature_name.replace(' ', '_')}_tearsheet.html"
        
        qs.reports.html(
            strategy_returns,
            benchmark=benchmark,
            output=output_file,
            title=f"{feature_name} - Walk-Forward Analysis",
            match_dates=False,  # Prevent timezone comparison issues
            compounded=False    # Use non-compounded returns (Carver methodology)
        )
        print(f"\n[OK] HTML tearsheet saved to: {output_file}")
        print("   Note: Using non-compounded returns (Robert Carver methodology)")
        
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
