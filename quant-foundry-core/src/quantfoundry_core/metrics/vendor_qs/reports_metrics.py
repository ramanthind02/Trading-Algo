# Vendored from QuantStats (Apache-2.0). Metrics table only; no tearsheets / yfinance.
from __future__ import annotations

import re as _regex
from datetime import datetime as _dt
from math import ceil as _ceil, sqrt as _sqrt

import numpy as _np
import pandas as _pd
from dateutil.relativedelta import relativedelta
from tabulate import tabulate as _tabulate

from . import stats as _stats, utils as _utils

__version__ = "0.0.77-vendored"


def iDisplay(x):
    return None


def iHTML(x):
    return x


_plots = None


def _get_trading_periods(periods_per_year=252):
    """
    Calculate trading periods for different time windows.

    This helper function computes the number of trading periods for full year
    and half year periods, which are commonly used in financial calculations
    for annualization and rolling window analysis.

    Parameters
    ----------
    periods_per_year : int, default 252
        Number of trading periods in a year (e.g., 252 for daily data,
        12 for monthly data)

    Returns
    -------
    tuple
        A tuple containing (periods_per_year, half_year_periods)

    Examples
    --------
    >>> _get_trading_periods(252)  # Daily data
    (252, 126)
    >>> _get_trading_periods(12)   # Monthly data
    (12, 6)
    """
    # Calculate half year periods using ceiling to ensure we get at least half
    half_year = _ceil(periods_per_year / 2)
    return periods_per_year, half_year


def _match_dates(returns, benchmark):
    """
    Align returns and benchmark data to start from the same date.

    This function ensures that both the returns and benchmark series start
    from the same date by finding the latest start date where both series
    have non-zero values. This is crucial for accurate performance comparisons.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Returns data that may be a Series or DataFrame with multiple columns
    benchmark : pd.Series
        Benchmark returns data

    Returns
    -------
    tuple
        A tuple containing (aligned_returns, aligned_benchmark) both starting
        from the same date

    Examples
    --------
    >>> returns_aligned, bench_aligned = _match_dates(returns, benchmark)
    """
    # Handle different types of returns data (Series vs DataFrame)
    if isinstance(returns, _pd.DataFrame):
        # For DataFrame, use the first column to find the start date
        loc = max(returns[returns.columns[0]].ne(0).idxmax(), benchmark.ne(0).idxmax())
    else:
        # For Series, find the maximum of start dates for both series
        loc = max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax())

    # Slice both series to start from the latest common start date
    returns = returns.loc[loc:]
    benchmark = benchmark.loc[loc:]

    return returns, benchmark

def metrics(
    returns,
    benchmark=None,
    rf=0.0,
    display=True,
    mode="basic",
    sep=False,
    compounded=True,
    periods_per_year=252,
    prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    """
    Calculate comprehensive performance metrics for portfolio analysis.

    This function computes a wide range of performance metrics including
    returns, risk measures, ratios, and statistical measures. It can handle
    both single strategies and multiple strategy comparisons with optional
    benchmark analysis.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns data for the strategy/portfolio
    benchmark : pd.Series, str, or None, default None
        Benchmark returns for comparison
    rf : float, default 0.0
        Risk-free rate for calculations (as decimal)
    display : bool, default True
        Whether to display results in formatted table
    mode : str, default "basic"
        Analysis mode - "basic" for essential metrics, "full" for comprehensive
    sep : bool, default False
        Whether to include separator rows in output
    compounded : bool, default True
        Whether to compound returns for calculations
    periods_per_year : int, default 252
        Number of trading periods per year
    prepare_returns : bool, default True
        Whether to prepare/clean returns data
    match_dates : bool, default True
        Whether to align returns and benchmark start dates
    **kwargs
        Additional keyword arguments:
        - strategy_title: Custom name for the strategy
        - benchmark_title: Custom name for the benchmark
        - as_pct: Whether to return percentages
        - internal: Internal calculation flag

    Returns
    -------
    pd.DataFrame or None
        DataFrame with performance metrics if display=False, else None

    Examples
    --------
    >>> metrics_df = metrics(returns, benchmark='^GSPC', display=False)
    >>> metrics(returns, mode="full", rf=0.02)
    """
    # Clean returns data if date matching is enabled
    if match_dates:
        returns = returns.dropna()
    if isinstance(returns.index, _pd.DatetimeIndex) and returns.index.tz is not None:
        returns = returns.copy()
        returns.index = returns.index.tz_convert("UTC").tz_localize(None)

    # Get trading periods for annualization calculations
    win_year, _ = _get_trading_periods(periods_per_year)

    # Extract column names from kwargs or use defaults
    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")

    # Handle benchmark column naming
    if benchmark is not None:
        if isinstance(benchmark, str):
            benchmark_colname = f"Benchmark ({benchmark.upper()})"
        elif isinstance(benchmark, _pd.DataFrame) and len(benchmark.columns) > 1:
            raise ValueError(
                "`benchmark` must be a pandas Series, "
                "but a multi-column DataFrame was passed"
            )

    # Handle strategy column naming for multiple strategies
    if isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            blank = [""] * len(returns.columns)
            if isinstance(strategy_colname, str):
                strategy_colname = list(returns.columns)
    else:
        blank = [""]

    # if isinstance(returns, _pd.DataFrame):
    #     if len(returns.columns) > 1:
    #         raise ValueError("`returns` needs to be a Pandas Series or one column DataFrame. "
    #                          "multi colums DataFrame was passed")
    #     returns = returns[returns.columns[0]]

    # Prepare returns data if requested
    if prepare_returns:
        df = _utils._prepare_returns(returns)

    # Create main DataFrame for calculations
    if isinstance(returns, _pd.Series):
        df = _pd.DataFrame({"returns": returns})
    elif isinstance(returns, _pd.DataFrame):
        df = _pd.DataFrame(
            {
                "returns_" + str(i + 1): returns[strategy_col]
                for i, strategy_col in enumerate(returns.columns)
            }
        )

    # Process benchmark data if provided
    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = _match_dates(returns, benchmark)
        df["benchmark"] = benchmark
        # Update blank list for proper formatting
        if isinstance(returns, _pd.Series):
            blank = ["", ""]
            df["returns"] = returns
        elif isinstance(returns, _pd.DataFrame):
            blank = [""] * len(returns.columns) + [""]
            for i, strategy_col in enumerate(returns.columns):
                df["returns_" + str(i + 1)] = returns[strategy_col]

    # Calculate start and end dates for each series
    if isinstance(returns, _pd.Series):
        s_start = {"returns": df["returns"].index.strftime("%Y-%m-%d")[0]}
        s_end = {"returns": df["returns"].index.strftime("%Y-%m-%d")[-1]}
        s_rf = {"returns": rf}
    elif isinstance(returns, _pd.DataFrame):
        df_strategy_columns = [col for col in df.columns if col != "benchmark"]
        s_start = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[0]
            for strategy_col in df_strategy_columns
        }
        s_end = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[-1]
            for strategy_col in df_strategy_columns
        }
        s_rf = {strategy_col: rf for strategy_col in df_strategy_columns}

    # Add benchmark dates if present
    if "benchmark" in df:
        s_start["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[0]
        s_end["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[-1]
        s_rf["benchmark"] = rf

    # Fill missing values with zeros for calculations
    df = df.fillna(0)

    # Determine percentage multiplier for display
    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # Initialize metrics DataFrame with basic information
    metrics = _pd.DataFrame()
    metrics["Start Period"] = _pd.Series(s_start)
    metrics["End Period"] = _pd.Series(s_end)
    metrics["Risk-Free Rate %"] = _pd.Series(s_rf) * 100
    metrics["Time in Market %"] = _stats.exposure(df, prepare_returns=False) * pct

    # Add separator row
    metrics["~"] = blank

    # Calculate return metrics based on compounding preference
    if compounded:
        metrics["Cumulative Return %"] = (_stats.comp(df) * pct).map("{:,.2f}".format)
    else:
        metrics["Total Return %"] = (df.sum() * pct).map("{:,.2f}".format)

    # Calculate annualized return (CAGR)
    metrics["CAGR﹪%"] = _stats.cagr(df, rf, compounded, win_year) * pct

    # Add separator row
    metrics["~~~~~~~~~~~~~~"] = blank

    # Calculate risk-adjusted return ratios
    metrics["Sharpe"] = _stats.sharpe(df, rf, win_year, True)
    metrics["Prob. Sharpe Ratio %"] = (
        _stats.probabilistic_sharpe_ratio(df, rf, win_year, False) * pct
    )

    # Add advanced Sharpe metrics for full mode
    if mode.lower() == "full":
        metrics["Smart Sharpe"] = _stats.smart_sharpe(df, rf, win_year, True)
        # metrics['Prob. Smart Sharpe Ratio %'] = _stats.probabilistic_sharpe_ratio(df, rf, win_year, False, True) * pct

    # Calculate Sortino ratio (downside deviation-based)
    metrics["Sortino"] = _stats.sortino(df, rf, win_year, True)
    if mode.lower() == "full":
        # metrics['Prob. Sortino Ratio %'] = _stats.probabilistic_sortino_ratio(df, rf, win_year, False) * pct
        metrics["Smart Sortino"] = _stats.smart_sortino(df, rf, win_year, True)
        # metrics['Prob. Smart Sortino Ratio %'] = _stats.probabilistic_sortino_ratio(
        #     df, rf, win_year, False, True) * pct

    # Calculate adjusted Sortino ratio
    metrics["Sortino/√2"] = metrics["Sortino"] / _sqrt(2)
    if mode.lower() == "full":
        # metrics['Prob. Sortino/√2 Ratio %'] = _stats.probabilistic_adjusted_sortino_ratio(
        #     df, rf, win_year, False) * pct
        metrics["Smart Sortino/√2"] = metrics["Smart Sortino"] / _sqrt(2)
        # metrics['Prob. Smart Sortino/√2 Ratio %'] = _stats.probabilistic_adjusted_sortino_ratio(
        #     df, rf, win_year, False, True) * pct

    # Calculate Omega ratio (probability-weighted ratio)
    if isinstance(returns, _pd.Series):
        metrics["Omega"] = _stats.omega(df["returns"], rf, 0.0, win_year)
    elif isinstance(returns, _pd.DataFrame):
        omega_values = [
            _stats.omega(df[strategy_col], rf, 0.0, win_year)
            for strategy_col in df_strategy_columns
        ]
        if "benchmark" in df:
            omega_values.append(_stats.omega(df["benchmark"], rf, 0.0, win_year))
        metrics["Omega"] = omega_values

    # Add separator and prepare for drawdown metrics
    metrics["~~~~~~~~"] = blank
    metrics["Max Drawdown %"] = blank
    metrics["Max DD Date"] = blank
    metrics["Max DD Period Start"] = blank
    metrics["Max DD Period End"] = blank
    metrics["Longest DD Days"] = blank

    # Add detailed volatility and risk metrics for full mode
    if mode.lower() == "full":
        # Calculate annualized volatility
        if isinstance(returns, _pd.Series):
            ret_vol = (
                _stats.volatility(df["returns"], win_year, True, prepare_returns=False)
                * pct
            )
        elif isinstance(returns, _pd.DataFrame):
            ret_vol = [
                _stats.volatility(
                    df[strategy_col], win_year, True, prepare_returns=False
                )
                * pct
                for strategy_col in df_strategy_columns
            ]

        # Add benchmark volatility if present
        if "benchmark" in df:
            bench_vol = (
                _stats.volatility(
                    df["benchmark"], win_year, True, prepare_returns=False
                )
                * pct
            )

            vol_ = [ret_vol, bench_vol]
            if isinstance(ret_vol, list):
                metrics["Volatility (ann.) %"] = list(_pd.core.common.flatten(vol_))
            else:
                metrics["Volatility (ann.) %"] = vol_

            # Calculate benchmark-relative metrics
            if isinstance(returns, _pd.Series):
                metrics["R^2"] = _stats.r_squared(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
                metrics["Information Ratio"] = _stats.information_ratio(
                    df["returns"], df["benchmark"], prepare_returns=False
                )
            elif isinstance(returns, _pd.DataFrame):
                metrics["R^2"] = (
                    [
                        _stats.r_squared(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Information Ratio"] = (
                    [
                        _stats.information_ratio(
                            df[strategy_col], df["benchmark"], prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
        else:
            # No benchmark case
            if isinstance(returns, _pd.Series):
                metrics["Volatility (ann.) %"] = [ret_vol]
            elif isinstance(returns, _pd.DataFrame):
                metrics["Volatility (ann.) %"] = ret_vol

        # Additional risk and return metrics
        metrics["Calmar"] = _stats.calmar(df, prepare_returns=False, periods=win_year)
        metrics["Skew"] = _stats.skew(df, prepare_returns=False)
        metrics["Kurtosis"] = _stats.kurtosis(df, prepare_returns=False)

        # Add separator
        metrics["~~~~~~~~~~"] = blank

        # Expected returns at different frequencies
        metrics["Expected Daily %%"] = (
            _stats.expected_return(df, compounded=compounded, prepare_returns=False)
            * pct
        )
        metrics["Expected Monthly %%"] = (
            _stats.expected_return(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Expected Yearly %%"] = (
            _stats.expected_return(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        # Risk management metrics
        metrics["Kelly Criterion %"] = (
            _stats.kelly_criterion(df, prepare_returns=False) * pct
        )
        metrics["Risk of Ruin %"] = _stats.risk_of_ruin(df, prepare_returns=False)

        # Value at Risk metrics
        metrics["Daily Value-at-Risk %"] = -abs(
            _stats.var(df, prepare_returns=False) * pct
        )
        metrics["Expected Shortfall (cVaR) %"] = -abs(
            _stats.cvar(df, prepare_returns=False) * pct
        )

    # Add separator
    metrics["~~~~~~"] = blank

    # Consecutive wins/losses analysis (full mode only)
    if mode.lower() == "full":
        metrics["Max Consecutive Wins *int"] = _stats.consecutive_wins(df)
        metrics["Max Consecutive Losses *int"] = _stats.consecutive_losses(df)

    # Pain-based metrics (Gain/Pain ratio)
    metrics["Gain/Pain Ratio"] = _stats.gain_to_pain_ratio(df, rf)
    metrics["Gain/Pain (1M)"] = _stats.gain_to_pain_ratio(df, rf, "ME")
    # if mode.lower() == 'full':
    #     metrics['GPR (3M)'] = _stats.gain_to_pain_ratio(df, rf, "QE")
    #     metrics['GPR (6M)'] = _stats.gain_to_pain_ratio(df, rf, "2Q")
    #     metrics['GPR (1Y)'] = _stats.gain_to_pain_ratio(df, rf, "YE")

    # Add separator
    metrics["~~~~~~~"] = blank

    # Trading-based performance metrics
    metrics["Payoff Ratio"] = _stats.payoff_ratio(df, prepare_returns=False)
    metrics["Profit Factor"] = _stats.profit_factor(df, prepare_returns=False)
    metrics["Common Sense Ratio"] = _stats.common_sense_ratio(df, prepare_returns=False)
    metrics["CPC Index"] = _stats.cpc_index(df, prepare_returns=False)
    metrics["Tail Ratio"] = _stats.tail_ratio(df, prepare_returns=False)
    metrics["Outlier Win Ratio"] = _stats.outlier_win_ratio(df, prepare_returns=False)
    metrics["Outlier Loss Ratio"] = _stats.outlier_loss_ratio(df, prepare_returns=False)

    # # returns
    metrics["~~"] = blank

    # Time-based return analysis
    today = df.index[-1]  # _dt.today()
    m3 = today - relativedelta(months=3)
    m6 = today - relativedelta(months=6)
    y1 = today - relativedelta(years=1)

    # Calculate period returns based on compounding preference
    if compounded:
        metrics["MTD %"] = (
            _stats.comp(df[df.index >= _dt(today.year, today.month, 1)]) * pct
        )
        metrics["3M %"] = _stats.comp(df[df.index >= m3]) * pct
        metrics["6M %"] = _stats.comp(df[df.index >= m6]) * pct
        metrics["YTD %"] = _stats.comp(df[df.index >= _dt(today.year, 1, 1)]) * pct
        metrics["1Y %"] = _stats.comp(df[df.index >= y1]) * pct
    else:
        metrics["MTD %"] = (
            _np.sum(df[df.index >= _dt(today.year, today.month, 1)], axis=0) * pct
        )
        metrics["3M %"] = _np.sum(df[df.index >= m3], axis=0) * pct
        metrics["6M %"] = _np.sum(df[df.index >= m6], axis=0) * pct
        metrics["YTD %"] = _np.sum(df[df.index >= _dt(today.year, 1, 1)], axis=0) * pct
        metrics["1Y %"] = _np.sum(df[df.index >= y1], axis=0) * pct

    # Multi-year annualized returns
    d = today - relativedelta(months=35)
    metrics["3Y (ann.) %"] = (
        _stats.cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(months=59)
    metrics["5Y (ann.) %"] = (
        _stats.cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    d = today - relativedelta(years=10)
    metrics["10Y (ann.) %"] = (
        _stats.cagr(df[df.index >= d], 0.0, compounded, win_year) * pct
    )

    metrics["All-time (ann.) %"] = _stats.cagr(df, 0.0, compounded, win_year) * pct

    # Best/worst period analysis (full mode only)
    # best/worst
    if mode.lower() == "full":
        metrics["~~~"] = blank
        metrics["Best Day %"] = (
            _stats.best(df, compounded=compounded, prepare_returns=False) * pct
        )
        metrics["Worst Day %"] = _stats.worst(df, prepare_returns=False) * pct
        metrics["Best Month %"] = (
            _stats.best(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Month %"] = (
            _stats.worst(df, aggregate="ME", prepare_returns=False) * pct
        )
        metrics["Best Year %"] = (
            _stats.best(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )
        metrics["Worst Year %"] = (
            _stats.worst(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

    # Calculate and integrate drawdown metrics
    # return drawdown (dd) df
    dd = _calc_dd(
        df,
        display=(display or "internal" in kwargs),
        as_pct=kwargs.get("as_pct", False),
    )

    # Add drawdown metrics to main metrics DataFrame
    # drawdown (dd) detail
    metrics["~~~~"] = blank
    # Properly integrate drawdown data into metrics
    for metric_name in dd.index:
        metrics[metric_name] = dd.loc[metric_name].values

    # Additional drawdown-based metrics
    metrics["Recovery Factor"] = _stats.recovery_factor(df)
    metrics["Ulcer Index"] = _stats.ulcer_index(df)
    metrics["Serenity Index"] = _stats.serenity_index(df, rf)

    # Win rate analysis (full mode only)
    # win rate
    if mode.lower() == "full":
        metrics["~~~~~"] = blank
        metrics["Avg. Up Month %"] = (
            _stats.avg_win(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Avg. Down Month %"] = (
            _stats.avg_loss(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Win Days %%"] = _stats.win_rate(df, prepare_returns=False) * pct
        metrics["Win Month %%"] = (
            _stats.win_rate(
                df, compounded=compounded, aggregate="ME", prepare_returns=False
            )
            * pct
        )
        metrics["Win Quarter %%"] = (
            _stats.win_rate(
                df, compounded=compounded, aggregate="QE", prepare_returns=False
            )
            * pct
        )
        metrics["Win Year %%"] = (
            _stats.win_rate(
                df, compounded=compounded, aggregate="YE", prepare_returns=False
            )
            * pct
        )

        # Greek letters and correlation analysis (if benchmark exists)
        if "benchmark" in df:
            metrics["~~~~~~~~~~~~"] = blank
            if isinstance(returns, _pd.Series):
                # Calculate Greek letters (Beta, Alpha) for single strategy
                greeks = _stats.greeks(
                    df["returns"], df["benchmark"], win_year, prepare_returns=False
                )
                metrics["Beta"] = [str(round(greeks["beta"], 2)), "-"]
                metrics["Alpha"] = [str(round(greeks["alpha"], 2)), "-"]
                metrics["Correlation"] = [
                    str(round(df["benchmark"].corr(df["returns"]) * pct, 2)) + "%",
                    "-",
                ]
                metrics["Treynor Ratio"] = [
                    str(
                        round(
                            _stats.treynor_ratio(
                                df["returns"], df["benchmark"], win_year, rf
                            )
                            * pct,
                            2,
                        )
                    )
                    + "%",
                    "-",
                ]
            elif isinstance(returns, _pd.DataFrame):
                # Calculate Greek letters for multiple strategies
                greeks = [
                    _stats.greeks(
                        df[strategy_col],
                        df["benchmark"],
                        win_year,
                        prepare_returns=False,
                    )
                    for strategy_col in df_strategy_columns
                ]
                metrics["Beta"] = [str(round(g["beta"], 2)) for g in greeks] + ["-"]
                metrics["Alpha"] = [str(round(g["alpha"], 2)) for g in greeks] + ["-"]
                metrics["Correlation"] = (
                    [
                        str(round(df["benchmark"].corr(df[strategy_col]) * pct, 2))
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Treynor Ratio"] = (
                    [
                        str(
                            round(
                                _stats.treynor_ratio(
                                    df[strategy_col], df["benchmark"], win_year, rf
                                )
                                * pct,
                                2,
                            )
                        )
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]

    # Format metrics for display
    # prepare for display
    for col in metrics.columns:
        try:
            # Try to convert to float and round
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except (ValueError, TypeError, AttributeError):
            pass
        # Handle integer columns (marked with *int)
        if (display or "internal" in kwargs) and "*int" in col:
            metrics[col] = metrics[col].str.replace(".0", "", regex=False)
            metrics.rename({col: col.replace("*int", "")}, axis=1, inplace=True)
        # Add percentage signs to percentage columns
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + "%"

    # Format drawdown days as integers
    try:
        metrics["Longest DD Days"] = _pd.to_numeric(metrics["Longest DD Days"]).astype(
            "int"
        )
        metrics["Avg. Drawdown Days"] = _pd.to_numeric(
            metrics["Avg. Drawdown Days"]
        ).astype("int")

        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = metrics["Longest DD Days"].astype(str)
            metrics["Avg. Drawdown Days"] = metrics["Avg. Drawdown Days"].astype(str)
    except Exception:
        metrics["Longest DD Days"] = "-"
        metrics["Avg. Drawdown Days"] = "-"
        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = "-"
            metrics["Avg. Drawdown Days"] = "-"

    # Clean up column names (remove separators and percentage signs)
    metrics.columns = [col if "~" not in col else "" for col in metrics.columns]
    metrics.columns = [col[:-1] if "%" in col else col for col in metrics.columns]
    metrics = metrics.T

    # Set appropriate column names
    if "benchmark" in df:
        column_names = [strategy_colname, benchmark_colname]
        if isinstance(strategy_colname, list):
            metrics.columns = list(_pd.core.common.flatten(column_names))
        else:
            metrics.columns = column_names
    else:
        if isinstance(strategy_colname, list):
            metrics.columns = strategy_colname
        else:
            metrics.columns = [strategy_colname]

    # Final data cleaning
    # cleanups
    metrics.replace([-0, "-0"], 0, inplace=True)
    metrics.replace(
        [
            _np.nan,
            -_np.nan,
            _np.inf,
            -_np.inf,
            "-nan%",
            "nan%",
            "-nan",
            "nan",
            "-inf%",
            "inf%",
            "-inf",
            "inf",
        ],
        "-",
        inplace=True,
    )

    # Reorder columns to put benchmark first if present
    # move benchmark to be the first column always if present
    if "benchmark" in df:
        metrics = metrics[
            [benchmark_colname]
            + [col for col in metrics.columns if col != benchmark_colname]
        ]

    # Handle display vs return
    if display:
        print(_tabulate(metrics, headers="keys", tablefmt="simple"))
        return None

    # Remove separator rows if not requested
    if not sep:
        metrics = metrics[metrics.index != ""]

    # Final formatting for programmatic use
    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [
        c.replace(" %", "").replace(" *int", "").strip() for c in metrics.columns
    ]
    metrics = metrics.T

    return metrics

def _calc_dd(df, display=True, as_pct=False):
    """
    Calculate drawdown statistics for performance analysis.

    This helper function computes comprehensive drawdown statistics including
    maximum drawdown, drawdown dates, recovery periods, and average drawdown
    metrics. It handles both single strategy and multiple strategy analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing returns data with columns for strategies
        and optionally benchmark
    display : bool, default True
        Whether the output is for display purposes (affects formatting)
    as_pct : bool, default False
        Whether to return percentages instead of decimals

    Returns
    -------
    pd.DataFrame
        DataFrame with drawdown statistics including:
        - Max Drawdown %: Maximum drawdown percentage
        - Max DD Date: Date of maximum drawdown
        - Max DD Period Start: Start date of worst drawdown period
        - Max DD Period End: End date of worst drawdown period
        - Longest DD Days: Duration of longest drawdown in days
        - Avg. Drawdown %: Average drawdown percentage
        - Avg. Drawdown Days: Average drawdown duration in days

    Examples
    --------
    >>> dd_stats = _calc_dd(returns_df, display=False)
    >>> dd_stats = _calc_dd(returns_df, as_pct=True)
    """
    # Convert returns to drawdown series
    dd = _stats.to_drawdown_series(df)
    dd_info = _stats.drawdown_details(dd)

    # Return empty DataFrame if no drawdowns found
    if dd_info.empty:
        return _pd.DataFrame()

    # Handle different column structures based on data type
    if "returns" in dd_info:
        ret_dd = dd_info["returns"]
    # to match multiple columns like returns_1, returns_2, ...
    elif (
        any(dd_info.columns.get_level_values(0).str.contains("returns"))
        and dd_info.columns.get_level_values(0).nunique() > 1
    ):
        ret_dd = dd_info.loc[
            :, dd_info.columns.get_level_values(0).str.contains("returns")
        ]
    else:
        ret_dd = dd_info

    # Calculate drawdown statistics based on data structure
    if (
        any(ret_dd.columns.get_level_values(0).str.contains("returns"))
        and ret_dd.columns.get_level_values(0).nunique() > 1
    ):
        # Multiple strategy columns case
        dd_stats = {
            col: {
                "Max Drawdown %": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["max drawdown"]
                .values[0]
                / 100,
                "Max DD Date": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["valley"]
                .values[0],
                "Max DD Period Start": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["start"]
                .values[0],
                "Max DD Period End": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["end"]
                .values[0],
                "Longest DD Days": str(
                    _np.round(
                        ret_dd[col]
                        .sort_values(by="days", ascending=False)["days"]
                        .values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd[col]["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd[col]["days"].mean())),
            }
            for col in ret_dd.columns.get_level_values(0)
        }
    else:
        # Single strategy case
        max_dd = ret_dd.sort_values(by="max drawdown", ascending=True)
        dd_stats = {
            "returns": {
                "Max Drawdown %": max_dd["max drawdown"].values[0] / 100,
                "Max DD Date": max_dd["valley"].values[0],
                "Max DD Period Start": max_dd["start"].values[0],
                "Max DD Period End": max_dd["end"].values[0],
                "Longest DD Days": str(
                    _np.round(
                        ret_dd.sort_values(by="days", ascending=False)["days"].values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(_np.round(ret_dd["days"].mean())),
            }
        }

    # Add benchmark drawdown statistics if present
    if "benchmark" in df and isinstance(dd_info.columns, _pd.MultiIndex):
        bench_dd = dd_info["benchmark"].sort_values(by="max drawdown")
        dd_stats["benchmark"] = {
            "Max Drawdown %": bench_dd.sort_values(by="max drawdown", ascending=True)[
                "max drawdown"
            ].values[0]
            / 100,
            "Max DD Date": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["valley"].values[0],
            "Max DD Period Start": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["start"].values[0],
            "Max DD Period End": bench_dd.sort_values(
                by="max drawdown", ascending=True
            )["end"].values[0],
            "Longest DD Days": str(
                _np.round(
                    bench_dd.sort_values(by="days", ascending=False)["days"].values[0]
                )
            ),
            "Avg. Drawdown %": bench_dd["max drawdown"].mean() / 100,
            "Avg. Drawdown Days": str(_np.round(bench_dd["days"].mean())),
        }

    # Apply percentage multiplier based on display settings
    # pct multiplier
    pct = 100 if display or as_pct else 1

    # Convert to DataFrame and apply percentage formatting
    dd_stats = _pd.DataFrame(dd_stats).T
    dd_stats["Max Drawdown %"] = dd_stats["Max Drawdown %"].astype(float) * pct
    dd_stats["Avg. Drawdown %"] = dd_stats["Avg. Drawdown %"].astype(float) * pct

    return dd_stats.T
