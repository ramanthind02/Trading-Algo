"""Catalog of metric row names produced by QuantStats ``reports.metrics`` (programmatic).

Rows depend on ``mode``, ``compounded``, and whether a benchmark is passed.
Reference version: QuantStats 0.0.77 (Apache-2.0).
"""

from __future__ import annotations

QUANTSTATS_METRICS_REFERENCE = (
    "quantfoundry_core.metrics.vendor_qs (Apache-2.0, QuantStats 0.0.77-derived)"
)

# ``display=False``, ``compounded=False``, single strategy, benchmark=None
BASIC_ROWS_STRATEGY_ONLY: tuple[str, ...] = (
    "Start Period",
    "End Period",
    "Risk-Free Rate",
    "Time in Market",
    "Total Return",
    "CAGR﹪",
    "Sharpe",
    "Prob. Sharpe Ratio",
    "Sortino",
    "Sortino/√2",
    "Omega",
    "Max Drawdown",
    "Max DD Date",
    "Max DD Period Start",
    "Max DD Period End",
    "Longest DD Days",
    "Gain/Pain Ratio",
    "Gain/Pain (1M)",
    "Payoff Ratio",
    "Profit Factor",
    "Common Sense Ratio",
    "CPC Index",
    "Tail Ratio",
    "Outlier Win Ratio",
    "Outlier Loss Ratio",
    "MTD",
    "3M",
    "6M",
    "YTD",
    "1Y",
    "3Y (ann.)",
    "5Y (ann.)",
    "10Y (ann.)",
    "All-time (ann.)",
    "Avg. Drawdown",
    "Avg. Drawdown Days",
    "Recovery Factor",
    "Ulcer Index",
    "Serenity Index",
)

EXTRA_FULL_ROWS: tuple[str, ...] = (
    "Smart Sharpe",
    "Smart Sortino",
    "Smart Sortino/√2",
    "Volatility (ann.)",
    "R^2",
    "Information Ratio",
    "Calmar",
    "Skew",
    "Kurtosis",
    "Expected Daily",
    "Expected Monthly",
    "Expected Yearly",
    "Kelly Criterion",
    "Risk of Ruin",
    "Daily Value-at-Risk",
    "Expected Shortfall (cVaR)",
    "Max Consecutive Wins",
    "Max Consecutive Losses",
    "Best Day",
    "Worst Day",
    "Best Month",
    "Worst Month",
    "Best Year",
    "Worst Year",
    "Avg. Up Month",
    "Avg. Down Month",
    "Win Days",
    "Win Month",
    "Win Quarter",
    "Win Year",
    "Beta",
    "Alpha",
    "Correlation",
    "Treynor Ratio",
)


def full_row_set_with_benchmark() -> frozenset[str]:
    return frozenset(BASIC_ROWS_STRATEGY_ONLY) | frozenset(EXTRA_FULL_ROWS)
