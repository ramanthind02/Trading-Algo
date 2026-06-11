"""
Metrics helpers: drawdown and equity-curve utilities.

Performance scalars (Sharpe, Sortino, etc.) live in
``features.validation.objective_metrics`` (backed by
``quantfoundry_core.metrics``). Only the small risk/equity helpers used by
prop-firm simulation and drawdown analysis remain here.
"""

from analysis.metrics.drawdown import drawdown_series, max_drawdown
from analysis.metrics.equity import cumulative_returns, equity_peak

__all__ = [
    "cumulative_returns",
    "drawdown_series",
    "equity_peak",
    "max_drawdown",
]
