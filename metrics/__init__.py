"""
Centralized Trading Metrics Library

Compatibility shell: performance scalars live in
``feature_selection.validation.objective_metrics`` (backed by
``quantfoundry_core.metrics``). Risk and equity helpers remain here for
prop-firm simulation and drawdown utilities.
"""

from features.validation.objective_metrics import metric_sharpe, metric_sortino

from metrics.equity import cumulative_returns, equity_peak
from metrics.risk import drawdown_series, max_drawdown

__all__ = [
    "cumulative_returns",
    "drawdown_series",
    "equity_peak",
    "max_drawdown",
    "metric_sharpe",
    "metric_sortino",
]

__version__ = "1.0.0"
