"""
Centralized Trading Metrics Library

This package provides a centralized location for all trading-related metrics including
performance metrics (Sharpe, Sortino, Calmar), risk metrics (drawdown, VaR), and
equity curve calculations.

Author: Trading Research Team
Date: 2025-01-XX
"""

from metrics.performance import (
    ObjectiveMetric,  # Base class (kept for backward compat)
    SharpeRatio,
    SortinoRatio,
)

from metrics.risk import (
    max_drawdown,
    drawdown_series,
)

from metrics.equity import (
    cumulative_returns,
    equity_curve,
    equity_peak,
    equity_tracking,
)

__all__ = [
    # Performance
    'ObjectiveMetric', 'SharpeRatio', 'SortinoRatio',
    # Risk
    'max_drawdown', 'drawdown_series',
    # Equity
    'cumulative_returns', 'equity_curve', 'equity_peak', 'equity_tracking',
]

__version__ = '1.0.0'

