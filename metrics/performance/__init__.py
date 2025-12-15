"""
Performance Metrics Package

This package provides risk-adjusted performance metrics for evaluating trading strategies.

Available Metrics:
- ObjectiveMetric: Abstract base class
- SharpeRatio: Classic Sharpe ratio (penalizes all volatility)
- SortinoRatio: Sortino ratio (penalizes only downside volatility)

Author: Trading Research Team
Date: 2025-01-XX
"""

from metrics.performance.base import ObjectiveMetric
from metrics.performance.sharpe import SharpeRatio
from metrics.performance.sortino import SortinoRatio

__all__ = [
    'ObjectiveMetric',
    'SharpeRatio',
    'SortinoRatio',
]

