"""
Objective Metric Package

This package provides objective metrics for evaluating risk-adjusted performance
in feature selection and trading strategy evaluation.

Available Metrics:
- ObjectiveMetric: Abstract base class
- SharpeRatio: Classic Sharpe ratio (penalizes all volatility)
- SortinoRatio: Sortino ratio (penalizes only downside volatility)

Author: Trading Research Team
Date: 2025-10-23
"""

from utils.objective_metric.base_metric import ObjectiveMetric
from utils.objective_metric.sharpe import SharpeRatio
from utils.objective_metric.sortino import SortinoRatio

__all__ = [
    'ObjectiveMetric',
    'SharpeRatio',
    'SortinoRatio',
]

__version__ = '1.0.0'
