"""
Objective Metric Package

This package provides objective metrics for evaluating risk-adjusted performance
in feature selection and trading strategy evaluation.

Available Metrics:
- ObjectiveMetric: Abstract base class
- SharpeRatio: Classic Sharpe ratio (penalizes all volatility)
- SortinoRatio: Sortino ratio (penalizes only downside volatility)

DEPRECATED: This module is deprecated. Use metrics.performance instead.

Author: Trading Research Team
Date: 2025-10-23
"""

import warnings

# Import from new location
from metrics.performance import ObjectiveMetric, SharpeRatio, SortinoRatio

# Issue deprecation warning
warnings.warn(
    "utils.objective_metric is deprecated and will be removed in a future version. "
    "Use metrics.performance instead. "
    "Example: from metrics.performance import SharpeRatio, SortinoRatio",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'ObjectiveMetric',
    'SharpeRatio',
    'SortinoRatio',
]

__version__ = '1.0.0'
