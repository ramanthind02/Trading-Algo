"""
Centralized Graphing and Visualization Module

This module provides a single source of truth for all plotting and visualization
functions used across the feature selection framework.

Author: Trading Research Team
Date: 2025-10-24
"""

from metrics.plotting.graphing.quantstats_reports import (
    generate_tearsheet,
    compute_baseline_results,
)

from metrics.plotting.graphing.portfolio_analytics import (
    PortfolioAnalytics,
    quick_metrics,
    quick_snapshot,
)

__all__ = [
    'generate_tearsheet',
    'compute_baseline_results',
    'PortfolioAnalytics',
    'quick_metrics',
    'quick_snapshot',
]
