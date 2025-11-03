"""
Centralized Graphing and Visualization Module

This module provides a single source of truth for all plotting and visualization
functions used across the feature selection framework.

Author: Trading Research Team
Date: 2025-10-24
"""

from feature_selection.graphing.quantstats_reports import (
    generate_tearsheet,
    compute_baseline_results
)

__all__ = [
    'generate_tearsheet',
    'compute_baseline_results'
]
