"""
Equity Curve Calculations

This package provides functions for computing equity curves, cumulative returns,
and equity tracking utilities.

Author: Trading Research Team
Date: 2025-01-XX
"""

from metrics.equity.cumulative import cumulative_returns, equity_curve
from metrics.equity.tracking import equity_peak

__all__ = [
    "cumulative_returns",
    "equity_curve",
    "equity_peak",
]

