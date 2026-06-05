"""
Risk Metrics Package

This package provides risk-related metrics including drawdown calculations,
Value at Risk (VaR), and Conditional Value at Risk (CVaR).

Author: Trading Research Team
Date: 2025-01-XX
"""

from metrics.risk.drawdown import (
    max_drawdown,
    drawdown_series,
)

__all__ = [
    'max_drawdown',
    'drawdown_series',
]

