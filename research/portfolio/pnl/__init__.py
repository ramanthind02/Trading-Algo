"""Dual-lane P&L engine package (WP-3, NautilusTrader migration).

Provides a ``PnLEngine`` protocol plus a default ``VectorizedPnLEngine`` that
delegates verbatim to the frozen vectorized P&L baseline. The ``nautilus`` lane
is not yet implemented (blocked on WP-2 Unit-2 data-precision fork).
"""
from __future__ import annotations

from research.portfolio.pnl.pnl_engine import (
    PnLEngine,
    VectorizedPnLEngine,
    make_pnl_engine,
)

__all__ = [
    "PnLEngine",
    "VectorizedPnLEngine",
    "make_pnl_engine",
]
