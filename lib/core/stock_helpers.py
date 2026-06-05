"""Back-compat shim: ``load_stock_data`` now lives in ``data_platform.loaders``.

Kept so existing ``from utils.core.stock_helpers import load_stock_data`` imports
work. New code should import from ``data_platform.loaders``.
"""
from __future__ import annotations

from data_platform.loaders import load_stock_data  # noqa: F401
from data_platform.providers.norgate.stocks import StockAdjustment  # noqa: F401

__all__ = ["load_stock_data", "StockAdjustment"]
