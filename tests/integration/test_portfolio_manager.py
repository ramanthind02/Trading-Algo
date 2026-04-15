"""``PortfolioManager`` was removed; multi-timeframe orchestration uses ``GlobalPortfolio``."""

from __future__ import annotations

import importlib.util


def test_portfolio_manager_module_removed() -> None:
    spec = importlib.util.find_spec("ensemble.portfolio_manager")
    assert spec is None, "Use ensemble.GlobalPortfolio instead of PortfolioManager"
