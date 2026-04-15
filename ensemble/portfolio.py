"""Ensemble portfolio package facade (stable import paths).

``TFPortfolio`` / ``GlobalPortfolio`` implementations live under
``ensemble.portfolio_impl`` (``tf_portfolio``, ``global_portfolio_impl``); this
module re-exports the public surface.
"""
from __future__ import annotations

from .portfolio_impl.global_portfolio_impl import (
    GlobalPortfolio,
    load_global_portfolio_snapshot,
    materialize_global_portfolio_predictions,
    prune_inactive_base_model_materializations,
)
from .portfolio_impl.portfolio_cache import PortfolioCacheQuery
from .portfolio_impl.tf_portfolio import (
    PortfolioWorld,
    TFPortfolio,
    _forecast_to_activity_signal,
)

Portfolio = TFPortfolio

__all__ = [
    "GlobalPortfolio",
    "Portfolio",
    "PortfolioCacheQuery",
    "PortfolioWorld",
    "TFPortfolio",
    "_forecast_to_activity_signal",
    "load_global_portfolio_snapshot",
    "materialize_global_portfolio_predictions",
    "prune_inactive_base_model_materializations",
]
