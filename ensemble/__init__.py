"""
Diversified Ensemble for Trading Strategy Combination

This module provides a diversified ensemble method specifically designed for 
combining multiple trading strategies using intra-feature correlations and 
instrument-level risk management.

Author: Trading Research Team
Date: 2025-11-23
"""

from .diversified_ensemble import DiversifiedEnsemble
from .portfolio import (
    GlobalPortfolio,
    Portfolio,
    PortfolioWorld,
    TFPortfolio,
    load_global_portfolio_snapshot,
    materialize_global_portfolio_predictions,
    prune_inactive_base_model_materializations,
)
from .portfolio_manager import PortfolioManager
from .weight_layer import (
    BaseWeightLayer,
    ClusteredWeightLayer,
    WeightLayer,
)

__all__ = [
    'DiversifiedEnsemble',
    'GlobalPortfolio',
    'Portfolio',
    'PortfolioWorld',
    'TFPortfolio',
    'load_global_portfolio_snapshot',
    'materialize_global_portfolio_predictions',
    'PortfolioManager',
    'prune_inactive_base_model_materializations',
    'WeightLayer',
    'BaseWeightLayer',
    'ClusteredWeightLayer',
]
