"""
Diversified Ensemble for Trading Strategy Combination

This module provides a diversified ensemble method specifically designed for 
combining multiple trading strategies using intra-feature correlations and 
instrument-level risk management.

Author: Trading Research Team
Date: 2025-11-23
"""

from .diversified_ensemble import DiversifiedEnsemble
from .global_weight_layer import GlobalWeightLayer, GlobalWeightLayerConfig
from .portfolio import GlobalPortfolio, Portfolio, TFPortfolio
from .portfolio_manager import PortfolioManager
from .weight_layer import (
    BaseWeightLayer,
    InverseCorrelationWeighter,
    InverseCorrelationWeightLayer,
    WeightLayer,
)

__all__ = [
    'DiversifiedEnsemble',
    'GlobalPortfolio',
    'GlobalWeightLayer',
    'GlobalWeightLayerConfig',
    'Portfolio',
    'TFPortfolio',
    'PortfolioManager',
    'WeightLayer',
    'BaseWeightLayer',
    'InverseCorrelationWeightLayer',
    'InverseCorrelationWeighter',
]
