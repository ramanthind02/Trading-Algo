"""
Base Models Package

This package contains sklearn-style models for walk-forward analysis and feature selection.
All models follow the sklearn estimator API with fit(), predict(), get_params(), and set_params() methods.

Available Models:
- BaseModel: Abstract base class defining the interface
- QuantileBinningModel: Quantile-based binning (unsupervised)
- DecisionTreeBinningModel: Decision tree-based binning (supervised)

Author: Trading Research Team
Date: 2025-10-23
"""

from feature_selection.base_models.base_model import BaseModel
from feature_selection.base_models.quantile_binning import QuantileBinningModel
from feature_selection.base_models.tree_binning import DecisionTreeBinningModel

__all__ = [
    'BaseModel',
    'QuantileBinningModel',
    'DecisionTreeBinningModel',
]

__version__ = '1.0.0'
