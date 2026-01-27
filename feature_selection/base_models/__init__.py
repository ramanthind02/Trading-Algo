"""
Base Models Package

This package contains models for walk-forward analysis and feature selection.

Available Classes:
- BaseModel: Complete feature extraction and binning model (owns bias nodes and binning model)
- BinningModelBase: Abstract base class for binning strategies
- QuantileBinningModel: Quantile-based binning (unsupervised, equal-frequency)
- UniformBinningModel: Uniform binning (unsupervised, equal-width)
- DecisionTreeBinningModel: Decision tree-based binning (supervised)
- TwoBinBinningModel: Two-bin binning based on positive/negative values (for momentum models)

Architecture:
- BaseModel: Orchestrates bias nodes (feature extraction) and owns a BinningModel (binning logic)
- BinningModelBase: Abstract interface for binning strategies (fit/predict pattern)
- QuantileBinningModel/DecisionTreeBinningModel/TwoBinBinningModel: Concrete binning implementations

Author: Trading Research Team
Date: 2025-01-07
"""

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.base_models.quantile_binning import QuantileBinningModel
from feature_selection.base_models.uniform_binning import UniformBinningModel
from feature_selection.base_models.tree_binning import DecisionTreeBinningModel
from feature_selection.base_models.twobin_binning import TwoBinBinningModel
from feature_selection.base_models.feature_base_model import BaseModel

__all__ = [
    'BaseModel',  # New BaseModel that owns bias nodes and binning models
    'BinningModelBase',  # Abstract base for binning models
    'QuantileBinningModel',
    'UniformBinningModel',
    'DecisionTreeBinningModel',
    'TwoBinBinningModel',
]

__version__ = '1.0.0'
