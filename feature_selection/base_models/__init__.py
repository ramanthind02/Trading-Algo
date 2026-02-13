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
- RuleBasedBinningModel: No binning; pass-through for rule-based bias nodes (outputs feature as-is)

Architecture:
- BaseModel: Orchestrates bias nodes (feature extraction) and owns a BinningModel (binning logic)
- BinningModelBase: Abstract interface for binning strategies (fit/predict pattern)
- QuantileBinningModel/DecisionTreeBinningModel/TwoBinBinningModel: Concrete binning implementations

Author: Trading Research Team
Date: 2025-01-07
"""

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.base_models.quantile_binning import QuantileBinningModel

from feature_selection.base_models.rule_based_binning import RuleBasedBinningModel
from feature_selection.base_models.feature_base_model import BaseModel

__all__ = [
    'BaseModel',  # New BaseModel that owns bias nodes and binning models
    'BinningModelBase',  # Abstract base for binning models
    'QuantileBinningModel',
    'RuleBasedBinningModel',
]

__version__ = '1.0.0'
