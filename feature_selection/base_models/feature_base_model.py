"""
Feature Base Model - Owns Bias Nodes and Binning Models

This module provides the new BaseModel class that owns bias nodes and binning models
via composition. This separates feature extraction (bias nodes) from binning logic.

Author: Trading Research Team
Date: 2025-01-07
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from utils.enums import Ticker, TimeFrame
from utils.models import Candle
from utils import helpers
from feature_selection.base_models.base_model import BinningModelBase


@dataclass(frozen=True)
class BiasNodeSpec:
    """Immutable specification for bias node creation."""
    module_name: str
    timeframes: List[TimeFrame]
    params: Dict[str, Any]


class BaseModel:
    """
    Base model that owns bias nodes and a binning model.
    
    This class orchestrates the complete feature extraction and binning pipeline:
    1. Instantiates and manages bias nodes from bias_node_spec
    2. Streams candles to bias nodes
    3. Extracts features from bias node outputs
    4. Delegates binning to owned BinningModel
    5. Generates predictions
    
    Responsibilities:
    - Feature extraction: bias node management
    - Orchestration: coordinates bias nodes and binning model
    - Prediction: generates trading signals
    
    The binning model handles all binning logic (thresholds, bin selection, etc.).
    
    Parameters
    ----------
    bias_node_spec : Dict[str, Any]
        Specification for bias nodes: {
            'module_name': str,
            'timeframes': [TimeFrame],
            'params': dict
        }
    binning_model : BinningModelBase
        Binning model instance (QuantileBinningModel or DecisionTreeBinningModel)
        
    Returns
    -------
    self
        BaseModel instance
    ticker : Ticker
        Ticker symbol for this model
    """
    
    def __init__(
        self,
        bias_node_spec: Dict[str, Any],
        binning_model: BinningModelBase,
        ticker: Ticker
    ):
        self.bias_node_spec = bias_node_spec
        self.binning_model = binning_model
        self.ticker = ticker
        
        # Feature column name (set after first feature extraction)
        self.feature_column: Optional[str] = None
        
        # Create bias nodes internally
        self.bias_nodes: Dict[TimeFrame, Any] = {}
        for tf in bias_node_spec['timeframes']:
            bias_node = helpers.create_bias_node(
                bias_node_spec['module_name'],
                ticker,
                tf,
                bias_node_spec['params']
            )
            self.bias_nodes[tf] = bias_node
        
        # Track feature values as candles are added
        # Maps datetime -> feature value
        self._feature_values: Dict[Any, float] = {}
        self._feature_datetimes: List[Any] = []
    
    def add_candle(self, candle: Candle, tf: TimeFrame) -> None:
        """
        Stream candle to appropriate bias node and track feature value.
        
        Parameters
        ----------
        candle : Candle
            Candle to process
        tf : TimeFrame
            Timeframe for this candle
        """
        if tf in self.bias_nodes:
            # Get feature value from bias node (returns list, take first element)
            result = self.bias_nodes[tf].add_candle(candle)
            if result and len(result) > 0:
                # Store feature value indexed by candle datetime
                self._feature_values[candle.datetime] = result[0]
                if candle.datetime not in self._feature_datetimes:
                    self._feature_datetimes.append(candle.datetime)
    
    def get_feature(self) -> pd.Series:
        """
        Extract feature from tracked bias node outputs.
        
        Returns feature values in the order candles were added.
        The feature column name is standardized using build_feature_column_name().
        
        Returns
        -------
        pd.Series
            Feature values with standardized column name, indexed by datetime
        """
        # Get the primary timeframe (first one in the spec)
        primary_tf = self.bias_node_spec['timeframes'][0]
        primary_node = self.bias_nodes[primary_tf]
        
        # Extract feature values in order of candle addition
        feature_values = [self._feature_values.get(dt, np.nan) for dt in self._feature_datetimes]
        
        # Get standardized column name
        output_feature = 'signal'  # default
        if hasattr(primary_node, 'output_features') and primary_node.output_features:
            output_feature = primary_node.output_features[0]
        
        column_name = helpers.build_feature_column_name(
            module=self.bias_node_spec['module_name'],
            feature=output_feature,
            tf=primary_tf,
            params=self.bias_node_spec['params']
        )
        
        # Create series with standardized name, indexed by datetime
        feature_series = pd.Series(
            feature_values,
            index=pd.DatetimeIndex(self._feature_datetimes),
            name=column_name
        )
        self.feature_column = column_name
        
        return feature_series
    
    def fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series
    ) -> 'FeatureBaseModel':
        """
        Fit model: update bias nodes, extract features, fit binning model.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with candles. Must have columns: datetime, open, high, low, close, volume, ticker, timeframe
        target_data : pd.Series
            Target values (returns) aligned with candles
            
        Returns
        -------
        self
            Fitted model
        """
        # Stream candles to bias nodes
        for _, row in candles_df.iterrows():
            candle = Candle(
                datetime=row['datetime'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0)),
                ticker=self.ticker,
                tf=row['timeframe']
            )
            self.add_candle(candle, row['timeframe'])
        
        # Extract feature
        feature_data = self.get_feature()
        
        # Align feature and target data
        # Both should have datetime index
        aligned_target = target_data.reindex(feature_data.index)
        
        # Fit binning model
        self.binning_model.fit(feature_data, aligned_target)
        
        return self
    
    def predict(
        self,
        candles_df: pd.DataFrame,
        strategy: str = 'long'
    ) -> pd.Series:
        """
        Predict: update bias nodes, extract features, predict with binning model.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with candles. Must have columns: datetime, open, high, low, close, volume, ticker, timeframe
        strategy : str, default='long'
            Strategy to use: 'long' or 'short'
            
        Returns
        -------
        pd.Series
            Predictions (binary signals or scaled positions) indexed by candle datetimes
        """
        # Store current feature count before adding new candles
        n_features_before = len(self._feature_datetimes)
        
        # Stream candles to bias nodes
        for _, row in candles_df.iterrows():
            candle = Candle(
                datetime=row['datetime'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0)),
                ticker=self.ticker,
                tf=row['timeframe']
            )
            self.add_candle(candle, row['timeframe'])
        
        # Extract feature (all features, but we'll slice to new ones)
        feature_data = self.get_feature()
        
        # Only predict on the new candles (slice from n_features_before onwards)
        new_feature_data = feature_data.iloc[n_features_before:]
        
        # Predict with binning model
        predictions = self.binning_model.predict(new_feature_data, strategy=strategy)
        
        return predictions
    
    def save_to_vault(
        self,
        ensemble_dir: str
    ) -> str:
        """
        Save base model to vault.
        
        This is the primary API for researchers to save validated features.
        Model ID is auto-generated based on binning model type and hyperparameters.
        The bias_node_spec is already stored in self.bias_node_spec.
        
        Parameters
        ----------
        ensemble_dir : str
            Path to ensemble directory in vault (e.g., 'vault/D/commodity_breakout_long')
            
        Returns
        -------
        str
            The auto-generated model_id
            
        Raises
        ------
        ValueError
            If feature_column not set, or if strategy doesn't match ensemble direction
        """
        # Import here to avoid circular dependency
        from ensemble.vault_manager import add_feature_to_ensemble
        
        if self.feature_column is None:
            raise ValueError(
                "feature_column not set. Call fit() first to set feature_column from bias node outputs."
            )
        
        model_id = add_feature_to_ensemble(
            ensemble_dir=ensemble_dir,
            feature_column=self.feature_column,
            bias_node_spec=self.bias_node_spec,
            base_model=self
        )
        
        return model_id
    
    def update_fitted_params_in_vault(
        self,
        ensemble_dir: str,
        model_id: str,
        train_start: str,
        train_end: str
    ) -> None:
        """
        Update fitted parameters in vault after fitting.
        
        Call this after fitting a model that was loaded from the vault.
        Model ID must be provided (should match the ID from initial save).
        Fitted params are extracted from the owned binning_model.
        
        Parameters
        ----------
        ensemble_dir : str
            Path to ensemble directory
        model_id : str
            Model ID to update (e.g., 'quantile_binning_3')
        train_start : str
            Training start date (YYYY-MM-DD)
        train_end : str
            Training end date (YYYY-MM-DD)
        """
        from ensemble.vault_manager import update_base_model_fitted_params
        
        if not self.binning_model.is_fitted_:
            raise ValueError("Binning model must be fitted before updating vault")
        
        if self.feature_column is None:
            raise ValueError("feature_column not set")
        
        # Extract fitted params from owned binning model
        fitted_params = {
            'thresholds': self.binning_model.thresholds_.tolist() if self.binning_model.thresholds_ is not None else None,
            'best_long_bin': self.binning_model.best_long_bin_,
            'best_short_bin': self.binning_model.best_short_bin_,
            'bin_stats': self.binning_model.bin_stats_
        }
        
        update_base_model_fitted_params(
            ensemble_dir=ensemble_dir,
            feature_column=self.feature_column,
            model_id=model_id,
            fitted_params=fitted_params,
            train_start=train_start,
            train_end=train_end
        )