"""
Feature Base Model - Owns Bias Nodes and Binning Models

This module provides the new BaseModel class that owns bias nodes and binning models
via composition. This separates feature extraction (bias nodes) from binning logic.

Author: Trading Research Team
Date: 2025-01-07
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
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
        feature_config: Dict[str, Any],
        tickers: Union[Ticker, List[Ticker]],
        binning_model: Optional[BinningModelBase] = None
    ):
        """
        Initialize base model with feature configuration.
        
        Creates bias nodes internally based on feature_config['bias_node_spec'].
        
        Parameters
        ----------
        feature_config : Dict[str, Any]
            Feature configuration from control file, including:
            - bias_node_spec: {
                'module_name': str,
                'timeframes': [TimeFrame],
                'params': dict
            }
        tickers : Union[Ticker, List[Ticker]]
            Ticker symbol(s) for this base model. Can be a single ticker or list of tickers.
            When multiple tickers are provided, the model is trained in aggregate across all tickers.
        binning_model : BinningModelBase, optional
            Binning model instance. If None, will be created from feature_config.
        """
        self.feature_config = feature_config
        
        # Normalize tickers to list
        if isinstance(tickers, Ticker):
            self.tickers = [tickers]
            self.ticker = tickers  # Keep for backward compatibility
        else:
            self.tickers = tickers
            self.ticker = tickers[0] if tickers else None  # Keep first for backward compatibility
        
        # Extract bias_node_spec from feature_config
        bias_node_spec = feature_config.get('bias_node_spec')
        if bias_node_spec is None:
            # Backward compatibility: if feature_config is actually bias_node_spec
            if 'module_name' in feature_config and 'timeframes' in feature_config:
                bias_node_spec = feature_config
                self.feature_config = {'bias_node_spec': bias_node_spec}
            else:
                raise ValueError(
                    "feature_config must include 'bias_node_spec' or be a bias_node_spec dict"
                )
        
        self.bias_node_spec = bias_node_spec
        
        # Get or create binning model
        if binning_model is None:
            # Try to create from feature_config
            from feature_selection.base_models.quantile_binning import QuantileBinningModel
            from feature_selection.base_models.tree_binning import DecisionTreeBinningModel
            from feature_selection.base_models.twobin_binning import TwoBinBinningModel
            
            model_type = feature_config.get('model_type', 'QuantileBinningModel')
            constructor_params = feature_config.get('constructor_params', {})
            
            if model_type == 'QuantileBinningModel':
                self.binning_model = QuantileBinningModel(**constructor_params)
            elif model_type == 'DecisionTreeBinningModel':
                self.binning_model = DecisionTreeBinningModel(**constructor_params)
            elif model_type == 'TwoBinBinningModel':
                self.binning_model = TwoBinBinningModel(**constructor_params)
            else:
                # Default to QuantileBinningModel
                self.binning_model = QuantileBinningModel()
        else:
            self.binning_model = binning_model
        
        # Feature column name (set after first feature extraction)
        self.feature_column: Optional[str] = None
        
        # Create bias nodes internally - one per ticker and timeframe
        # Key: (ticker, timeframe) tuple
        self.bias_nodes: Dict[Tuple[Ticker, TimeFrame], Any] = {}
        
        # Extract single values from lists in params (BaseModel doesn't do grid expansion)
        # If params contain lists, extract first value (for compatibility with extract_features_for_bias_node)
        cleaned_params = {}
        for key, value in bias_node_spec['params'].items():
            if isinstance(value, list):
                if len(value) == 0:
                    raise ValueError(f"Parameter '{key}' has empty list. Provide at least one value.")
                elif len(value) > 1:
                    raise ValueError(
                        f"Parameter '{key}' has multiple values {value}. "
                        f"BaseModel creates one feature per parameter combination. "
                        f"For multiple combinations, create separate BaseModel instances or use extract_features_for_bias_node()."
                    )
                # Extract single value from list
                cleaned_params[key] = value[0]
            else:
                cleaned_params[key] = value
        
        for ticker in self.tickers:
            for tf in bias_node_spec['timeframes']:
                bias_node = helpers.create_bias_node(
                    bias_node_spec['module_name'],
                    ticker,
                    tf,
                    cleaned_params
                )
                self.bias_nodes[(ticker, tf)] = bias_node
        
        # Track feature values as candles are added
        # Maps datetime -> feature value
        self._feature_values: Dict[Any, float] = {}
        self._feature_datetimes: List[Any] = []
        
        # Caching for fit and predict operations
        # Cache key: (date_range_start, date_range_end) as tuple of dates
        self._fit_cache: Dict[Tuple[Any, Any], bool] = {}  # Maps date_range -> is_fitted flag
        self._predict_cache: Dict[Tuple[Any, ...], pd.Series] = {}  # Maps (date_range, strategy) -> predictions
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to binning_model for compatibility.
        
        This allows the ensemble to access binning_model attributes
        (is_fitted_, strategy, thresholds_, etc.) directly on BaseModel.
        """
        # List of attributes to delegate to binning_model
        delegated_attrs = {
            'is_fitted_', 'strategy', 'n_bins', 'thresholds_',
            'best_long_bin_', 'best_short_bin_', 'bin_stats_'
        }
        
        if name in delegated_attrs:
            return getattr(self.binning_model, name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def add_candle(self, candle: Candle, tf: TimeFrame, ticker: Optional[Ticker] = None) -> None:
        """
        Stream candle to appropriate bias node and track feature value.
        
        Parameters
        ----------
        candle : Candle
            Candle to process
        tf : TimeFrame
            Timeframe for this candle
        ticker : Optional[Ticker]
            Ticker for this candle. If None, extracts from candle.ticker.
        """
        # Extract ticker from candle if not provided
        if ticker is None:
            if hasattr(candle, 'ticker'):
                ticker = candle.ticker
            else:
                raise ValueError("ticker must be provided or available in candle.ticker")
        
        # Convert string ticker to enum if needed
        if isinstance(ticker, str):
            try:
                ticker = Ticker[ticker]
            except (KeyError, AttributeError):
                pass
        
        # Find the appropriate bias node for this ticker and timeframe
        key = (ticker, tf)
        if key in self.bias_nodes:
            # Get feature value from bias node (returns list, take first element)
            result = self.bias_nodes[key].add_candle(candle)
            if result and len(result) > 0:
                # Store feature value indexed by candle datetime
                self._feature_values[candle.datetime] = result[0]
                if candle.datetime not in self._feature_datetimes:
                    self._feature_datetimes.append(candle.datetime)
    
    def get_feature(self) -> pd.Series:
        """
        Extract feature from tracked bias node outputs.
        
        For multi-ticker models, aggregates feature values across tickers by base datetime
        (removing millisecond offsets) using mean aggregation.
        
        Returns feature values in the order candles were added.
        The feature column name is standardized using build_feature_column_name().
        
        Returns
        -------
        pd.Series
            Feature values with standardized column name, indexed by datetime
        """
        # Get the primary timeframe (first one in the spec) and first ticker
        primary_tf = self.bias_node_spec['timeframes'][0]
        primary_ticker = self.tickers[0]
        primary_node = self.bias_nodes[(primary_ticker, primary_tf)]
        
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
        
        # Handle multi-ticker aggregation
        if len(self.tickers) > 1:
            # For multi-ticker models, we need to preserve all samples to match EDA behavior
            # Instead of aggregating by base datetime (which loses samples), we keep all feature values
            # and use the original datetimes as index. This preserves the full sample count.
            feature_data = []
            for dt in self._feature_datetimes:
                if dt in self._feature_values:
                    feature_data.append({
                        'datetime': dt,  # Keep original datetime (with microseconds)
                        'value': self._feature_values[dt]
                    })
            
            if not feature_data:
                # No feature values extracted - return empty series
                return pd.Series(dtype=float, name=column_name)
            
            # Create DataFrame and sort by datetime, but DON'T aggregate
            # This preserves all samples from all tickers
            df = pd.DataFrame(feature_data)
            df = df.sort_values('datetime')
            
            feature_series = pd.Series(
                df['value'].values,
                index=pd.DatetimeIndex(df['datetime']),
                name=column_name
            )
        else:
            # Single ticker: extract feature values in order of candle addition
            feature_values = [self._feature_values.get(dt, np.nan) for dt in self._feature_datetimes]
            
            # Create series with standardized name, indexed by datetime
            feature_series = pd.Series(
                feature_values,
                index=pd.DatetimeIndex(self._feature_datetimes),
                name=column_name
            )
        
        self.feature_column = column_name
        
        return feature_series
    
    def _get_date_range_key(self, candles_df: pd.DataFrame) -> Tuple[Any, Any]:
        """
        Generate cache key from date range of candles DataFrame.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame with datetime column
            
        Returns
        -------
        Tuple
            (min_date, max_date) as tuple of date objects (not datetime)
        """
        if candles_df.empty or 'datetime' not in candles_df.columns:
            return (None, None)
        
        datetimes = pd.to_datetime(candles_df['datetime'])
        min_date = datetimes.min().date()
        max_date = datetimes.max().date()
        return (min_date, max_date)
    
    def fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series
    ) -> 'BaseModel':
        """
        Fit model: update bias nodes, extract features, fit binning model.
        
        Uses caching based on date_range to avoid refitting for the same date range.
        
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
        # Check cache first
        date_range_key = self._get_date_range_key(candles_df)
        if date_range_key in self._fit_cache and self.binning_model.is_fitted_:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"BaseModel.fit() cache HIT for date_range: {date_range_key}")
            self._cache_hits += 1
            return self
        
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"BaseModel.fit() cache MISS for date_range: {date_range_key}")
        self._cache_misses += 1
        
        # Extract unique tickers from candles_df
        unique_tickers = candles_df['ticker'].unique()
        
        # Convert string tickers to enum if needed
        normalized_tickers = []
        for ticker in unique_tickers:
            if isinstance(ticker, str):
                try:
                    normalized_tickers.append(Ticker[ticker])
                except (KeyError, AttributeError):
                    normalized_tickers.append(ticker)
            else:
                normalized_tickers.append(ticker)
        
        # Validate that all tickers in candles_df are in self.tickers
        for ticker in normalized_tickers:
            if ticker not in self.tickers:
                raise ValueError(
                    f"Ticker {ticker} in candles_df is not in model's tickers {self.tickers}. "
                    f"Model was initialized with tickers: {self.tickers}"
                )
        
        # Clear previous feature values to start fresh
        self._feature_values.clear()
        self._feature_datetimes.clear()
        
        # Stream candles to bias nodes for all tickers
        # IMPORTANT: Sort by datetime to ensure proper sequential processing
        # This is critical for features that require lookback windows (e.g., EWMAC, RSI)
        for ticker in normalized_tickers:
            ticker_candles = candles_df[candles_df['ticker'] == ticker].copy()
            
            # Sort by datetime to ensure sequential processing
            if 'datetime' in ticker_candles.columns:
                ticker_candles = ticker_candles.sort_values('datetime')
            
            # Extract features for this ticker
            for _, row in ticker_candles.iterrows():
                candle = Candle.from_row(row)
                self.add_candle(candle, row['timeframe'], ticker=ticker)
        
        # Extract aggregated features (all tickers)
        feature_data = self.get_feature()
        
        # Debug: Check if we have any feature values
        if len(feature_data) == 0:
            raise ValueError(
                f"No feature values extracted. This can happen if:\n"
                f"1. Bias nodes need more candles to warm up (e.g., EWMAC with spanSlow=256 needs ~256 candles)\n"
                f"2. All feature values are NaN\n"
                f"3. Candles are not being streamed correctly\n"
                f"Debug info:\n"
                f"  - Feature datetimes tracked: {len(self._feature_datetimes)}\n"
                f"  - Feature values stored: {len(self._feature_values)}\n"
                f"  - Tickers: {self.tickers}\n"
                f"  - Bias nodes: {list(self.bias_nodes.keys())}\n"
                f"  - Candles processed: {len(candles_df)}"
            )
        
        # Set feature_column from feature data
        if feature_data.name:
            self.feature_column = feature_data.name
        else:
            # If name is missing, rebuild it
            if self.feature_column is None:
                # Rebuild column name
                primary_tf = self.bias_node_spec['timeframes'][0]
                primary_ticker = self.tickers[0]
                primary_node = self.bias_nodes[(primary_ticker, primary_tf)]
                output_feature = 'signal'  # default
                if hasattr(primary_node, 'output_features') and primary_node.output_features:
                    output_feature = primary_node.output_features[0]
                column_name = helpers.build_feature_column_name(
                    module=self.bias_node_spec['module_name'],
                    feature=output_feature,
                    tf=primary_tf,
                    params=self.bias_node_spec['params']
                )
                self.feature_column = column_name
            # Ensure feature_data has the name
            feature_data.name = self.feature_column
        
        # Align feature and target data
        # Both should have datetime index
        # For multi-ticker models, both feature_data and target_data should have the same datetimes
        # (since returns are calculated from the same candles used for features)
        # Use reindex to align by exact datetime matching (preserves all samples)
        aligned_target = target_data.reindex(feature_data.index)
        
        # Check alignment quality
        n_aligned = aligned_target.notna().sum()
        n_features = len(feature_data)
        alignment_ratio = n_aligned / n_features if n_features > 0 else 0.0
        
        # If alignment resulted in all NaN or very poor alignment, try aligning by base datetime
        # This is a fallback for when datetimes don't match exactly
        if (aligned_target.isna().all() or alignment_ratio < 0.5) and len(target_data) > 0:
            # Use the same method as BaseModel.get_feature() for consistency
            # BaseModel.get_feature() uses: base_dt = dt.replace(microsecond=0)
            # We'll use the same approach here
            
            # Convert indices to base datetime (remove microseconds) to match get_feature() behavior
            def to_base_datetime(dt):
                """Convert datetime to base datetime (remove microseconds) - matches get_feature() behavior."""
                if isinstance(dt, pd.Timestamp):
                    return dt.replace(microsecond=0)
                else:
                    return pd.to_datetime(dt).replace(microsecond=0)
            
            # Create base datetime indices (matching get_feature() behavior)
            feature_base_index = pd.DatetimeIndex([to_base_datetime(dt) for dt in feature_data.index])
            target_base_index = pd.DatetimeIndex([to_base_datetime(dt) for dt in target_data.index])
            
            # Create mapping from base datetime to feature values
            feature_by_base = pd.Series(feature_data.values, index=feature_base_index)
            # Group by base datetime and take mean if multiple values per base datetime
            feature_by_base = feature_by_base.groupby(feature_by_base.index).mean()
            
            # Align target to base datetimes
            target_by_base = pd.Series(target_data.values, index=target_base_index)
            target_by_base = target_by_base.groupby(target_by_base.index).mean()  # Use mean to match feature aggregation
            
            # Find common base datetimes
            common_base_dt = feature_by_base.index.intersection(target_by_base.index)
            
            if len(common_base_dt) == 0:
                raise ValueError(
                    f"Cannot align feature and target data. "
                    f"Feature index range: {feature_data.index.min()} to {feature_data.index.max()}, "
                    f"Target index range: {target_data.index.min()} to {target_data.index.max()}. "
                    f"No overlapping base datetimes found."
                )
            
            # Reindex both to common base datetimes
            feature_data = feature_by_base.reindex(common_base_dt)
            aligned_target = target_by_base.reindex(common_base_dt)
            
            # Validate alignment after base datetime matching
            final_alignment_ratio = aligned_target.notna().sum() / len(feature_data) if len(feature_data) > 0 else 0.0
            if final_alignment_ratio < 0.8:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Poor alignment after base datetime matching: {final_alignment_ratio:.1%} aligned. "
                    f"Feature range: {feature_data.index.min()} to {feature_data.index.max()}, "
                    f"Target range: {target_by_base.index.min()} to {target_by_base.index.max()}, "
                    f"Common datetimes: {len(common_base_dt)}"
                )
        
        # Final validation: ensure we have enough aligned data
        final_n_aligned = aligned_target.notna().sum()
        if final_n_aligned < 20:
            raise ValueError(
                f"Insufficient aligned data: {final_n_aligned} samples aligned out of {len(feature_data)} features. "
                f"This suggests a datetime alignment issue between features and returns."
            )
        
        # Fit binning model
        self.binning_model.fit(feature_data, aligned_target)
        
        # Store in cache
        self._fit_cache[date_range_key] = True
        
        return self
    
    def predict(
        self,
        candles_df: pd.DataFrame,
        strategy: str = 'long'
    ) -> pd.Series:
        """
        Predict: update bias nodes, extract features, predict with binning model.
        
        Uses caching based on date_range and strategy to avoid recomputation.
        
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
            Returns predictions for ALL input candles, not just new ones.
        """
        # Check cache first
        date_range_key = self._get_date_range_key(candles_df)
        cache_key = (date_range_key, strategy)
        
        if cache_key in self._predict_cache:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"BaseModel.predict() cache HIT for date_range: {date_range_key}, strategy: {strategy}")
            self._cache_hits += 1
            # Return cached predictions, but reindex to match input candles datetimes
            cached_pred = self._predict_cache[cache_key]
            input_datetimes = pd.to_datetime(candles_df['datetime'])
            # Reindex cached predictions to input datetimes (forward fill for same dates)
            aligned_pred = cached_pred.reindex(input_datetimes, method='ffill')
            return aligned_pred.fillna(0)  # Fill any missing with 0
        
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"BaseModel.predict() cache MISS for date_range: {date_range_key}, strategy: {strategy}")
        self._cache_misses += 1
        
        # Note: We don't reset state here because base models may be called multiple times
        # with different tickers, and we want to preserve historical context for features
        # that require lookback windows (e.g., moving averages, RSI, etc.)
        
        # Extract unique tickers from candles_df
        unique_tickers = candles_df['ticker'].unique()
        
        # Convert string tickers to enum if needed
        normalized_tickers = []
        for ticker in unique_tickers:
            if isinstance(ticker, str):
                try:
                    normalized_tickers.append(Ticker[ticker])
                except (KeyError, AttributeError):
                    normalized_tickers.append(ticker)
            else:
                normalized_tickers.append(ticker)
        
        # Validate that all tickers in candles_df are in self.tickers
        for ticker in normalized_tickers:
            if ticker not in self.tickers:
                raise ValueError(
                    f"Ticker {ticker} in candles_df is not in model's tickers {self.tickers}. "
                    f"Model was initialized with tickers: {self.tickers}"
                )
        
        # Collect input datetimes
        all_input_datetimes = []
        for _, row in candles_df.iterrows():
            candle = Candle.from_row(row)
            all_input_datetimes.append(candle.datetime)
        
        # Extract features for all tickers
        for ticker in normalized_tickers:
            ticker_candles = candles_df[candles_df['ticker'] == ticker].copy()
            
            # Extract features for this ticker
            for _, row in ticker_candles.iterrows():
                candle = Candle.from_row(row)
                self.add_candle(candle, row['timeframe'], ticker=ticker)
        
        # Extract aggregated features (all tickers)
        feature_data = self.get_feature()
        
        # Get feature values for the input datetimes (in order)
        # This ensures we return predictions for all input candles, even if some were seen before
        feature_values = []
        feature_index = []
        for dt in all_input_datetimes:
            # Check if feature exists in feature_data (preferred) or in _feature_values (fallback)
            if dt in feature_data.index:
                feature_values.append(feature_data.loc[dt])
                feature_index.append(dt)
            elif dt in self._feature_values:
                # Feature value exists but not in feature_data index (edge case)
                feature_values.append(self._feature_values[dt])
                feature_index.append(dt)
            else:
                # No feature for this datetime - this shouldn't happen if bias nodes are working
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(
                    f"BaseModel.predict() no feature value for datetime {dt}. "
                    f"Bias node may not have processed this candle."
                )
                # Use NaN as placeholder - will be handled by binning model
                feature_values.append(np.nan)
                feature_index.append(dt)
        
        if not feature_index:
            # No features extracted at all - this is a problem, log it
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"BaseModel.predict() extracted no features for {len(all_input_datetimes)} candles. "
                f"Feature data length: {len(feature_data)}, "
                f"Feature datetimes: {len(self._feature_datetimes)}, "
                f"Input datetimes: {len(all_input_datetimes)}, "
                f"Base model tickers: {self.tickers}"
            )
            # Return empty Series with correct index
            return pd.Series(dtype=float, index=pd.DatetimeIndex(all_input_datetimes))
        
        # Create Series with feature values for input datetimes
        input_feature_data = pd.Series(feature_values, index=pd.DatetimeIndex(feature_index))
        
        # Drop NaN values before prediction (bias nodes should always return values)
        if input_feature_data.isna().any():
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"BaseModel.predict() has {input_feature_data.isna().sum()} NaN values out of {len(input_feature_data)}. "
                f"This indicates bias nodes are not processing all candles correctly."
            )
            # For now, fill NaN with 0 (or could drop them)
            input_feature_data = input_feature_data.fillna(0.0)
        
        # Predict with binning model
        predictions = self.binning_model.predict(input_feature_data, strategy=strategy)
        
        # Store in cache (indexed by input datetimes for exact matching)
        self._predict_cache[cache_key] = predictions.copy()
        
        return predictions
    
    def save_to_vault(
        self,
        ensemble_dir: Optional[str] = None,
        tickers: Optional[List['Ticker']] = None
    ) -> str:
        """
        Save base model to vault.
        
        This is the primary API for researchers to save validated features.
        Model ID is auto-generated based on binning model type and hyperparameters.
        The bias_node_spec is already stored in self.bias_node_spec.
        
        Parameters
        ----------
        ensemble_dir : str, optional
            Path to ensemble directory in vault (e.g., 'vault/D/commodity_breakout_long').
            If None, uses default ensemble directory set via set_default_ensemble_dir().
        tickers : List[Ticker], optional
            List of tickers this ensemble was trained on. If None, infers from self.ticker.
            This is stored in the feature spec so base models can be created separately for each ticker.
            
        Returns
        -------
        str
            The auto-generated model_id
            
        Raises
        ------
        ValueError
            If feature_column not set, or if strategy doesn't match ensemble direction,
            or if ensemble_dir is None and no default is set
        """
        # Import here to avoid circular dependency
        from ensemble.vault_manager import add_feature_to_ensemble
        from utils.enums import Ticker
        
        # Generate feature_column from bias_node_spec if not set (for unfitted models)
        if self.feature_column is None:
            # Get feature column name from bias node spec
            primary_tf = self.bias_node_spec['timeframes'][0]
            
            # Try to get output feature name from bias node if available
            # Otherwise use 'signal' as default (most common output feature)
            output_feature = 'signal'  # default
            if self.tickers:
                primary_ticker = self.tickers[0]
                primary_node = self.bias_nodes.get((primary_ticker, primary_tf))
                if primary_node and hasattr(primary_node, 'output_features') and primary_node.output_features:
                    output_feature = primary_node.output_features[0]
            
            # Build feature column name from spec
            self.feature_column = helpers.build_feature_column_name(
                module=self.bias_node_spec['module_name'],
                feature=output_feature,
                tf=primary_tf,
                params=self.bias_node_spec['params']
            )
        
        model_id = add_feature_to_ensemble(
            feature_column=self.feature_column,
            bias_node_spec=self.bias_node_spec,
            base_model=self,
            ensemble_dir=ensemble_dir,
            tickers=tickers
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
    
    def clear_cache(self) -> None:
        """Clear all cached fit and predict results."""
        self._fit_cache.clear()
        self._predict_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with cache hits, misses, hit rate, and cache sizes
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0.0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate': hit_rate,
            'fit_cache_size': len(self._fit_cache),
            'predict_cache_size': len(self._predict_cache)
        }