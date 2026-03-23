"""
Feature Base Model - Owns Bias Nodes and Binning Models

This module provides the new BaseModel class that owns bias nodes and binning models
via composition. This separates feature extraction (bias nodes) from binning logic.

Author: Trading Research Team
Date: 2025-01-07
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

from utils.core.enums import Direction, DirectionInput, Ticker, TimeFrame
from utils.core.models import Candle
from utils.core import helpers
from feature_selection.base_models.base_model import BinningModelBase

logger = logging.getLogger(__name__)


def _normalize_ticker_key(ticker_val: object) -> str:
    """Normalize enum/string ticker representations to stable symbol keys."""
    if hasattr(ticker_val, "name"):
        return str(getattr(ticker_val, "name"))
    text = str(ticker_val)
    if text.startswith("Ticker."):
        return text.split(".", 1)[1]
    return text


def _align_feature_and_target(
    feature_data: pd.Series,
    target_data: pd.Series,
    *,
    rule_based: bool = False,
    min_aligned: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Align feature and target by datetime index; fallback to base-datetime matching if needed.

    Returns (feature_data, aligned_target) both on a common index with at least min_aligned
    non-NaN target values. Raises ValueError if alignment fails or insufficient data.
    """
    aligned_target = target_data.reindex(feature_data.index)
    n_aligned = aligned_target.notna().sum()
    n_features = len(feature_data)
    alignment_ratio = n_aligned / n_features if n_features > 0 else 0.0

    all_nan = aligned_target.isna().all()
    if hasattr(all_nan, "all"):
        all_nan = bool(all_nan)
    if (all_nan or alignment_ratio < 0.5) and len(target_data) > 0:
        def to_base_datetime(dt: datetime | pd.Timestamp) -> pd.Timestamp:
            if isinstance(dt, pd.Timestamp):
                return dt.replace(microsecond=0)
            return pd.to_datetime(dt).replace(microsecond=0)

        feature_base_index = pd.DatetimeIndex([to_base_datetime(dt) for dt in feature_data.index])
        target_base_index = pd.DatetimeIndex([to_base_datetime(dt) for dt in target_data.index])
        feature_by_base = pd.Series(feature_data.values, index=feature_base_index)
        feature_by_base = feature_by_base.groupby(feature_by_base.index).mean()
        target_by_base = pd.Series(target_data.values, index=target_base_index)
        target_by_base = target_by_base.groupby(target_by_base.index).mean()
        common_base_dt = feature_by_base.index.intersection(target_by_base.index)

        if len(common_base_dt) == 0:
            raise ValueError(
                "Cannot align feature and target data. "
                f"Feature index range: {feature_data.index.min()} to {feature_data.index.max()}, "
                f"Target index range: {target_data.index.min()} to {target_data.index.max()}. "
                "No overlapping base datetimes found."
            )
        feature_data = feature_by_base.reindex(common_base_dt)
        if rule_based:
            feature_data = feature_data.round().clip(-1, 1).fillna(0).astype(int)
        aligned_target = target_by_base.reindex(common_base_dt)

        final_alignment_ratio = aligned_target.notna().sum() / len(feature_data) if len(feature_data) > 0 else 0.0
        if final_alignment_ratio < 0.8:
            logger.warning(
                "Poor alignment after base datetime matching: %s aligned. "
                "Feature range: %s to %s, Target range: %s to %s, Common datetimes: %s",
                f"{final_alignment_ratio:.1%}",
                feature_data.index.min(),
                feature_data.index.max(),
                target_by_base.index.min(),
                target_by_base.index.max(),
                len(common_base_dt),
            )

    final_n_aligned = aligned_target.notna().sum()
    if final_n_aligned < min_aligned:
        raise ValueError(
            f"Insufficient aligned data: {final_n_aligned} samples aligned out of {len(feature_data)} features. "
            "This suggests a datetime alignment issue between features and returns."
        )
    return (feature_data, aligned_target)


@dataclass(frozen=True)
class BiasNodeSpec:
    """Immutable specification for bias node creation.

    Parameters
    ----------
    module_name : str
        Canonical module key (e.g. ``'rsi'``, ``'ewmac'``).
    timeframes : list[TimeFrame]
        Timeframes to create nodes for.
    params : dict[str, object]
        Keyword arguments forwarded to the bias node constructor.
    filters : tuple
        Optional chain of :class:`filters.FilterSpec` instances.
        Default is an empty tuple (no filtering).
    """
    module_name: str
    timeframes: list[TimeFrame]
    params: dict[str, object]
    filters: tuple = ()


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
    bias_node_spec : dict[str, object]
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
        feature_config: dict[str, object],
        tickers: Ticker | list[Ticker],
        binning_model: Optional[BinningModelBase] = None,
        use_cache: bool = True
    ):
        """
        Initialize base model with feature configuration.
        
        Creates bias nodes internally based on feature_config['bias_node_spec'].
        
        Parameters
        ----------
        feature_config : dict[str, object]
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
        use_cache : bool, default=True
            If True, uses vectorized cached data when available.
            If False, uses streaming candle-by-candle processing.
        """
        self.feature_config = feature_config
        self.use_cache = use_cache
        
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
            from feature_selection.base_models.continuous_binning import ContinuousBinningModel
        
            from feature_selection.base_models.rule_based import RuleBasedModel

            model_type = feature_config.get('model_type', 'QuantileBinningModel')
            constructor_params = feature_config.get('constructor_params', {})

            if model_type in ('QuantileBinningModel', 'ContinuousBinningModel'):
                self.binning_model = ContinuousBinningModel(**constructor_params)
            elif model_type == 'RuleBasedBinningModel':
                self.binning_model = RuleBasedModel(**constructor_params)
            else:
                raise ValueError(
                    f"Unknown model_type: {model_type!r}. "
                    "Expected 'QuantileBinningModel', 'ContinuousBinningModel', or 'RuleBasedBinningModel'."
                )
        else:
            self.binning_model = binning_model
        
        # Feature column name (set after first feature extraction)
        self.feature_column: Optional[str] = None
        
        # Create bias nodes internally - one per ticker and timeframe
        # Key: (ticker, timeframe) tuple
        self.bias_nodes: dict[tuple[Ticker, TimeFrame], object] = {}
        
        # Extract single values from lists in params (BaseModel doesn't do grid expansion)
        # If params contain lists, extract first value (for compatibility with extract_features_for_bias_node)
        # Exception: scalar-list params (e.g. cross_tickers) must NOT be unwrapped.
        from utils.data.cross_ticker_store import SCALAR_LIST_PARAM_KEYS

        cleaned_params = {}
        for key, value in bias_node_spec['params'].items():
            if key in SCALAR_LIST_PARAM_KEYS:
                # Inherently list-valued param; never unwrap
                cleaned_params[key] = value
            elif isinstance(value, list):
                if len(value) == 0:
                    raise ValueError(f"Parameter '{key}' has empty list. Provide at least one value.")
                elif len(value) > 1:
                    raise ValueError(
                        f"Parameter '{key}' has multiple values {value}. "
                        "BaseModel requires a single param combo."
                    )
                cleaned_params[key] = value[0]
            else:
                cleaned_params[key] = value
        
        for ticker in self.tickers:
            for tf in bias_node_spec['timeframes']:
                # TODO: pass filter_specs from bias_node_spec to apply signal filters
                filter_specs = list(bias_node_spec.get('filters', ()))
                bias_node = helpers.create_filtered_bias_node(
                    bias_node_spec['module_name'],
                    ticker,
                    tf,
                    cleaned_params,
                    filter_specs=filter_specs,
                )
                self.bias_nodes[(ticker, tf)] = bias_node

        # Pre-load cross-ticker data for pairs/spread nodes
        self._preload_cross_ticker_data(cleaned_params, bias_node_spec['timeframes'])
        
        # Track feature values as candles are added
        # Maps datetime -> feature value
        self._feature_values: dict[datetime, float] = {}
        self._feature_datetimes: list[datetime] = []
        
        # Multi-member container: list of (member_name, binning_model) tuples
        # Each member is an independent binning model that can be fitted on the same features
        self.members: list[tuple[str, BinningModelBase]] = []
        # Optional: member_name -> feature_column for per-member feature columns (DataFrame input)
        self._member_feature_columns: dict[str, str] = {}

    def add_member(
        self,
        name: str,
        binning_model: BinningModelBase,
        feature_column: Optional[str] = None,
    ) -> None:
        """
        Add a member model to this base model.

        Parameters
        ----------
        name : str
            Unique identifier for this member
        binning_model : BinningModelBase
            The binning model instance to add as a member
        feature_column : str, optional
            Feature column name for this member. When provided, emit_member_signals(feature_data=pd.DataFrame)
            will use feature_data[feature_column] for this member. If None, the member uses the primary
            feature (single Series or first column) when feature_data is provided.
        """
        self.members.append((name, binning_model))
        if feature_column is not None:
            self._member_feature_columns[name] = feature_column

    def get_member_feature_columns(self) -> list[str]:
        """
        Return the list of feature column names used by members (for ensemble required_columns).

        Returns the primary feature_column if set, plus each member's feature_column when
        provided via add_member(..., feature_column=...). Deduplicated and order-preserving.
        """
        columns: list[str] = []
        if self.feature_column:
            columns.append(self.feature_column)
        for _name, _bm in self.members:
            col = self._member_feature_columns.get(_name)
            if col and col not in columns:
                columns.append(col)
        return columns

    def __getattr__(self, name: str) -> object:
        """
        Delegate attribute access to binning_model for compatibility.
        
        This allows the ensemble to access binning_model attributes
        (is_fitted_, strategy, thresholds_, etc.) directly on BaseModel.
        """
        # List of attributes to delegate to binning_model
        delegated_attrs = {
            'is_fitted_', 'strategy', 'n_bins', 'thresholds_',
            'best_long_bin_', 'best_short_bin_', 'bin_stats_', 'bin_edges_',
            'active_bins_by_strategy_',
        }
        
        if name in delegated_attrs:
            return getattr(self.binning_model, name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @staticmethod
    def _preload_cross_ticker_data(params: dict, timeframes: list) -> None:
        """Pre-load cross-ticker data into the store for ``cross_tickers`` params."""
        from utils.data.cross_ticker_store import (
            CrossTickerDataStore,
            extract_cross_ticker_names,
        )

        cross_names = extract_cross_ticker_names(params)
        if not cross_names:
            return

        from utils.core.enums import Ticker as _Ticker, TimeFrame as _TF
        store = CrossTickerDataStore.get_instance()
        for normalized in cross_names:
            try:
                ct = _Ticker[normalized]
            except KeyError:
                continue
            for tf in timeframes:
                tf_enum = _TF[tf] if isinstance(tf, str) else tf
                if not store.is_loaded(ct, tf_enum):
                    store.load(ct, tf_enum)

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
        
        For multi-ticker models, aggregates feature values across tickers by bar datetime
        (primary key (datetime, ticker)) using mean aggregation.
        
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
    
    def fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> 'BaseModel':
        """
        Fit model: update bias nodes, extract features, fit binning model.

        Dispatches to vectorized_fit (cached) or stream_fit based on use_cache setting.

        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with candles. Must have columns: datetime, open, high, low, close, volume, ticker, timeframe
        target_data : pd.Series
            Target values (returns) aligned with candles
        start_date : datetime, optional
            Start date for vectorized fit (required if use_cache=True)
        end_date : datetime, optional
            End date for vectorized fit (required if use_cache=True)

        Returns
        -------
        self
            Fitted model
        """
        if self.use_cache:
            return self.vectorized_fit(candles_df, target_data, start_date, end_date)
        return self.stream_fit(candles_df, target_data)

    def stream_fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series
    ) -> 'BaseModel':
        """
        Fit model using streaming candle-by-candle processing.

        This is the original fit implementation that processes candles one at a time.

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
        # Convert target_data to Series if it's a DataFrame
        if isinstance(target_data, pd.DataFrame):
            # If DataFrame has 'datetime' column, use it as index
            if 'datetime' in target_data.columns:
                target_data = target_data.set_index('datetime')
            # Get the first non-datetime column as the target
            target_cols = [c for c in target_data.columns if c != 'datetime']
            if target_cols:
                target_data = target_data[target_cols[0]]
            else:
                target_data = target_data.iloc[:, 0]

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
            # Handle both enum and string ticker values in dataframe
            ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
            ticker_candles = candles_df[
                (candles_df['ticker'] == ticker) | (candles_df['ticker'] == ticker_str)
            ].copy()
            
            # Sort by datetime to ensure sequential processing
            if 'datetime' in ticker_candles.columns:
                ticker_candles = ticker_candles.sort_values('datetime')
            
            # Extract features for this ticker (itertuples is much faster than iterrows)
            for row in ticker_candles.itertuples(index=False):
                candle = Candle.from_row_fast(row)
                self.add_candle(candle, candle.tf, ticker=ticker)
        
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
        
        # Align feature and target data (shared logic with vectorized_fit)
        is_rule_based = getattr(self.binning_model, "model_type", None) == "rule_based"
        feature_data, aligned_target = _align_feature_and_target(
            feature_data, target_data, rule_based=is_rule_based
        )
        self.binning_model.fit(feature_data, aligned_target)
        return self

    def vectorized_fit(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> 'BaseModel':
        """
        Fit model using vectorized cached data.

        Uses get_cached_values() from bias nodes for bulk feature extraction,
        avoiding the candle-by-candle processing loop.

        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with candles. Used for extracting date range if not provided.
        target_data : pd.Series
            Target values (returns) aligned with candles
        start_date : datetime, optional
            Start date for cached data. If None, inferred from candles_df.
        end_date : datetime, optional
            End date for cached data. If None, inferred from candles_df.

        Returns
        -------
        self
            Fitted model

        Raises
        ------
        CacheMissError
            If cache is not available for any bias node
        """
        from utils.cache.bias_node_cache import CacheMissError

        # Convert target_data to Series if it's a DataFrame
        if isinstance(target_data, pd.DataFrame):
            # If DataFrame has 'datetime' column, use it as index
            if 'datetime' in target_data.columns:
                target_data = target_data.set_index('datetime')
            # Get the first non-datetime column as the target
            target_cols = [c for c in target_data.columns if c != 'datetime']
            if target_cols:
                target_data = target_data[target_cols[0]]
            else:
                target_data = target_data.iloc[:, 0]

        # Infer date range from candles_df if not provided
        if start_date is None:
            start_date = pd.to_datetime(candles_df['datetime']).min()
        if end_date is None:
            end_date = pd.to_datetime(candles_df['datetime']).max()

        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Get the primary timeframe and build feature column name
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

        # Collect cached features from all bias nodes
        all_features = []
        for ticker in self.tickers:
            for tf in self.bias_node_spec['timeframes']:
                bias_node = self.bias_nodes[(ticker, tf)]
                try:
                    cached_values = bias_node.get_cached_values(
                        start=start_date,
                        end=end_date,
                        require_cache=True
                    )
                    if cached_values is not None and len(cached_values) > 0:
                        all_features.append(cached_values)
                except CacheMissError as exc:
                    logger.error(
                        f"Cache miss for {bias_node.module_name} ({ticker}, {tf}). "
                        f"Vectorized fit requires complete cache coverage.",
                        exc_info=True
                    )
                    raise exc

        if not all_features:
            raise ValueError(
                f"No cached features available for any bias node. "
                f"Run cache population first or set use_cache=False."
            )

        # Combine features from all tickers/timeframes
        if len(all_features) == 1:
            feature_data = all_features[0]
        else:
            # Concatenate and aggregate by datetime (mean for multi-ticker)
            combined = pd.concat(all_features)
            feature_data = combined.groupby(combined.index).mean()

        feature_data.name = column_name
        # Rule-based expects discrete {-1, 0, 1}; mean() across tickers produces fractions
        if getattr(self.binning_model, "model_type", None) == "rule_based":
            feature_data = feature_data.round().clip(-1, 1).fillna(0).astype(int)

        # Ensure unique index for reindex (multi-ticker target_data can have duplicate datetimes)
        if target_data.index.duplicated().any():
            target_data = target_data.groupby(level=0).mean()
        if feature_data.index.duplicated().any():
            feature_data = feature_data.groupby(feature_data.index).mean()

        is_rule_based = getattr(self.binning_model, "model_type", None) == "rule_based"
        feature_data, aligned_target = _align_feature_and_target(
            feature_data, target_data, rule_based=is_rule_based
        )
        if feature_data.name is None:
            feature_data.name = column_name
        self.binning_model.fit(feature_data, aligned_target)
        return self

    def predict(
        self,
        candles_df: pd.DataFrame,
        strategy: DirectionInput = Direction.LONG,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Predict: update bias nodes, extract features, predict with binning model.

        Dispatches to vectorized_predict (cached) or stream_predict based on use_cache setting.

        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with candles. Must have columns: datetime, open, high, low, close, volume, ticker, timeframe
        strategy : DirectionInput, default=Direction.LONG
            Strategy to use.
        start_date : datetime, optional
            Start date for vectorized predict (required if use_cache=True)
        end_date : datetime, optional
            End date for vectorized predict (required if use_cache=True)

        Returns
        -------
        pd.Series
            Predictions (binary signals or scaled positions) indexed by candle datetimes
            Returns predictions for ALL input candles, not just new ones.
        """
        if self.use_cache:
            return self.vectorized_predict(candles_df, strategy, start_date, end_date)
        return self.stream_predict(candles_df, strategy)

    def stream_predict(
        self,
        candles_df: pd.DataFrame,
        strategy: DirectionInput = Direction.LONG
    ) -> pd.Series:
        """
        Predict using streaming candle-by-candle processing.

        This is the original predict implementation.

        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with candles.
        strategy : DirectionInput, default=Direction.LONG
            Strategy to use.

        Returns
        -------
        pd.Series
            Predictions indexed by candle datetimes
        """
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
        
        # Collect input datetimes (vectorized, preserves row order)
        all_input_datetimes = pd.to_datetime(candles_df['datetime']).tolist()
        
        # Extract features for all tickers
        for ticker in normalized_tickers:
            # Handle both enum and string ticker values in dataframe
            ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
            ticker_candles = candles_df[
                (candles_df['ticker'] == ticker) | (candles_df['ticker'] == ticker_str)
            ].copy()
            if 'datetime' in ticker_candles.columns:
                ticker_candles = ticker_candles.sort_values('datetime')
            # Extract features for this ticker (itertuples is much faster than iterrows)
            for row in ticker_candles.itertuples(index=False):
                candle = Candle.from_row_fast(row)
                self.add_candle(candle, candle.tf, ticker=ticker)
        
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
        
        # NaN handling: for rule-based, 0 means neutral; filling NaN with 0 would inject fake signals.
        # For continuous binning, fillna(0) is acceptable. For rule-based, drop NaN rows.
        is_rule_based = getattr(self.binning_model, "model_type", None) == "rule_based"
        has_nan = input_feature_data.isna().any()
        if hasattr(has_nan, "any"):
            has_nan = bool(has_nan)
        if has_nan:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"BaseModel.predict() has {input_feature_data.isna().sum()} NaN values out of {len(input_feature_data)}. "
                f"This indicates bias nodes are not processing all candles correctly."
            )
            if is_rule_based:
                input_feature_data = input_feature_data.dropna()
                if input_feature_data.empty:
                    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            else:
                input_feature_data = input_feature_data.fillna(0.0)

        # Predict with binning model
        predictions = self.binning_model.predict(input_feature_data, strategy=strategy)

        return predictions

    def vectorized_predict(
        self,
        candles_df: pd.DataFrame,
        strategy: DirectionInput = Direction.LONG,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Predict using vectorized cached data.

        Uses get_cached_values() from bias nodes for bulk feature extraction.

        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with candles. Used for extracting date range if not provided.
        strategy : DirectionInput, default=Direction.LONG
            Strategy to use.
        start_date : datetime, optional
            Start date for cached data. If None, inferred from candles_df.
        end_date : datetime, optional
            End date for cached data. If None, inferred from candles_df.

        Returns
        -------
        pd.Series
            Predictions indexed by candle datetimes
        """
        from utils.cache.bias_node_cache import CacheMissError

        # Infer date range from candles_df if not provided
        if start_date is None:
            start_date = pd.to_datetime(candles_df['datetime']).min()
        if end_date is None:
            end_date = pd.to_datetime(candles_df['datetime']).max()

        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Determine which tickers are actually in the input candles
        if 'ticker' in candles_df.columns:
            candles_ticker_set = {
                _normalize_ticker_key(ticker_val)
                for ticker_val in candles_df['ticker'].unique()
            }
            tickers_to_predict = [
                ticker
                for ticker in self.tickers
                if _normalize_ticker_key(ticker) in candles_ticker_set
            ]
        else:
            # No ticker column - use all tickers (single-ticker mode)
            tickers_to_predict = self.tickers
        
        # Collect cached features from relevant bias nodes
        all_features = []
        for ticker in tickers_to_predict:
            for tf in self.bias_node_spec['timeframes']:
                bias_node = self.bias_nodes[(ticker, tf)]
                try:
                    cached_values = bias_node.get_cached_values(
                        start=start_date,
                        end=end_date,
                        require_cache=True
                    )
                    if cached_values is not None and len(cached_values) > 0:
                        all_features.append(cached_values)
                except CacheMissError as exc:
                    logger.error(
                        f"Cache miss for {bias_node.module_name} ({ticker}, {tf}). "
                        f"Vectorized predict requires complete cache coverage.",
                        exc_info=True
                    )
                    raise exc

        if not all_features:
            raise ValueError(
                f"No cached features available for any bias node. "
                f"Run cache population first or set use_cache=False."
            )

        # Combine features from all tickers/timeframes
        if len(all_features) == 1:
            feature_data = all_features[0]
        else:
            # Concatenate and aggregate by datetime
            # For multi-timeframe: take mean across timeframes for same ticker
            # For multi-ticker (should not happen - ensemble passes single ticker): take mean
            combined = pd.concat(all_features)
            feature_data = combined.groupby(combined.index).mean()

        # NaN handling: for rule-based, 0 means neutral; filling NaN with 0 would inject fake signals.
        # For continuous binning, fillna(0) is acceptable. For rule-based, drop NaN rows.
        is_rule_based = getattr(self.binning_model, "model_type", None) == "rule_based"
        has_nan = feature_data.isna().any()
        if hasattr(has_nan, "any"):
            has_nan = bool(has_nan)
        if has_nan:
            logger.warning(
                f"BaseModel.vectorized_predict() has {feature_data.isna().sum()} NaN values. "
                f"{'Dropping NaNs (rule-based).' if is_rule_based else 'Filling with 0.0.'}"
            )
            if is_rule_based:
                feature_data = feature_data.dropna()
                if feature_data.empty:
                    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            else:
                feature_data = feature_data.fillna(0.0)

        # Predict with binning model
        predictions = self.binning_model.predict(feature_data, strategy=strategy)

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
        from utils.core.enums import Ticker

        # Generate feature_column from bias_node_spec if not set (for unfitted models)
        if self.feature_column is None:
            primary_tf = self.bias_node_spec['timeframes'][0]
            output_feature = 'signal'
            if self.tickers:
                primary_ticker = self.tickers[0]
                primary_node = self.bias_nodes.get((primary_ticker, primary_tf))
                if primary_node and hasattr(primary_node, 'output_features') and primary_node.output_features:
                    output_feature = primary_node.output_features[0]
            self.feature_column = helpers.build_feature_column_name(
                module=self.bias_node_spec['module_name'],
                feature=output_feature,
                tf=primary_tf,
                params=self.bias_node_spec['params']
            )

        parsed = helpers.parse_feature_column_name(self.feature_column)
        tf = parsed.get('tf')
        tf_name = tf.name if hasattr(tf, "name") else str(tf)
        feature_name = (
            f"{parsed.get('module')}_{parsed.get('feature')}_{tf_name}"
            if parsed.get('module') and parsed.get('feature') and tf_name
            else self.feature_column
        )
        bias_node_params = dict(self.bias_node_spec.get('params', {}))
        bias_node_spec = {
            'module_name': self.bias_node_spec['module_name'],
            'timeframes': self.bias_node_spec['timeframes'],
        }
        model_id = add_feature_to_ensemble(
            feature_name=feature_name,
            bias_node_spec=bias_node_spec,
            bias_node_params=bias_node_params,
            base_model=self,
            ensemble_dir=ensemble_dir,
            tickers=tickers,
        )

        # Initialize decay monitoring if model is fitted
        bm = self.binning_model
        if (
            bm.is_fitted_
            and hasattr(bm, "_training_feature_data")
            and hasattr(bm, "_training_target_data")
        ):
            try:
                from ensemble.monitoring_store import initialize_monitoring
                signal_vector = bm.get_fitted_vector(strategy=bm.strategy)
                initialize_monitoring(
                    ensemble_dir=ensemble_dir,
                    feature_name=feature_name,
                    model_id=model_id,
                    signals=signal_vector,
                    targets=bm._training_target_data,
                )
            except FileExistsError:
                pass  # Already initialized — idempotent

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

        parsed = helpers.parse_feature_column_name(self.feature_column)
        tf = parsed.get('tf')
        tf_name = tf.name if hasattr(tf, "name") else str(tf)
        feature_name = (
            f"{parsed.get('module')}_{parsed.get('feature')}_{tf_name}"
            if parsed.get('module') and parsed.get('feature') and tf_name
            else self.feature_column
        )

        fitted_params = self.binning_model.get_fitted_params()
        if fitted_params.get("model_version") != "binning_v2":
            raise ValueError(
                "Unsupported fitted schema. Expected 'binning_v2'. "
                "Refit model with current binning implementation."
            )

        update_base_model_fitted_params(
            ensemble_dir=ensemble_dir,
            feature_name=feature_name,
            model_id=model_id,
            fitted_params=fitted_params,
            train_start=train_start,
            train_end=train_end
        )
