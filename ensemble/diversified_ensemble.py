"""
Diversified Ensemble Class for Strategy Combination

This module implements a diversified ensemble class that combines multiple
binary trading strategies using intra-feature correlation for diversified weights
and applies risk-adjusted position sizing based on volatility and exposure fractions.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import utils.helpers as helpers
from utils.enums import TimeFrame, Ticker
from utils.models import Candle
from .ensemble_utils import filter_dataframe_by_timeframe

logger = logging.getLogger(__name__)


def _normalize_ticker_name(ticker_val: object) -> str:
    """Normalize ticker identifiers (enum, string, etc.) to a bare ticker name string."""
    if isinstance(ticker_val, Ticker):
        return ticker_val.name
    if isinstance(ticker_val, str):
        return ticker_val.replace("Ticker.", "")
    return getattr(ticker_val, "name", str(ticker_val))

class DiversifiedEnsemble:
    """
    Diversified ensemble class for combining binary strategy signals.
    
    This ensemble class is designed for trading strategies where:
    - Features are binary (0 or 1) indicating trade signals
    - Target is continuous log returns (required but ignored in this implementation)
    - Weights are derived from intra-feature correlations for diversification
    - Final prediction includes risk scaling and exposure adjustment
    - Supports instrument-level weighting for sector diversification
    
    Parameters
    ----------
    target_volatility : float, default=0.15
        Target annual portfolio volatility (τ in the formula, e.g., 0.15 = 15%)
    instrument_weights : dict, optional
        Dictionary mapping ticker symbols to weights for instrument-level allocation
    config_path : str, optional
        Path to saved configuration file for loading pre-fitted parameters
    save_path : str, optional
        Default path to save fitted parameters
        
    Attributes
    ----------
    weights_ : dict
        Feature name -> weight mapping after fitting (based on diversification)
    exposure_fractions_ : dict
        Feature name -> exposure fraction mapping (fraction of time in market)
    feature_names_ : list
        List of feature column names
    target_volatility_ : float
        Target annual portfolio volatility
    unique_tickers_ : list
        List of unique ticker symbols seen during training
    instrument_weights_ : dict
        Ticker symbol -> weight mapping for instrument allocation
    n_tickers_ : int
        Number of unique ticker symbols
    is_fitted_ : bool
        Whether the model has been fitted
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        instrument_weights: Optional[Dict[str, float]] = None,
        config_path: Optional[str] = None,
        save_path: Optional[str] = None,
        control_file_path: Optional[str] = None,
        base_tf: Optional[TimeFrame] = None
    ):
        self.target_volatility = target_volatility
        self.instrument_weights = instrument_weights
        self.save_path = save_path
        
        # Base model ownership
        self.base_models: Dict[str, Any] = {}  # Dict[str, BaseModel]
        self.column_to_model: Dict[str, str] = {}  # Map feature columns to model names
        self.required_columns: List[str] = []
        self.control_file_data: Optional[Dict[str, Any]] = None
        self.base_tf: Optional[TimeFrame] = None  # Will be set from file or parameter
        
        # Fitted parameters (set during fit() or load_config())
        self.weights_ = None
        self.exposure_fractions_ = None
        self.model_exposure_fractions_ = None  # Based on 1/n_bins per model
        self.feature_names_ = None
        self.target_volatility_ = None
        self.unique_tickers_ = None
        self.instrument_weights_ = None
        self.n_tickers_ = None
        self.is_fitted_ = False
        
        # Caching for fit and predict operations
        # Cache key: (date_range_start, date_range_end) as tuple of dates
        self._fit_cache: Dict[Tuple[Any, Any], bool] = {}  # Maps date_range -> is_fitted flag
        self._predict_cache: Dict[Tuple[Any, ...], Union[pd.DataFrame, Dict[str, Any]]] = {}  # Maps (date_range, strategy) -> predictions
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Validate that control_file_path is provided
        if control_file_path is None:
            raise ValueError(
                "control_file_path must be provided. "
                "This is the unified control file containing base model configs and optional fitted parameters."
            )
        
        # Initialize from control file
        self._initialize_from_control_file(control_file_path)
        
        # Set base_tf from parameter if provided (overrides file value)
        if base_tf is not None:
            self.base_tf = base_tf
        
        # Load configuration if provided (legacy support)
        if config_path is not None:
            self.load_config(config_path)
    
    def _validate_input_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        ticker: Optional[Union[pd.Series, np.ndarray]] = None,
        volatility: Optional[Union[pd.Series, np.ndarray]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        method: str = "fit"
    ) -> tuple:
        """
        Validate input data format and requirements.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix with binary values (no volatility column)
        ticker : pd.Series or np.ndarray, optional
            Ticker symbols for each sample
        volatility : pd.Series or np.ndarray, optional
            Volatility values for each sample
        y : pd.Series or np.ndarray, optional
            Target values (required for fit, not for predict)
        method : str
            Method name for error messages ('fit' or 'predict')
            
        Returns
        -------
        tuple
            (validated_X, validated_ticker, validated_volatility, validated_y)
            
        Raises
        ------
        ValueError
            If validation fails
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            raise ValueError(
                f"{method}() requires DataFrame input with column names for feature identification."
            )
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"{method}() requires pandas DataFrame input")
        
        # Get feature columns (all columns in X)
        feature_cols = list(X.columns)
        
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found in input DataFrame")
        
        # Validate feature columns are binary for fit method
        if method == "fit":
            for col in feature_cols:
                unique_vals = X[col].dropna().unique()
                if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    raise ValueError(
                        f"Feature column '{col}' must be binary (0 or 1). "
                        f"Found unique values: {sorted(unique_vals)}"
                    )
        
        # Validate ticker parameter
        validated_ticker = None
        if ticker is not None:
            if isinstance(ticker, np.ndarray):
                validated_ticker = pd.Series(ticker, index=X.index)
            elif isinstance(ticker, pd.Series):
                validated_ticker = ticker
            else:
                raise ValueError("ticker parameter must be pandas Series or numpy array")
            
            if len(validated_ticker) != len(X):
                raise ValueError(
                    f"ticker parameter length ({len(validated_ticker)}) must match X length ({len(X)})"
                )
        
        # Validate volatility parameter
        validated_volatility = None
        if volatility is not None:
            if isinstance(volatility, np.ndarray):
                validated_volatility = pd.Series(volatility, index=X.index)
            elif isinstance(volatility, pd.Series):
                validated_volatility = volatility
            else:
                raise ValueError("volatility parameter must be pandas Series or numpy array")
            
            if len(validated_volatility) != len(X):
                raise ValueError(
                    f"volatility parameter length ({len(validated_volatility)}) must match X length ({len(X)})"
                )
            
            # Check for zero or negative volatility
            if (validated_volatility <= 0).any():
                raise ValueError(
                    "volatility parameter contains non-positive values. "
                    "All volatility values must be positive."
                )
        
        # Validate target if provided
        validated_y = None
        if y is not None:
            if isinstance(y, np.ndarray):
                validated_y = pd.Series(y, index=X.index)
            elif isinstance(y, pd.Series):
                validated_y = y
            else:
                raise ValueError("Target y must be pandas Series or numpy array")
            
            if len(validated_y) != len(X):
                raise ValueError(
                    f"Target y length ({len(validated_y)}) must match X length ({len(X)})"
                )
            
            # Check target is continuous
            if validated_y.dtype not in ['float64', 'float32']:
                raise ValueError(
                    f"Target y must be continuous (float). Found dtype: {validated_y.dtype}"
                )
        
        return X, validated_ticker, validated_volatility, validated_y
    
    def _initialize_from_control_file(self, filepath: str) -> None:
        """
        Initialize ensemble from unified control file.
        
        Parameters
        ----------
        filepath : str
            Path to control file
        """
        from ensemble.ensemble_utils import (
            parse_control_file,
            create_base_model_from_config
        )
        
        # Load control file
        control_file = parse_control_file(filepath)
        
        # Extract metadata
        metadata = control_file.get('metadata', {})
        is_fit = metadata.get('is_fit', False)
        
        # Extract base_tf from metadata if available
        if 'base_tf' in metadata:
            tf_str = metadata['base_tf']
            try:
                self.base_tf = TimeFrame[tf_str]
            except (KeyError, AttributeError):
                pass
        
        # If base_tf not in metadata, try to parse from filename
        if self.base_tf is None:
            import os
            filename = os.path.basename(filepath)
            name_parts = os.path.splitext(filename)[0].split('_')
            if len(name_parts) > 1:
                tf_str = name_parts[-1]
                try:
                    self.base_tf = TimeFrame[tf_str]
                except (KeyError, AttributeError):
                    pass
        
        # Store control file data
        self.control_file_data = control_file
        
        # Load fitted ensemble state if is_fit=True
        if is_fit:
            fitted_ensemble = control_file.get('fitted_ensemble', {})
            if fitted_ensemble:
                self.weights_ = fitted_ensemble.get('weights')
                self.exposure_fractions_ = fitted_ensemble.get('exposure_fractions')
                self.model_exposure_fractions_ = fitted_ensemble.get('model_exposure_fractions')
                self.feature_names_ = fitted_ensemble.get('feature_names')
                self.target_volatility_ = fitted_ensemble.get('target_volatility')
                self.unique_tickers_ = fitted_ensemble.get('unique_tickers')
                self.instrument_weights_ = fitted_ensemble.get('instrument_weights')
                self.n_tickers_ = fitted_ensemble.get('n_tickers')
                self.is_fitted_ = True
        
        # Get fitted base models if available
        fitted_base_models = control_file.get('fitted_base_models', {}) if is_fit else {}
        
        # Initialize base models
        self.base_models = {}
        self.column_to_model = {}
        self.required_columns = []
        
        for model_config in control_file['base_models']:
            model_name = model_config['name']
            feature_column = model_config['feature_column']
            
            # Get fitted params if available
            fitted_params = fitted_base_models.get(model_name)
            
            # Create base model instance
            # create_base_model_from_config will use tickers from model_config if available
            base_model = create_base_model_from_config(model_config, fitted_params=fitted_params)
            
            # Store model
            self.base_models[model_name] = base_model
            self.column_to_model[feature_column] = model_name
            self.required_columns.append(feature_column)
    
    def _create_base_model_instance(
        self,
        model_config: Dict[str, Any],
        fitted_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Factory method to create base model instances.
        
        Parameters
        ----------
        model_config : Dict[str, Any]
            Base model configuration
        fitted_params : Dict[str, Any], optional
            Fitted parameters to restore
            
        Returns
        -------
        BaseModel
            Instantiated base model
        """
        from ensemble.ensemble_utils import create_base_model_from_config
        return create_base_model_from_config(model_config, fitted_params=fitted_params)
    
    def get_required_columns(self) -> List[str]:
        """
        Return list of feature columns needed by this ensemble.
        
        Returns
        -------
        List[str]
            List of required feature column names
        """
        return self.required_columns.copy()
    
    def get_required_bias_nodes(self) -> List[Dict[str, Any]]:
        """
        Get bias node specifications needed by this ensemble.
        
        Parses feature column names to extract bias node specs that can be
        used to initialize MLManager with the required bias nodes.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of bias node specifications in format:
            [{'module_name': str, 'timeframes': [TimeFrame], 'params': dict}, ...]
        """
        from ensemble.ensemble_utils import extract_bias_node_specs_from_control_file
        
        # If we have a control file path, use it
        if hasattr(self, 'control_file_data') and self.control_file_data is not None:
            # Extract from control file data
            bias_node_specs = []
            seen_specs = set()
            
            for model_config in self.control_file_data.get('base_models', []):
                feature_column = model_config.get('feature_column')
                if not feature_column:
                    continue
                
                # Parse feature column name to extract module, params, timeframe
                parsed = helpers.parse_feature_column_name(feature_column)
                module_name = parsed.get('module')
                params = parsed.get('params', {})
                tf = parsed.get('tf')
                
                if module_name is None or tf is None:
                    continue
                
                # Convert tf to TimeFrame enum if it's a string
                if isinstance(tf, str):
                    try:
                        tf = TimeFrame[tf]
                    except (KeyError, AttributeError):
                        continue
                
                # Create spec key for deduplication
                spec_key = (module_name, tf.name if hasattr(tf, 'name') else str(tf), tuple(sorted(params.items())))
                if spec_key in seen_specs:
                    continue
                seen_specs.add(spec_key)
                
                # Create bias node spec
                bias_node_spec = {
                    'module_name': module_name,
                    'timeframes': [tf],
                    'params': params
                }
                bias_node_specs.append(bias_node_spec)
            
            return bias_node_specs
        else:
            # Fallback: extract from required_columns
            bias_node_specs = []
            seen_specs = set()
            
            for feature_column in self.required_columns:
                # Parse feature column name
                parsed = helpers.parse_feature_column_name(feature_column)
                module_name = parsed.get('module')
                params = parsed.get('params', {})
                tf = parsed.get('tf')
                
                if module_name is None or tf is None:
                    continue
                
                # Convert tf to TimeFrame enum if it's a string
                if isinstance(tf, str):
                    try:
                        tf = TimeFrame[tf]
                    except (KeyError, AttributeError):
                        continue
                
                # Create spec key for deduplication
                spec_key = (module_name, tf.name if hasattr(tf, 'name') else str(tf), tuple(sorted(params.items())))
                if spec_key in seen_specs:
                    continue
                seen_specs.add(spec_key)
                
                # Create bias node spec
                bias_node_spec = {
                    'module_name': module_name,
                    'timeframes': [tf],
                    'params': params
                }
                bias_node_specs.append(bias_node_spec)
            
            return bias_node_specs
    
    def _calculate_diversified_weights(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate diversified weights based on intra-feature correlations.
        
        Uses negative correlation between features to promote diversification.
        Features with lower correlation to other features get higher weights.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with binary columns
            
        Returns
        -------
        Dict[str, float]
            Feature name -> weight mapping
        """
        feature_cols = list(X.columns)
        
        if len(feature_cols) == 1:
            # Single feature case
            return {feature_cols[0]: 1.0}
        
        # Calculate correlation matrix between features
        corr_matrix = X[feature_cols].corr()
        
        # Calculate diversification score for each feature
        # Lower average absolute correlation = higher diversification = higher weight
        diversification_scores = {}
        
        for feature in feature_cols:
            # Average absolute correlation with other features
            other_features = [f for f in feature_cols if f != feature]
            if len(other_features) > 0:
                avg_abs_corr = abs(corr_matrix.loc[feature, other_features]).mean()
                # Convert to diversification score (inverse relationship)
                diversification_scores[feature] = 1.0 / (1.0 + avg_abs_corr)
            else:
                diversification_scores[feature] = 1.0
        
        # Normalize weights to sum to 1
        total_score = sum(diversification_scores.values())
        if total_score == 0:
            # Fallback to equal weights
            weights = {feature: 1.0 / len(feature_cols) for feature in feature_cols}
        else:
            weights = {feature: score / total_score 
                      for feature, score in diversification_scores.items()}
        
        return weights
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        ticker: Union[pd.Series, np.ndarray],
        volatility: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        instrument_weights: Optional[Dict[str, float]] = None,
        normalization_data: Optional[pd.DataFrame] = None
    ) -> 'DiversifiedEnsemble':
        """
        Fit the ensemble model to training data.
        
        First fits all base models on their respective features, generates binary signals,
        then fits the ensemble on the binary signals.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with raw feature values (not binary).
            Must contain all required feature columns.
        ticker : pd.Series or np.ndarray
            Ticker symbols for each sample
        volatility : pd.Series or np.ndarray
            Volatility values for each sample (positive values)
        y : pd.Series or np.ndarray
            Target values (returns) for training base models
        instrument_weights : dict, optional
            Custom instrument weights mapping ticker -> weight
        normalization_data : pd.DataFrame, optional
            Normalization data (EWSD/ATR) for each feature column.
            Columns should match feature columns in X.
            
        Returns
        -------
        self
            Fitted ensemble model
            
        Raises
        ------
        ValueError
            If input validation fails or insufficient data
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if ticker is None or volatility is None:
            raise ValueError("ticker and volatility parameters are required for fit()")
        
        if y is None:
            raise ValueError("y parameter is required for fit()")
        
        # Filter X by timeframe if base_tf is set
        if self.base_tf is not None:
            X = filter_dataframe_by_timeframe(X, self.base_tf)
            if normalization_data is not None:
                normalization_data = filter_dataframe_by_timeframe(normalization_data, self.base_tf)
        
        # Check that all required columns are present
        missing_columns = set(self.required_columns) - set(X.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required feature columns: {missing_columns}. "
                f"Required: {self.required_columns}"
            )
        
        # Filter X to only required columns
        X_filtered = X[self.required_columns].copy()
        
        # Convert ticker, volatility, y to Series if needed
        if isinstance(ticker, np.ndarray):
            ticker = pd.Series(ticker, index=X_filtered.index)
        if isinstance(volatility, np.ndarray):
            volatility = pd.Series(volatility, index=X_filtered.index)
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X_filtered.index)
        
        # Step 1: Fit each base model on its feature column
        binary_signals = {}
        for model_name, base_model in self.base_models.items():
            # Find feature column for this model
            feature_column = None
            for col, name in self.column_to_model.items():
                if name == model_name:
                    feature_column = col
                    break
            if feature_column is None:
                raise ValueError(f"Could not find feature column for model: {model_name}")
            
            # Get feature data
            feature_data = X_filtered[feature_column]
            
            # Get normalization data for this feature if provided
            norm_data = None
            if normalization_data is not None and feature_column in normalization_data.columns:
                norm_data = normalization_data[feature_column]
            
            # Fit model if not already fitted (from ensemble_model)
            if not base_model.is_fitted_:
                base_model.fit(
                    feature_data=feature_data,
                    target_data=y,
                    normalization_data=norm_data
                )
            
            # Generate binary signals using the model's strategy
            binary_signals[model_name] = base_model.predict(
                feature_data,
                strategy=base_model.strategy,
                normalization_data=norm_data
            )
        
        # Step 2: Combine binary signals into DataFrame
        binary_df = pd.DataFrame(binary_signals, index=X_filtered.index)
        
        # Step 3: Fit ensemble on binary signals
        # Use the existing ensemble fit logic but with binary_df as X
        feature_cols = list(binary_df.columns)
        
        # Align data and drop NaN values
        df = pd.DataFrame({
            **{col: binary_df[col] for col in feature_cols},
            'ticker': ticker,
            'volatility': volatility,
            'target': y
        }).dropna()
        
        if len(df) < 10:
            raise ValueError(f"Insufficient data after dropping NaN. Need at least 10 samples, got {len(df)}")
        
        # Calculate diversified weights
        self.weights_ = self._calculate_diversified_weights(df[feature_cols])
        
        # Calculate exposure fractions (h_i) - empirical (fraction of time signal=1)
        self.exposure_fractions_ = {}
        for col in feature_cols:
            feature_data = df[col]
            # Fraction of time feature == 1
            self.exposure_fractions_[col] = feature_data.mean()

        # Calculate model exposure fractions from actual binary signals (empirical)
        # This accounts for non-uniform binning where bins may have different sample counts
        # EXCEPTION: buy_hold models are always in market (h_i = 1.0)
        self.model_exposure_fractions_ = {}
        for model_name, base_model in self.base_models.items():
            # Check if this is a buy_hold model (always in market)
            bias_node_spec = getattr(base_model, 'bias_node_spec', None)
            is_buy_hold = False
            if bias_node_spec is not None:
                is_buy_hold = bias_node_spec.get('module_name') == 'buy_hold'
            
            if is_buy_hold:
                # Buy_hold is always in market: h_i = 1.0
                self.model_exposure_fractions_[model_name] = 1.0
            else:
                # Calculate from actual binary signals (fraction of time signal=1)
                # This handles non-uniform binning correctly
                if model_name in binary_df.columns:
                    self.model_exposure_fractions_[model_name] = binary_df[model_name].mean()
                else:
                    # Fallback: use theoretical 1/n_bins if binary signals not available
                    n_bins = getattr(base_model, 'n_bins', 10)
                    self.model_exposure_fractions_[model_name] = 1.0 / n_bins
            
        # Store feature names (base model names, not column names)
        self.feature_names_ = feature_cols
        
        # Store target volatility
        self.target_volatility_ = self.target_volatility
        
        # Process ticker information
        self.unique_tickers_ = sorted(df['ticker'].unique())
        self.n_tickers_ = len(self.unique_tickers_)
        
        # Set up instrument weights
        if instrument_weights is not None:
            # Validate custom weights
            missing_tickers = set(self.unique_tickers_) - set(instrument_weights.keys())
            if missing_tickers:
                raise ValueError(f"instrument_weights missing tickers: {missing_tickers}")
            
            # Normalize weights
            total_weight = sum(instrument_weights[ticker] for ticker in self.unique_tickers_)
            self.instrument_weights_ = {
                ticker: instrument_weights[ticker] / total_weight 
                for ticker in self.unique_tickers_
            }
        elif self.instrument_weights is not None:
            # Use instance-level weights
            missing_tickers = set(self.unique_tickers_) - set(self.instrument_weights.keys())
            if missing_tickers:
                raise ValueError(f"Instance instrument_weights missing tickers: {missing_tickers}")
            
            total_weight = sum(self.instrument_weights[ticker] for ticker in self.unique_tickers_)
            self.instrument_weights_ = {
                ticker: self.instrument_weights[ticker] / total_weight 
                for ticker in self.unique_tickers_
            }
        else:
            # Equal weighting
            equal_weight = 1.0 / self.n_tickers_
            self.instrument_weights_ = {ticker: equal_weight for ticker in self.unique_tickers_}
        
        # Set fitted status
        self.is_fitted_ = True
        
        return self
    
    def _get_supported_tickers(self) -> set:
        """
        Get the set of tickers supported by all base models in this ensemble.
        
        Returns the intersection of tickers from all base models (all models must support a ticker).
        If no base models exist, returns empty set.
        
        Returns
        -------
        set
            Set of ticker enums supported by all base models
        """
        if not self.base_models:
            return set()
        
        # Get tickers from first base model
        first_model = next(iter(self.base_models.values()))
        supported_tickers = set(getattr(first_model, 'tickers', []))
        
        # Intersect with tickers from all other base models
        # All models must support a ticker for it to be in the ensemble
        for base_model in self.base_models.values():
            model_tickers = set(getattr(base_model, 'tickers', []))
            supported_tickers = supported_tickers.intersection(model_tickers)
        
        return supported_tickers
    
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
    
    def fit_from_candles(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series
    ) -> 'DiversifiedEnsemble':
        """
        Fit all base models using candles DataFrame.
        
        This is the new DataFrame-based API for fitting ensembles.
        Uses ensemble-level caching to extract features once per (bias_node_spec, ticker)
        combination and reuse them across base models.
        
        Uses date_range-based caching to avoid refitting for the same date range.
        
        CRITICAL ALIGNMENT FIX:
        - Features at time T are based on candle T
        - Returns at time T represent return from T-1 to T
        - We need to shift returns forward by 1 so feature at T pairs with return from T to T+1
        - Must group by ticker first, then shift, then aggregate to match feature aggregation
        
        IMPORTANT: Filters candles_df to only include tickers supported by all base models
        in this ensemble. This prevents errors when candles contain tickers not in the ensemble.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
        target_data : pd.Series
            Target values (returns) from all tickers, indexed by datetime.
            May have duplicate datetime indices (one per ticker).
            NOTE: This parameter is currently ignored - returns are calculated directly from candles.
            
        Returns
        -------
        self
            Fitted ensemble model
        """
        # Check cache first
        date_range_key = self._get_date_range_key(candles_df)
        if date_range_key in self._fit_cache and self.is_fitted_:
            logger.debug(f"DiversifiedEnsemble.fit_from_candles() cache HIT for date_range: {date_range_key}")
            self._cache_hits += 1
            return self
        
        logger.debug(f"DiversifiedEnsemble.fit_from_candles() cache MISS for date_range: {date_range_key}")
        self._cache_misses += 1
        
        # Get supported tickers from all base models
        supported_tickers = self._get_supported_tickers()
        
        if not supported_tickers:
            raise ValueError(
                "No base models found or base models have no tickers configured. "
                "Cannot fit ensemble without knowing which tickers are supported."
            )
        
        # Convert ticker enums to strings for filtering
        # Handle both enum and string ticker formats
        supported_ticker_names = set()
        for ticker in supported_tickers:
            if isinstance(ticker, Ticker):
                supported_ticker_names.add(ticker.name)
            elif isinstance(ticker, str):
                supported_ticker_names.add(ticker)
            else:
                # Try to get name attribute
                supported_ticker_names.add(getattr(ticker, 'name', str(ticker)))
        
        # Filter candles_df to only include supported tickers
        candles_df = candles_df.copy()
        candles_df['ticker_normalized'] = candles_df['ticker'].apply(_normalize_ticker_name)
        filtered_candles = candles_df[candles_df['ticker_normalized'].isin(supported_ticker_names)].copy()
        filtered_candles = filtered_candles.drop(columns=['ticker_normalized'])
        
        if filtered_candles.empty:
            raise ValueError(
                f"No candles found for supported tickers: {sorted(supported_ticker_names)}. "
                f"Available tickers in candles_df: {sorted(candles_df['ticker'].apply(_normalize_ticker_name).unique())}"
            )
        
        logger.debug(
            f"Filtered candles: {len(candles_df)} -> {len(filtered_candles)} rows "
            f"(keeping tickers: {sorted(supported_ticker_names)})"
        )
        
        # Calculate returns directly from filtered candles with proper alignment
        # CRITICAL: Group by ticker, calculate returns, shift forward by 1, then aggregate
        aggregated_returns = self._calculate_aligned_returns_from_candles(filtered_candles)
        
        # Debug: Log aggregation info
        logger.debug(
            f"Returns calculation: filtered candles length={len(filtered_candles)}, "
            f"aggregated_returns length={len(aggregated_returns)}, "
            f"unique base datetimes in filtered candles={len(pd.to_datetime(filtered_candles['datetime']).dt.floor('s').unique())}"
        )
        
        # Fit each base model ONCE with all candles (not per ticker)
        # BaseModel.fit() handles multi-ticker aggregation internally
        for model_name, base_model in self.base_models.items():
            try:
                # Skip if already fitted (e.g., loaded from vault with fitted params)
                if base_model.is_fitted_:
                    logger.debug(f"Skipping already-fitted model '{model_name}'")
                    continue
                
                # Fit base model with filtered candles and aggregated returns
                # BaseModel.fit() will:
                # 1. Stream candles from all supported tickers
                # 2. Extract features and aggregate by base datetime (mean)
                # 3. Align aggregated returns with aggregated features
                base_model.fit(filtered_candles, aggregated_returns)
                
                # Debug: Log fitting results
                if base_model.binning_model.is_fitted_:
                    bin_stats = base_model.binning_model.bin_stats_
                    if bin_stats:
                        logger.debug(
                            f"Model '{model_name}' fitted: "
                            f"n_bins={base_model.binning_model.n_bins}, "
                            f"best_long_bin={base_model.binning_model.best_long_bin_}, "
                            f"n_bins_with_stats={len(bin_stats)}"
                        )
            except Exception as e:
                logger.error(
                    f"Error fitting base model '{model_name}': {e}",
                    exc_info=True
                )
        
        # Calculate model exposure fractions from actual binary signals (empirical)
        # This accounts for non-uniform binning where bins may have different sample counts
        # Generate binary signals from all base models to calculate actual exposure fractions
        if self.model_exposure_fractions_ is None:
            self.model_exposure_fractions_ = {}
            
            # Generate binary signals from all base models using filtered training candles
            for model_name, base_model in self.base_models.items():
                # Check if this is a buy_hold model (always in market)
                bias_node_spec = getattr(base_model, 'bias_node_spec', None)
                is_buy_hold = False
                if bias_node_spec is not None:
                    is_buy_hold = bias_node_spec.get('module_name') == 'buy_hold'
                
                if is_buy_hold:
                    # Buy_hold is always in market: h_i = 1.0
                    self.model_exposure_fractions_[model_name] = 1.0
                else:
                    # Generate binary signals for this model across supported tickers only
                    model_signals = []
                    # Get tickers supported by this specific base model
                    model_tickers = set(getattr(base_model, 'tickers', []))
                    model_ticker_names = {_normalize_ticker_name(t) for t in model_tickers}
                    
                    # Filter candles to only this model's supported tickers
                    model_candles = filtered_candles[
                        filtered_candles['ticker'].apply(_normalize_ticker_name).isin(model_ticker_names)
                    ].copy()
                    
                    if not model_candles.empty:
                        # Group by ticker and generate signals
                        for ticker_name in model_candles['ticker'].unique():
                            ticker_candles = model_candles[model_candles['ticker'] == ticker_name].copy()
                            try:
                                # Generate binary signals from base model
                                pred = base_model.predict(ticker_candles)
                                if pred is not None and len(pred) > 0:
                                    model_signals.append(pred)
                            except Exception as e:
                                logger.warning(
                                    f"Error generating signals for model '{model_name}' on ticker '{ticker_name}': {e}"
                                )
                    
                    # Calculate exposure fraction from actual binary signals
                    if model_signals:
                        all_signals = pd.concat(model_signals) if len(model_signals) > 1 else model_signals[0]
                        self.model_exposure_fractions_[model_name] = all_signals.mean()
                    else:
                        # Fallback: use theoretical 1/n_bins if no signals generated
                        n_bins = getattr(base_model, 'n_bins', 10)
                        self.model_exposure_fractions_[model_name] = 1.0 / n_bins
                        logger.warning(
                            f"Could not generate binary signals for model '{model_name}', "
                            f"using theoretical exposure fraction 1/{n_bins}"
                        )
        
        # Set target volatility if not already set
        if self.target_volatility_ is None:
            self.target_volatility_ = self.target_volatility
        
        # After fitting base models, we still need to fit the ensemble weights
        # This requires computing binary signals from all base models
        # For now, mark as fitted - full ensemble fitting can be done via the regular fit() method
        self.is_fitted_ = True
        
        # Store in cache
        self._fit_cache[date_range_key] = True
        
        return self
    
    def _calculate_aligned_returns_from_candles(
        self,
        candles_df: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate returns from candles with proper alignment for feature pairing.
        
        CRITICAL ALIGNMENT LOGIC:
        - Features at time T are extracted from candle T
        - Returns at time T represent return from T-1 to T (calculated from close[T-1] to close[T])
        - To pair feature at T with forward return, we need return from T to T+1
        - Solution: Calculate returns per ticker, shift forward by 1, then aggregate
        
        Steps:
        1. Group candles by ticker
        2. For each ticker: calculate log returns, shift forward by 1 period
        3. Remove microsecond precision (base datetime) to match BaseModel.get_feature()
        4. Aggregate by base datetime (mean across tickers) to match feature aggregation
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame with columns: datetime, ticker, close, etc.
            
        Returns
        -------
        pd.Series
            Aggregated returns indexed by base datetime (microseconds removed),
            with mean aggregation across tickers for each base datetime.
            Returns are shifted forward by 1 period so feature at T pairs with return from T to T+1.
        """
        if candles_df.empty:
            return pd.Series(dtype=float, name='returns')
        
        # Ensure datetime is datetime type
        candles_df = candles_df.copy()
        candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
        
        # Group by ticker and calculate returns with forward shift
        all_returns = []
        for ticker in candles_df['ticker'].unique():
            ticker_candles = candles_df[candles_df['ticker'] == ticker].copy()
            ticker_candles = ticker_candles.sort_values('datetime')
            
            if len(ticker_candles) < 2:
                continue
            
            # Calculate log returns: return at T = log(close[T] / close[T-1])
            # This represents return from T-1 to T
            ticker_candles['returns'] = np.log(ticker_candles['close'] / ticker_candles['close'].shift(1))
            
            # CRITICAL: Shift returns forward by 1 period
            # Feature at time T should pair with return from T to T+1
            # So we shift the return series forward: return[T] (from T-1 to T) -> return[T+1] (from T to T+1)
            ticker_candles['forward_returns'] = ticker_candles['returns'].shift(-1)
            
            # Set datetime as index
            ticker_candles = ticker_candles.set_index('datetime')
            
            # Extract forward returns (drop NaN from shift)
            ticker_forward_returns = ticker_candles['forward_returns'].dropna()
            
            if len(ticker_forward_returns) > 0:
                all_returns.append(ticker_forward_returns)
        
        if not all_returns:
            return pd.Series(dtype=float, name='returns')
        
        # Combine all ticker returns (may have duplicate datetime indices)
        combined_returns = pd.concat(all_returns)
        
        # For multi-ticker models, preserve all samples (don't aggregate by base datetime)
        # This matches the new BaseModel.get_feature() behavior which keeps all samples
        # Sort by datetime to ensure consistent ordering
        combined_returns = combined_returns.sort_index()
        
        # Debug: Log details
        logger.debug(
            f"Aligned returns calculation: input candles={len(candles_df)}, "
            f"combined returns={len(combined_returns)} samples (preserving all ticker samples), "
            f"date range={combined_returns.index.min()} to {combined_returns.index.max()}"
        )
        
        # Return combined returns with original datetimes (preserves all samples)
        aggregated_returns = combined_returns.copy()
        aggregated_returns.name = 'returns'
        
        return aggregated_returns
    
    def _calculate_volatility_from_candles(
        self,
        candles_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate blended volatility from EWSD bias nodes per ticker.
        
        Creates EWSD (Exponentially Weighted Standard Deviation) nodes for each ticker
        and streams candles to build up volatility estimates. EWSD nodes implement
        Carver's blended volatility:
        - 70% EWMA-32 (short-run estimate)
        - 30% long-run historical average
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame with columns: datetime, ticker, open, high, low, close, volume, timeframe
            
        Returns
        -------
        Dict[str, float]
            Mapping from ticker to annualized blended volatility (as decimal, not percentage)
        """
        from utils.enums import Ticker, TimeFrame
        from nodes.ewsd import EWSDNode
        from utils.models import Candle
        
        volatility_dict = {}
        
        # Group by ticker
        for ticker_name in candles_df['ticker'].unique():
            ticker_candles = candles_df[candles_df['ticker'] == ticker_name].copy()
            ticker_candles = ticker_candles.sort_values('datetime')
            
            if len(ticker_candles) == 0:
                volatility_dict[ticker_name] = 0.20  # Default fallback
                continue
            
            # Convert ticker name to Ticker enum
            try:
                if isinstance(ticker_name, str):
                    # Handle both 'ES' and 'Ticker.ES' formats
                    ticker_str = ticker_name.replace('Ticker.', '') if 'Ticker.' in ticker_name else ticker_name
                    ticker = Ticker[ticker_str]
                else:
                    ticker = ticker_name
            except (KeyError, AttributeError):
                logger.warning(f"Unknown ticker '{ticker_name}', using default volatility")
                volatility_dict[ticker_name] = 0.20
                continue
            
            # Create EWSD node for this ticker
            ewsd_node = EWSDNode(
                ticker=ticker,
                tf=TimeFrame.D,  # Use daily timeframe for volatility calculation
                lambda_short=0.06061,  # 32-day span
                long_run_window=2520,  # 10 years (2520 trading days)
                blend_short_weight=0.7,
                blend_long_weight=0.3
            )
            
            # Stream all candles to build up EWSD state
            ewsd_value = None
            for _, row in ticker_candles.iterrows():
                candle = Candle.from_row(row)
                ewsd_output = ewsd_node.add_candle(candle)
                if ewsd_output and len(ewsd_output) >= 2:
                    # ewsd_output[1] is annual_pct (in percentage)
                    # Convert to decimal
                    ewsd_value = ewsd_output[1] / 100.0
            
            # Use latest EWSD value or fallback
            if ewsd_value is None or np.isnan(ewsd_value):
                ticker_candles['returns'] = ticker_candles['close'].pct_change()
                daily_vol = ticker_candles['returns'].std()
                annual_vol = daily_vol * np.sqrt(252)  # Annualize
                ewsd_value = annual_vol if not np.isnan(annual_vol) else 0.20
                logger.warning(
                    f"EWSD calculation failed for ticker '{ticker_name}'. "
                    f"Using simple volatility calculation as fallback."
                )
            
            volatility_dict[ticker_name] = ewsd_value
        
        return volatility_dict
    
    def predict_from_candles(
        self,
        candles_df: pd.DataFrame,
        volatility: Optional[Dict[str, float]] = None,
        return_base_model_predictions: bool = False
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Aggregate predictions from all base models using candles DataFrame.
        
        Applies volatility scaling per base model before aggregation:
        F_i = (tau / (sigma * sqrt(h_i))) * X_i * N
        where N = number of instruments (accounts for instrument weights)
        
        This is the new DataFrame-based API for prediction.
        Routes candles to each base model, which computes features and predicts internally.
        
        Uses date_range-based caching to avoid recomputation for the same date range.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
        volatility : Dict[str, float], optional
            Volatility per ticker (annualized). If None, calculated from candles.
        return_base_model_predictions : bool, default=False
            If True, return base model-level predictions in result dict
            
        Returns
        -------
        pd.DataFrame or Dict[str, Any]
            If return_base_model_predictions=False: DataFrame with aggregated ensemble predictions
            If return_base_model_predictions=True: Dict with structure:
            {
                'ensemble': pd.DataFrame,  # Aggregated ensemble predictions
                'base_models': Dict[str, pd.DataFrame]  # Individual base model predictions
            }
        """
        if not self.is_fitted_:
            raise ValueError(
                "Ensemble must be fitted before calling predict_from_candles(). "
                "Call fit_from_candles() or fit() first."
            )
        
        # Check cache first
        date_range_key = self._get_date_range_key(candles_df)
        cache_key = (date_range_key, return_base_model_predictions)
        
        if cache_key in self._predict_cache:
            self._cache_hits += 1
            cache_msg = (
                f"DiversifiedEnsemble.predict_from_candles() cache HIT for date_range: {date_range_key} "
                f"(hits: {self._cache_hits}, misses: {self._cache_misses})"
            )
            logger.info(cache_msg)
            cached_result = self._predict_cache[cache_key]
            # Return a deep copy to avoid modifying cache
            return self._deep_copy_result(cached_result)
        
        logger.debug(f"DiversifiedEnsemble.predict_from_candles() cache MISS for date_range: {date_range_key}")
        self._cache_misses += 1
        
        # Calculate or use provided volatility
        if volatility is None:
            volatility = self._calculate_volatility_from_candles(candles_df)
        
        all_predictions = []
        base_model_predictions_dict = {}
        
        # Group candles by ticker
        for ticker_name in candles_df['ticker'].unique():
            ticker_candles = candles_df[candles_df['ticker'] == ticker_name].copy()
            
            # Get ticker enum
            try:
                ticker = Ticker[ticker_name] if isinstance(ticker_name, str) else ticker_name
            except (KeyError, AttributeError):
                logger.warning(f"Unknown ticker '{ticker_name}', skipping")
                continue
            
            # Get volatility for this ticker
            ticker_vol = volatility.get(ticker_name, 0.20)
            
            # Get predictions from each base model (BaseModel.predict() handles caching internally)
            ticker_predictions = []
            for model_name, base_model in self.base_models.items():
                try:
                    # Check if this base model supports this ticker
                    model_tickers = set(getattr(base_model, 'tickers', []))
                    if model_tickers:
                        # Normalize ticker names for comparison
                        model_ticker_names = {_normalize_ticker_name(t) for t in model_tickers}
                        normalized_ticker_name = _normalize_ticker_name(ticker_name)
                        
                        # Skip this base model if it doesn't support this ticker
                        if normalized_ticker_name not in model_ticker_names:
                            logger.debug(
                                f"Base model '{model_name}' does not support ticker '{ticker_name}'. "
                                f"Supported tickers: {model_ticker_names}. Skipping."
                            )
                            continue
                    
                    # BaseModel.predict() returns a Series indexed by datetime with binary signals
                    # BaseModel.predict() handles feature caching internally
                    pred = base_model.predict(ticker_candles)
                    
                    # Debug: Log prediction details
                    if len(pred) == 0:
                        logger.warning(
                            f"Base model '{model_name}' returned empty predictions for ticker '{ticker_name}'. "
                            f"Input candles: {len(ticker_candles)}"
                        )
                        continue
                    
                    # Apply volatility scaling per base model
                    # Formula: F_i = (tau / (sigma * sqrt(h_i))) * X_i
                    # Note: Instrument weights are applied at Portfolio layer, not here
                    # Get exposure fraction for this model
                    h_i = self.model_exposure_fractions_.get(model_name, 0.1)
                    sqrt_h_i = np.sqrt(max(h_i, 1e-8))
                    
                    # Calculate volatility-adjusted forecast
                    # X_i is the binary signal (pred.values)
                    forecast_if_active = self.target_volatility_ / (ticker_vol * sqrt_h_i)
                    
                    # Cap forecast at 2.0 (per spec: max position is 2.0)
                    forecast_if_active = min(forecast_if_active, 2.0)
                    
                    # Apply signal: forecast_if_active if signal=1, else 0
                    volatility_adjusted_forecast = forecast_if_active * pred.values
                    
                    # Convert to DataFrame with ticker and datetime
                    pred_df = pd.DataFrame({
                        'ticker': ticker_name,
                        'datetime': pred.index,
                        'model_name': model_name,
                        'forecast': volatility_adjusted_forecast
                    })
                    ticker_predictions.append(pred_df)
                    
                    # Store base model prediction if requested
                    if return_base_model_predictions:
                        # Create DataFrame for this base model (volatility-adjusted)
                        base_model_df = pd.DataFrame({
                            'ticker': ticker_name,
                            'datetime': pred.index,
                            'forecast_score': volatility_adjusted_forecast
                        })
                        
                        # Append to base model predictions (combine across tickers)
                        if model_name not in base_model_predictions_dict:
                            base_model_predictions_dict[model_name] = []
                        base_model_predictions_dict[model_name].append(base_model_df)
                        
                except Exception as e:
                    logger.error(
                        f"Error predicting with base model '{model_name}' for ticker '{ticker_name}': {e}",
                        exc_info=True
                    )
            
            if not ticker_predictions:
                # No predictions for this ticker - this is expected if no base models support it
                logger.debug(f"No ticker_predictions for ticker {ticker_name}. Base models: {list(self.base_models.keys())}")
                continue
                
            # Combine predictions for this ticker
            ticker_df = pd.concat(ticker_predictions, ignore_index=True)
            
            # Debug: Check if ticker_df is empty
            if ticker_df.empty:
                logger.warning(f"ticker_df is empty for ticker {ticker_name} after concatenating {len(ticker_predictions)} predictions")
                continue
            
            # Debug: Log ticker_df info
            logger.debug(f"ticker_df for {ticker_name}: shape={ticker_df.shape}, columns={ticker_df.columns.tolist()}, dtypes={ticker_df.dtypes.to_dict()}")
            
            # Aggregate by datetime (mean across models, or weighted if weights available)
            # Note: forecasts are already volatility-adjusted, so we just aggregate them
            if self.weights_ is not None:
                # Weighted aggregation: calculate weighted mean per datetime
                # First, add weights column
                ticker_df_weighted = ticker_df.copy()
                ticker_df_weighted['weight'] = ticker_df_weighted['model_name'].map(
                    lambda m: self.weights_.get(m, 1.0 / len(ticker_df_weighted))
                )
                ticker_df_weighted['weighted_forecast'] = ticker_df_weighted['forecast'] * ticker_df_weighted['weight']
                
                # Group by datetime and sum weighted forecasts and weights, then divide
                grouped = ticker_df_weighted.groupby('datetime').agg({
                    'weighted_forecast': 'sum',
                    'weight': 'sum'
                })
                grouped['forecast_score'] = grouped['weighted_forecast'] / grouped['weight']
                
                # Convert to DataFrame with datetime as column
                aggregated = grouped[['forecast_score']].reset_index()
            else:
                # Simple mean aggregation
                aggregated_series = ticker_df.groupby('datetime')['forecast'].mean()
                aggregated = aggregated_series.to_frame('forecast_score').reset_index()
            
            # Debug: Check if aggregated is empty
            if aggregated.empty:
                logger.warning(f"Aggregated result is empty for ticker {ticker_name}. ticker_df shape: {ticker_df.shape}")
                continue
            
            aggregated['ticker'] = ticker_name
            all_predictions.append(aggregated)
        
        if not all_predictions:
            empty_df = pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score'])
            if return_base_model_predictions:
                return {
                    'ensemble': empty_df,
                    'base_models': {}
                }
            return empty_df
        
        # Combine all ticker predictions for ensemble
        ensemble_result = pd.concat(all_predictions, ignore_index=True)
        ensemble_result = ensemble_result[['ticker', 'datetime', 'forecast_score']]
        
        # Return structure based on flag
        if return_base_model_predictions:
            # Combine base model predictions across tickers
            combined_base_models = {}
            for model_name, model_dfs in base_model_predictions_dict.items():
                combined_base_models[model_name] = pd.concat(model_dfs, ignore_index=True)
            
            result = {
                'ensemble': ensemble_result,
                'base_models': combined_base_models
            }
            # Store in cache
            self._predict_cache[cache_key] = {
                'ensemble': ensemble_result.copy(),
                'base_models': {k: v.copy() for k, v in combined_base_models.items()}
            }
            return result
        
        # Store in cache
        self._predict_cache[cache_key] = ensemble_result.copy()
        return ensemble_result
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        ticker: Union[pd.Series, np.ndarray],
        volatility: Union[pd.Series, np.ndarray, Dict[str, float]],
        normalization_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate per-model forecasts for use with WeightLayer.

        Returns a DataFrame with one row per (sample, base_model) combination.
        Each row contains the volatility-adjusted forecast for that model.

        The forecast is calculated as:
            F_i = (tau / (sigma * sqrt(h_i))) * X_i

        Where:
        - tau: Target annual portfolio volatility
        - sigma: Instrument's blended annualized volatility (70% EWMA-32 + 30% 10-year average)
        - h_i: Exposure fraction (1/n_bins for the base model)
        - X_i: Binary signal (0 or 1)
        
        Note: Instrument weights are applied at Portfolio layer, not here.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with raw feature values (not binary).
            Must contain all required feature columns.
        ticker : pd.Series or np.ndarray
            Ticker symbols for each sample
        volatility : pd.Series, np.ndarray, or Dict[str, float]
            Blended volatility values. Can be:
            - pd.Series: Volatility for each sample (index must align with X)
            - np.ndarray: Volatility for each sample
            - Dict[str, float]: Mapping from ticker -> volatility
        normalization_data : pd.DataFrame, optional
            Normalization data (EWSD/ATR) for each feature column.
            Columns should match feature columns in X.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - ticker: Instrument identifier
            - model_name: Base model identifier
            - forecast: Volatility-adjusted forecast (0 if signal inactive)
            - signal: Binary signal {0, 1}

            Note: One row per (sample, base_model) combination.
            Signal combination happens in WeightLayer, not here.
            
        Raises
        ------
        ValueError
            If model not fitted or input validation fails
        """
        if not self.is_fitted_:
            raise ValueError(
                "Model must be fitted before calling predict(). "
                "Call fit() first or load an ensemble model file."
            )
        
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if ticker is None or volatility is None:
            raise ValueError("ticker and volatility parameters are required for predict()")
        
        # Filter X by timeframe if base_tf is set
        if self.base_tf is not None:
            X = filter_dataframe_by_timeframe(X, self.base_tf)
            if normalization_data is not None:
                normalization_data = filter_dataframe_by_timeframe(normalization_data, self.base_tf)
        
        # Check that all required columns are present
        missing_columns = set(self.required_columns) - set(X.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required feature columns: {missing_columns}. "
                f"Required: {self.required_columns}"
            )
        
        # Filter X to only required columns
        X_filtered = X[self.required_columns].copy()
        
        # Convert ticker to Series if needed
        if isinstance(ticker, np.ndarray):
            ticker = pd.Series(ticker, index=X_filtered.index)

        # Handle volatility input types
        if isinstance(volatility, dict):
            # Dict mapping ticker -> volatility
            vol_dict = volatility
        elif isinstance(volatility, np.ndarray):
            # Array - convert to Series first, then will be handled per-sample
            volatility = pd.Series(volatility, index=X_filtered.index)
            vol_dict = None
        elif isinstance(volatility, pd.Series):
            vol_dict = None
        else:
            raise ValueError(
                f"volatility must be pd.Series, np.ndarray, or Dict[str, float], "
                f"got {type(volatility)}"
            )
        
        # Check for unseen tickers
        unseen_tickers = set(ticker.unique()) - set(self.unique_tickers_)
        if unseen_tickers:
            raise ValueError(
                f"Unseen ticker symbols in predict(): {unseen_tickers}. "
                f"Known tickers: {self.unique_tickers_}"
            )
        
        # Step 1: Get binary signals from base models
        binary_signals = {}
        for model_name, base_model in self.base_models.items():
            # Find feature column for this model
            feature_column = None
            for col, name in self.column_to_model.items():
                if name == model_name:
                    feature_column = col
                    break
            if feature_column is None:
                raise ValueError(f"Could not find feature column for model: {model_name}")
            
            # Get feature data
            feature_data = X_filtered[feature_column]
            
            # Get normalization data for this feature if provided
            norm_data = None
            if normalization_data is not None and feature_column in normalization_data.columns:
                norm_data = normalization_data[feature_column]
            
            # Get binary signals using the model's strategy
            binary_signals[model_name] = base_model.predict(
                feature_data,
                strategy=base_model.strategy,
                normalization_data=norm_data
            )
        
        # Step 2: Combine binary signals into DataFrame
        binary_df = pd.DataFrame(binary_signals, index=X_filtered.index)

        # Step 3: Calculate per-model forecasts
        # Formula: F_i = (tau / (sigma * sqrt(h_i))) * X_i
        # Note: Instrument weights are applied at Portfolio layer, not here
        results = []

        for row_idx in range(len(X_filtered)):
            tick = ticker.iloc[row_idx] if hasattr(ticker, 'iloc') else ticker[row_idx]

            # Get volatility for this sample
            if vol_dict is not None:
                vol = vol_dict.get(tick)
                if vol is None:
                    raise ValueError(f"Missing volatility for ticker: {tick}")
            else:
                vol = volatility.iloc[row_idx] if hasattr(volatility, 'iloc') else volatility[row_idx]

            # Ensure volatility is positive
            vol = max(vol, 1e-8)

            # Calculate forecast for each base model
            for model_name in self.feature_names_:
                X_i = binary_df.iloc[row_idx, binary_df.columns.get_loc(model_name)]
                signal = int(X_i)

                # Get exposure fraction from model_exposure_fractions_
                h_i = self.model_exposure_fractions_.get(model_name, 0.1)
                sqrt_h_i = np.sqrt(max(h_i, 1e-8))

                # Calculate volatility-adjusted forecast
                # This is the forecast assuming signal=1
                # Note: Instrument weights are applied at Portfolio layer, not here
                forecast_if_active = self.target_volatility_ / (vol * sqrt_h_i)
                
                # Cap forecast at 2.0 (per spec: max position is 2.0)
                forecast_if_active = min(forecast_if_active, 2.0)

                # Apply signal: 0 if inactive, forecast_if_active if active
                forecast = forecast_if_active if signal == 1 else 0.0

                results.append({
                    'ticker': tick,
                    'model_name': model_name,
                    'forecast': forecast,
                    'signal': signal
                })

        return pd.DataFrame(results)
    
    def save_config(self, filepath: Optional[str] = None) -> str:
        """
        Save fitted parameters to a JSON configuration file.
        
        Parameters
        ----------
        filepath : str, optional
            Path to save configuration. If None, uses self.save_path.
            
        Returns
        -------
        str
            Path where configuration was saved
            
        Raises
        ------
        ValueError
            If model not fitted or no filepath provided
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving configuration")
        
        if filepath is None:
            filepath = self.save_path
        
        if filepath is None:
            raise ValueError("No filepath provided and no default save_path set")
        
        # Create configuration dictionary
        config = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'model_type': 'DiversifiedEnsemble',
                'num_features': len(self.feature_names_),
                'target_volatility': self.target_volatility_,
                'num_tickers': self.n_tickers_
            },
            'weights': self.weights_,
            'exposure_fractions': self.exposure_fractions_,
            'feature_names': self.feature_names_,
            'target_volatility': self.target_volatility_,
            'unique_tickers': self.unique_tickers_,
            'instrument_weights': self.instrument_weights_,
            'n_tickers': self.n_tickers_,
            'parameters': {
                'target_volatility': self.target_volatility_
            }
        }
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        return filepath
    
    def load_config(self, filepath: str) -> 'DiversifiedEnsemble':
        """
        Load fitted parameters from a JSON configuration file.
        
        Parameters
        ----------
        filepath : str
            Path to configuration file
            
        Returns
        -------
        self
            Ensemble with loaded parameters
            
        Raises
        ------
        FileNotFoundError
            If configuration file not found
        ValueError
            If configuration file format is invalid
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        
        # Validate configuration structure
        required_keys = [
            'weights', 'exposure_fractions', 'feature_names', 'target_volatility',
            'unique_tickers', 'instrument_weights', 'n_tickers'
        ]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Configuration file missing required keys: {missing_keys}")
        
        # Load parameters
        self.weights_ = config['weights']
        self.exposure_fractions_ = config['exposure_fractions']
        self.feature_names_ = config['feature_names']
        self.target_volatility_ = config['target_volatility']
        self.unique_tickers_ = config['unique_tickers']
        self.instrument_weights_ = config['instrument_weights']
        self.n_tickers_ = config['n_tickers']
        
        # Load model parameters if available
        if 'parameters' in config:
            params = config['parameters']
            if 'target_volatility' in params:
                self.target_volatility = params['target_volatility']
        
        # Validate loaded data consistency
        if set(self.weights_.keys()) != set(self.feature_names_):
            raise ValueError("Inconsistent feature names between weights and feature_names")
        
        if set(self.exposure_fractions_.keys()) != set(self.feature_names_):
            raise ValueError("Inconsistent feature names between exposure_fractions and feature_names")
        
        if set(self.instrument_weights_.keys()) != set(self.unique_tickers_):
            raise ValueError("Inconsistent ticker names between instrument_weights and unique_tickers")
        
        # Set fitted status
        self.is_fitted_ = True
        
        return self
    
    def save_control_file(self, filepath: str) -> str:
        """
        Save ensemble to unified control file.
        
        Saves base model configs and fitted parameters (if fitted) to a single control file.
        The is_fit flag in metadata indicates whether fitted parameters are included.
        
        Parameters
        ----------
        filepath : str
            Path to save control file
            
        Returns
        -------
        str
            Path where control file was saved
            
        Raises
        ------
        ValueError
            If control_file_data not available
        """
        if self.control_file_data is None:
            raise ValueError("control_file_data not available. Cannot save control file.")
        
        from ensemble.ensemble_utils import save_control_file
        
        # Prepare metadata
        metadata = self.control_file_data.get('metadata', {}).copy()
        metadata['is_fit'] = self.is_fitted_
        if 'created_at' not in metadata:
            metadata['created_at'] = datetime.now().isoformat()
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['version'] = '2.0.0'
        if self.base_tf is not None:
            metadata['base_tf'] = self.base_tf.name
        
        # Extract fitted base model states if fitted
        fitted_base_models = None
        fitted_ensemble = None
        
        if self.is_fitted_:
            # Only save fitted params for base models that are actually fitted
            # This allows partial fitted states (some models fitted, some not)
            fitted_base_models = {}
            for model_name, base_model in self.base_models.items():
                if base_model.is_fitted_:
                    fitted_base_models[model_name] = {
                        'thresholds': base_model.thresholds_.tolist() if base_model.thresholds_ is not None else None,
                        'best_long_bin': base_model.best_long_bin_,
                        'best_short_bin': base_model.best_short_bin_,
                        'bin_stats': base_model.bin_stats_
                    }
            # Note: fitted_base_models can be empty dict if no base models are fitted
            
            fitted_ensemble = {
                'weights': self.weights_,
                'exposure_fractions': self.exposure_fractions_,
                'model_exposure_fractions': self.model_exposure_fractions_,
                'feature_names': self.feature_names_,
                'target_volatility': self.target_volatility_,
                'unique_tickers': self.unique_tickers_,
                'instrument_weights': self.instrument_weights_,
                'n_tickers': self.n_tickers_
            }
        
        # Save using utility function
        save_control_file(
            filepath=filepath,
            base_models=self.control_file_data['base_models'],
            metadata=metadata,
            fitted_base_models=fitted_base_models,
            fitted_ensemble=fitted_ensemble,
            tickers=self.control_file_data.get('tickers', [])
        )
        
        return filepath
    
    def load_control_file(self, filepath: str) -> 'DiversifiedEnsemble':
        """
        Load control file and restore all states.
        
        This is called automatically in __init__ if control_file_path is provided.
        This method is provided for reloading after initialization.
        
        Parameters
        ----------
        filepath : str
            Path to control file
            
        Returns
        -------
        self
            Ensemble with loaded parameters
            
        Raises
        ------
        FileNotFoundError
            If file not found
        ValueError
            If file format is invalid
        """
        self._initialize_from_control_file(filepath)
        return self
    
    def __repr__(self) -> str:
        """String representation of the ensemble model."""
        if self.is_fitted_:
            return (f"DiversifiedEnsemble(target_volatility={self.target_volatility_}, fitted=True, "
                   f"n_features={len(self.feature_names_)}, n_tickers={self.n_tickers_})")
        else:
            return f"DiversifiedEnsemble(target_volatility={self.target_volatility}, fitted=False)"
    
    def __str__(self) -> str:
        """Detailed string description of the ensemble model."""
        if not self.is_fitted_:
            return f"DiversifiedEnsemble (not fitted)\n  Target Volatility: {self.target_volatility}"
        
        lines = [
            f"DiversifiedEnsemble (fitted)",
            f"  Target Volatility: {self.target_volatility_}",
            f"  Features: {len(self.feature_names_)}",
            f"  Tickers: {len(self.unique_tickers_)}",
            f"  Top 3 weights: {dict(sorted(self.weights_.items(), key=lambda x: x[1], reverse=True)[:3])}"
        ]
        
        return "\n".join(lines)
    
    def _deep_copy_result(self, obj: Any) -> Any:
        """
        Recursively deep copy DataFrames in nested structures.
        
        Handles:
        - pd.DataFrame: returns .copy()
        - dict: recursively copies values
        - list: recursively copies elements
        - other: returns as-is (immutable or primitive types)
        
        Parameters
        ----------
        obj : Any
            Object to deep copy (DataFrame, dict, list, or primitive)
            
        Returns
        -------
        Any
            Deep copied object with all DataFrames copied
        """
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        elif isinstance(obj, dict):
            return {k: self._deep_copy_result(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy_result(item) for item in obj]
        else:
            return obj
    
    def clear_cache(self) -> None:
        """Clear all cached fit and predict results."""
        self._fit_cache.clear()
        self._predict_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        # Also clear cache in base models
        for base_model in self.base_models.values():
            if hasattr(base_model, 'clear_cache'):
                base_model.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for this ensemble and all base models.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with cache hits, misses, hit rate, and cache sizes
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0.0
        
        base_model_stats = {}
        for model_name, base_model in self.base_models.items():
            if hasattr(base_model, 'get_cache_stats'):
                base_model_stats[model_name] = base_model.get_cache_stats()
        
        return {
            'ensemble': {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'total': total,
                'hit_rate': hit_rate,
                'fit_cache_size': len(self._fit_cache),
                'predict_cache_size': len(self._predict_cache)
            },
            'base_models': base_model_stats
        }
