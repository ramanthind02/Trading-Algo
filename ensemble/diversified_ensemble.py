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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import utils.core.helpers as helpers
from utils.compute.daily_ewsd_volatility import (
    align_daily_ewsd_volatility_to_candles,
)
from utils.core.enums import TimeFrame, Ticker
from utils.core.models import Candle
from .ensemble_utils import (
    filter_dataframe_by_timeframe,
    normalize_candles_datetime_column,
    normalize_ticker_key,
)

logger = logging.getLogger(__name__)


def _normalize_ticker_name(ticker_val: object) -> str:
    """Normalize ticker identifiers; delegate to shared helper."""
    return normalize_ticker_key(ticker_val)


def _normalize_candles_datetime_column(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime is only a column; delegate to shared helper."""
    return normalize_candles_datetime_column(candles_df)


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
        base_models: Optional[Dict[str, Any]] = None,
        required_columns: Optional[List[str]] = None,
        base_tf: Optional[TimeFrame] = None,
        use_cache: bool = True,
    ):
        self.target_volatility = target_volatility
        self.instrument_weights = instrument_weights
        self.save_path = save_path
        self.use_cache = use_cache
        self.retry_on_cache_miss = True
        
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
        
        # Initialize from control file or programmatically from base_models + required_columns
        if control_file_path is not None:
            self._initialize_from_control_file(control_file_path)
            if base_tf is not None:
                self.base_tf = base_tf
        else:
            # Programmatic path: require base_models and required_columns
            if base_models is None or required_columns is None:
                raise ValueError(
                    "Either control_file_path or both base_models and required_columns must be provided."
                )
            self.base_models = base_models
            self.required_columns = list(required_columns)
            for model_name, bm in self.base_models.items():
                feature_column = getattr(bm, "feature_column", None)
                if not feature_column:
                    raise ValueError(
                        f"Base model '{model_name}' missing feature_column in single-feature mode."
                    )
                self.column_to_model[feature_column] = model_name
            if base_tf is None:
                raise ValueError("base_tf is required when not using control_file_path.")
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
        
        # Validate feature columns are binary (0/1) or signed (-1/0/1) for fit method
        if method == "fit":
            for col in feature_cols:
                unique_vals = X[col].dropna().unique()
                if not set(unique_vals).issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
                    raise ValueError(
                        f"Feature column '{col}' must be binary (0 or 1) or signed (-1, 0, 1). "
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
        
        # Initialize base models
        self.base_models = {}
        self.column_to_model = {}
        self.required_columns = []
        
        for model_config in control_file['base_models']:
            model_name = model_config['name']
            feature_column = model_config['feature_column']
            model_config_for_create = model_config.copy()
            if 'tickers' not in model_config_for_create and 'tickers' in control_file:
                model_config_for_create['tickers'] = control_file.get('tickers', [])
            base_model = create_base_model_from_config(
                model_config_for_create,
                use_cache=self.use_cache
            )
            
            # Store model
            self.base_models[model_name] = base_model
            self.column_to_model[feature_column] = model_name
            self.required_columns.append(feature_column)
    
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
            Normalization data (EWSD) for each feature column.
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
            
            if feature_column not in X_filtered.columns:
                raise ValueError(
                    f"Required feature column '{feature_column}' not found in input data."
                )

            signal_series = X_filtered[feature_column].astype(float)
            if not set(signal_series.dropna().unique()).issubset({-1.0, 0.0, 1.0}):
                raise ValueError(
                    f"Feature column '{feature_column}' must contain signed discrete signals."
                )
            binary_signals[model_name] = signal_series
        
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
        
        # Calculate exposure fractions (h_i) - empirical
        # For long/short: fraction of time in market (signal != 0). For long-only: fraction signal==1.
        self.exposure_fractions_ = {}
        for col in feature_cols:
            feature_data = df[col]
            if set(feature_data.dropna().unique()).issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
                # Signed signals: exposure = fraction of time in market (long or short)
                in_market = (feature_data != 0) & (feature_data.notna())
                self.exposure_fractions_[col] = in_market.astype(float).mean()
            else:
                # Binary long-only: fraction of time signal == 1
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
                # Calculate from actual signals: long_short = fraction in market; long-only = fraction signal=1
                if model_name in binary_df.columns:
                    sig = binary_df[model_name]
                    if set(sig.dropna().unique()).issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
                        self.model_exposure_fractions_[model_name] = (sig != 0).astype(float).mean()
                    else:
                        self.model_exposure_fractions_[model_name] = sig.mean()
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
    
    def fit_from_candles(
        self,
        candles_df: pd.DataFrame,
        target_data: pd.Series,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
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
        start_date : datetime, optional
            Start date for cached data. If None, inferred from candles_df.
        end_date : datetime, optional
            End date for cached data. If None, inferred from candles_df.

        Returns
        -------
        self
            Fitted ensemble model
        """
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
        at_least_one_fit_viable = False
        for model_name, base_model in self.base_models.items():
            try:
                bias_node_spec = getattr(base_model, "bias_node_spec", None)
                if isinstance(bias_node_spec, dict) and bias_node_spec.get("module_name") == "buy_hold":
                    logger.debug("Skipping fit for buy_hold model '%s'", model_name)
                    at_least_one_fit_viable = True
                    continue

                # Fit base model with filtered candles and aggregated returns
                # BaseModel.fit() will:
                # 1. Stream candles from all supported tickers (or use cache if use_cache=True)
                # 2. Extract features and aggregate by base datetime (mean)
                # 3. Align aggregated returns with aggregated features
                try:
                    base_model.fit(filtered_candles, aggregated_returns, start_date, end_date)
                except Exception as exc:
                    cache_miss = "Cache miss" in str(exc)
                    if (
                        self.retry_on_cache_miss
                        and getattr(base_model, "use_cache", False)
                        and cache_miss
                    ):
                        logger.warning(
                            "Cache miss while fitting model '%s'; retrying with use_cache=False",
                            model_name,
                        )
                        base_model.use_cache = False
                        base_model.fit(filtered_candles, aggregated_returns, start_date, end_date)
                    else:
                        raise
                at_least_one_fit_viable = True
                logger.debug("Model '%s' prepared from frozen domain-discrete spec", model_name)
            except Exception as e:
                logger.error(
                    f"Error fitting base model '{model_name}': {e}",
                    exc_info=True
                )

        if not at_least_one_fit_viable:
            raise RuntimeError(
                "Ensemble fitting failed: no base model could be fitted or was already fitted. "
                "Check logs for per-model errors."
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
                                # Generate binary signals from base model using its configured strategy
                                pred = base_model.predict(
                                    ticker_candles,
                                    strategy=base_model.strategy
                                )
                                if pred is not None and len(pred) > 0:
                                    model_signals.append(pred)
                            except Exception as e:
                                logger.warning(
                                    f"Error generating signals for model '{model_name}' on ticker '{ticker_name}': {e}"
                                )
                    
                    # Calculate exposure fraction from actual signals
                    if model_signals:
                        all_signals = pd.concat(model_signals) if len(model_signals) > 1 else model_signals[0]
                        if set(all_signals.dropna().unique()).issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
                            self.model_exposure_fractions_[model_name] = (all_signals != 0).astype(float).mean()
                        else:
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

        # Only mark fitted when at least one model is viable (already ensured by at_least_one_fit_viable above)
        self.is_fitted_ = True

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
        
        # Ensure datetime is only a column (not also an index level) to avoid ambiguous sort_values
        candles_df = _normalize_candles_datetime_column(candles_df)
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
    
    def predict_from_candles(
        self,
        candles_df: pd.DataFrame,
        daily_volatility_df: pd.DataFrame,
        return_base_model_predictions: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
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
        daily_volatility_df : pd.DataFrame
            Daily EWSD volatility with required columns:
            ['datetime', 'ticker', 'ewsd_annual_vol'].
        return_base_model_predictions : bool, default=False
            If True, return base model-level predictions in result dict
        start_date : datetime, optional
            Start date for cached data. If None, inferred from candles_df.
        end_date : datetime, optional
            End date for cached data. If None, inferred from candles_df.

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

        if daily_volatility_df is None:
            raise ValueError(
                "daily_volatility_df is required for predict_from_candles(). "
                "Expected columns: ['datetime', 'ticker', 'ewsd_annual_vol']."
            )
        
        # Ensure 'datetime' is only a column (not also an index level) so merge/groupby are unambiguous
        candles_df = _normalize_candles_datetime_column(candles_df)

        aligned_volatility = align_daily_ewsd_volatility_to_candles(
            daily_volatility_df=daily_volatility_df,
            candles_df=candles_df,
        )
        aligned_volatility["datetime"] = pd.to_datetime(aligned_volatility["datetime"]).dt.floor("s")
        aligned_volatility["ticker"] = aligned_volatility["ticker"].map(_normalize_ticker_name)
        vol_lookup = (
            aligned_volatility
            .drop_duplicates(subset=["ticker", "datetime"], keep="last")
            .set_index(["ticker", "datetime"])["ewsd_annual_vol"]
        )
        
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
            
            # Get predictions from each base model (BaseModel.predict() handles caching internally)
            ticker_predictions = []
            for model_name, base_model in self.base_models.items():
                try:
                    # Only use this base model for tickers it explicitly supports.
                    # If tickers is empty, skip (do not assume "all tickers"); per-ticker
                    # weights are correct only when each model is used only for its tickers.
                    model_tickers = set(getattr(base_model, 'tickers', []))
                    if not model_tickers:
                        logger.debug(
                            f"Base model '{model_name}' has no tickers configured; skipping for ticker '{ticker_name}'."
                        )
                        continue
                    model_ticker_names = {_normalize_ticker_name(t) for t in model_tickers}
                    normalized_ticker_name = _normalize_ticker_name(ticker_name)
                    if normalized_ticker_name not in model_ticker_names:
                        logger.debug(
                            f"Base model '{model_name}' does not support ticker '{ticker_name}'. "
                            f"Supported tickers: {model_ticker_names}. Skipping."
                        )
                        continue

                    bias_node_spec = getattr(base_model, "bias_node_spec", None)
                    is_buy_hold_model = (
                        isinstance(bias_node_spec, dict)
                        and bias_node_spec.get("module_name") == "buy_hold"
                    )

                    # BaseModel.predict() returns a Series indexed by datetime with binary signals
                    # BaseModel.predict() handles feature caching internally (if use_cache=True)
                    if is_buy_hold_model:
                        pred = pd.Series(
                            np.ones(len(ticker_candles), dtype=float),
                            index=pd.to_datetime(ticker_candles["datetime"]),
                        )
                    else:
                        try:
                            pred = base_model.predict(
                                ticker_candles,
                                strategy=base_model.strategy,
                                start_date=start_date,
                                end_date=end_date
                            )
                        except Exception as exc:
                            cache_miss = "No cached features available" in str(exc) or "Cache miss" in str(exc)
                            if (
                                self.retry_on_cache_miss
                                and getattr(base_model, "use_cache", False)
                                and cache_miss
                            ):
                                logger.warning(
                                    "Cache miss while predicting model '%s' for ticker '%s'; retrying with use_cache=False",
                                    model_name,
                                    ticker_name,
                                )
                                base_model.use_cache = False
                                pred = base_model.predict(
                                    ticker_candles,
                                    strategy=base_model.strategy,
                                    start_date=start_date,
                                    end_date=end_date
                                )
                            else:
                                raise
                    # Debug: Log prediction details
                    if len(pred) == 0:
                        logger.warning(
                            f"Base model '{model_name}' returned empty predictions for ticker '{ticker_name}'. "
                            f"Input candles: {len(ticker_candles)}"
                        )
                        continue
                    
                    pred_datetimes = pd.to_datetime(pred.index).floor("s")
                    vol_keys = pd.MultiIndex.from_arrays(
                        [
                            np.repeat(_normalize_ticker_name(ticker_name), len(pred_datetimes)),
                            pred_datetimes,
                        ]
                    )
                    aligned_pred_vol = vol_lookup.reindex(vol_keys)
                    if aligned_pred_vol.isna().any():
                        missing_dt = pred_datetimes[int(np.where(aligned_pred_vol.isna().to_numpy())[0][0])]
                        raise ValueError(
                            "Missing aligned daily EWSD volatility for "
                            f"ticker '{ticker_name}' at datetime {missing_dt}."
                        )

                    vol_arr = np.maximum(aligned_pred_vol.to_numpy(dtype=float), 1e-8)

                    # Apply volatility scaling per base model
                    # Formula: F_i = (tau / sigma) * X_i
                    # Direct Carver vol-targeting: when the signal fires, size to hit the
                    # annualised vol target (tau) given current instrument vol (sigma).
                    # The sqrt(h_i) amplification is NOT applied here: it assumed every
                    # bin of a binning model fires independently h_i of the time, but for
                    # a single rule-based binary signal the correct scaling is h_i = 1.
                    # Instrument weights are applied at the Portfolio layer, not here.
                    forecast_if_active = self.target_volatility_ / vol_arr
                    if is_buy_hold_model:
                        if self.unique_tickers_ is not None and len(self.unique_tickers_) > 1:
                            forecast_if_active = 1.0
                        elif np.all(np.abs(forecast_if_active - 1.0) < 0.25):
                            # Stabilize around the analytical buy-hold case where tau ~= sigma.
                            forecast_if_active = 1.0
                    
                    
                    # Cap forecast at 2.0 (per spec: max position is 2.0)
                    forecast_if_active = np.minimum(forecast_if_active, 2.0)
                    
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
        
        # CRITICAL FIX: Align predictions to candles' datetime index to ensure all tickers have same rows
        # This fixes the issue where different tickers have different prediction counts (e.g. ES:1008, NQ:1007)
        # causing the merge in portfolio._apply_risk_management_to_forecasts to fail
        candles_datetime_index = candles_df.reset_index(drop=True)[['ticker', 'datetime']].copy()
        candles_datetime_index['datetime'] = pd.to_datetime(candles_datetime_index['datetime']).dt.floor('s')
        ensemble_result['datetime'] = pd.to_datetime(ensemble_result['datetime']).dt.floor('s')
        
        # Right merge: keep all candles datetimes, fill missing forecasts with 0
        ensemble_result = candles_datetime_index.merge(
            ensemble_result,
            on=['ticker', 'datetime'],
            how='left'
        )
        ensemble_result['forecast_score'] = ensemble_result['forecast_score'].fillna(0.0)
        ensemble_result = ensemble_result[['ticker', 'datetime', 'forecast_score']]
        
        # Return structure based on flag
        if return_base_model_predictions:
            # Combine base model predictions across tickers (only tickers this model supports).
            # Do NOT merge with full candles_datetime_index: that would add rows for unsupported
            # tickers (e.g. TLT for indices models) with 0, and the weight layer would then
            # count that model for every ticker. Align only to candles for this model's tickers.
            combined_base_models: Dict[str, pd.DataFrame] = {}
            for model_name, model_dfs in base_model_predictions_dict.items():
                base_model_df = pd.concat(model_dfs, ignore_index=True)
                base_model_df['datetime'] = pd.to_datetime(base_model_df['datetime']).dt.floor('s')
                tickers_in_model = base_model_df['ticker'].unique()
                candles_subset = candles_datetime_index[
                    candles_datetime_index['ticker'].isin(tickers_in_model)
                ]
                base_model_df = candles_subset.merge(
                    base_model_df,
                    on=['ticker', 'datetime'],
                    how='left'
                )
                base_model_df['forecast_score'] = base_model_df['forecast_score'].fillna(0.0)
                base_model_df = base_model_df[['ticker', 'datetime', 'forecast_score']]
                combined_base_models[model_name] = base_model_df

            return {
                'ensemble': ensemble_result,
                'base_models': combined_base_models,
            }

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
            F_i = (tau / sigma) * X_i   [capped at 2.0]

        Where:
        - tau: Target annual portfolio volatility
        - sigma: Instrument's blended annualized volatility (70% EWMA-32 + 30% 10-year average)
        - X_i: Binary signal (0 or 1)

        h_i is not applied here. The sqrt(h_i) term was designed for bin-based ensembles
        where n independent bins each fire 1/n of the time; for a rule-based binary signal
        (all-in or all-out) the correct scaling is h_i = 1, giving F = tau/sigma directly.
        The cap at 2.0 limits leverage to 2x when sigma < tau/2.

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
            Normalization data (EWSD) for each feature column.
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
            
            if feature_column not in X_filtered.columns:
                raise ValueError(
                    f"Required feature column '{feature_column}' not found in input data."
                )

            signal_series = X_filtered[feature_column].astype(float)
            if not set(signal_series.dropna().unique()).issubset({-1.0, 0.0, 1.0}):
                raise ValueError(
                    f"Feature column '{feature_column}' must contain signed discrete signals."
                )
            binary_signals[model_name] = signal_series
        
        # Step 2: Combine binary signals into DataFrame
        binary_df = pd.DataFrame(binary_signals, index=X_filtered.index)

        # Step 3: Calculate per-model forecasts (vectorized)
        # Formula: F_i = (tau / sigma) * X_i  [h_i = 1; see predict docstring]
        # Note: Instrument weights are applied at Portfolio layer, not here
        n_rows = len(X_filtered)
        model_names = self.feature_names_

        ticker_vals = ticker.values if hasattr(ticker, 'values') else np.asarray(ticker)
        if vol_dict is not None:
            vol_arr = np.array([vol_dict.get(t) for t in ticker_vals], dtype=float)
            if np.any(np.isnan(vol_arr)):
                missing = ticker_vals[np.where(np.isnan(vol_arr))[0][0]]
                raise ValueError(f"Missing volatility for ticker: {missing}")
        else:
            vol_arr = np.asarray(volatility, dtype=float).reshape(-1)
            if vol_arr.size != n_rows:
                vol_arr = vol_arr[:n_rows] if vol_arr.size >= n_rows else np.resize(vol_arr, n_rows)
        vol_arr = np.maximum(vol_arr, 1e-8)

        # Direct Carver vol-targeting (h_i = 1): F = tau / sigma.
        # See predict_with_candles for rationale on dropping sqrt(h_i).
        # (n_rows, n_models): forecast if signal=1, then cap at 2.0
        forecast_if_active = self.target_volatility_ / vol_arr[:, np.newaxis]
        forecast_if_active = np.minimum(forecast_if_active, 2.0)

        signals = binary_df[model_names].astype(float).fillna(0).values
        forecasts = forecast_if_active * signals

        ticker_flat = np.repeat(ticker_vals, len(model_names))
        model_flat = np.tile(model_names, n_rows)
        forecast_flat = forecasts.ravel()
        signal_flat = signals.ravel().astype(int)

        return pd.DataFrame({
            'ticker': ticker_flat,
            'model_name': model_flat,
            'forecast': forecast_flat,
            'signal': signal_flat,
        })
    
    def save_config(self, filepath: Optional[str] = None) -> str:
        """
        Save fitted parameters to a JSON configuration file.

        .. note::
            Legacy API. Prefer :meth:`save_control_file` for full ensemble state
            (base models + fitted params) in the unified control-file schema.
        
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

        .. note::
            Legacy API. Prefer loading from a control file via the constructor
            (control_file_path=...) or :meth:`load_control_file` for full state.
        
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
        if self.is_fitted_ and 'selection_method' not in metadata:
            metadata['selection_method'] = 'manual'

        fitted_ensemble = None
        
        if self.is_fitted_:
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
    
