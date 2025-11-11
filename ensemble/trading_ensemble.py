"""
Trading Ensemble Class for Strategy Combination

This module implements a trading-specific ensemble class that combines multiple
binary trading strategies using correlation coefficients as weights and applies
risk-adjusted position sizing based on volatility and exposure fractions.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Optional, Union
from datetime import datetime
from utils.enums import Ticker, TimeFrame, Direction, Style


class TradingEnsemble:
    """
    Trading-specific ensemble class for combining binary strategy signals.
    
    This ensemble class is designed for trading strategies where:
    - Features are binary (0 or 1) indicating trade signals
    - Target is continuous log returns
    - Weights are correlation coefficients normalized to sum to 1
    - Final prediction includes risk scaling and exposure adjustment
    
    Parameters
    ----------
    ticker : Ticker
        Market ticker symbol from utils.enums.Ticker
    timeframe : TimeFrame
        Trading timeframe from utils.enums.TimeFrame (D, W, M)
    direction : Direction
        Trading direction from utils.enums.Direction (long_only, long_and_short, short_only)
    style : Style
        Trading style from utils.enums.Style (mean_reversion, momentum)
    r : float, default=0.15
        Target annual portfolio risk (e.g., 0.15 = 15% annual risk)
    config_path : str, optional
        Path to saved configuration file for loading pre-fitted parameters
        
    Attributes
    ----------
    ticker : Ticker
        Market ticker symbol
    timeframe : TimeFrame
        Trading timeframe
    direction : Direction
        Trading direction
    style : Style
        Trading style
    weights_ : dict
        Feature name -> correlation coefficient mapping after fitting
    exposure_fractions_ : dict
        Feature name -> exposure fraction mapping (fraction of time in market)
    feature_names_ : list
        List of feature column names (excluding 'annualized_volatility')
    is_fitted_ : bool
        Whether the model has been fitted
    r : float
        Target annual portfolio risk
    """
    
    def __init__(
        self,
        ticker: Optional[Ticker] = None,
        timeframe: Optional[TimeFrame] = None,
        direction: Optional[Direction] = None,
        style: Optional[Style] = None,
        r: float = 0.15,
        config_path: Optional[str] = None
    ):
        # Initialize metadata fields
        self.ticker = ticker
        self.timeframe = timeframe
        self.direction = direction
        self.style = style
        self.r = r
        
        # Fitted parameters (set during fit() or load_config())
        self.weights_ = None
        self.exposure_fractions_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        
        # Load configuration if provided
        if config_path is not None:
            self.load_config(config_path)
        elif any(param is None for param in [ticker, timeframe, direction, style]):
            # If no config_path and required parameters are missing, raise error
            raise ValueError(
                "Either provide config_path for loading saved configuration, "
                "or provide all required parameters: ticker, timeframe, direction, style"
            )
    
    def _validate_input_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        method: str = "fit"
    ) -> pd.DataFrame:
        """
        Validate input data format and requirements.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix with 'annualized_volatility' column
        y : pd.Series or np.ndarray, optional
            Target values (required for fit, not for predict)
        method : str
            Method name for error messages ('fit' or 'predict')
            
        Returns
        -------
        pd.DataFrame
            Validated DataFrame with proper column names
            
        Raises
        ------
        ValueError
            If validation fails
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            raise ValueError(
                f"{method}() requires DataFrame input with column names. "
                f"'annualized_volatility' column must be identifiable."
            )
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"{method}() requires pandas DataFrame input")
        
        # Check for required volatility column
        if 'annualized_volatility' not in X.columns:
            raise ValueError(
                f"Input DataFrame must contain 'annualized_volatility' column. "
                f"Found columns: {list(X.columns)}"
            )
        
        # Validate volatility column is continuous
        vol_col = X['annualized_volatility']
        if vol_col.dtype not in ['float64', 'float32', 'int64', 'int32']:
            raise ValueError(
                f"'annualized_volatility' column must be numeric. "
                f"Found dtype: {vol_col.dtype}"
            )
        
        # Check for zero or negative volatility
        if (vol_col <= 0).any():
            raise ValueError(
                f"'annualized_volatility' column contains non-positive values. "
                f"All volatility values must be positive."
            )
        
        # Get feature columns (all except volatility)
        feature_cols = [col for col in X.columns if col != 'annualized_volatility']
        
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found (only 'annualized_volatility')")
        
        # Validate feature columns are binary for fit method
        if method == "fit":
            for col in feature_cols:
                unique_vals = X[col].dropna().unique()
                if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    raise ValueError(
                        f"Feature column '{col}' must be binary (0 or 1). "
                        f"Found unique values: {sorted(unique_vals)}"
                    )
        
        # Validate target if provided
        if y is not None:
            if isinstance(y, np.ndarray):
                y = pd.Series(y, index=X.index)
            elif not isinstance(y, pd.Series):
                raise ValueError("Target y must be pandas Series or numpy array")
            
            # Check target is continuous
            if y.dtype not in ['float64', 'float32']:
                raise ValueError(
                    f"Target y must be continuous (float). Found dtype: {y.dtype}"
                )
            
            # Align indices
            common_idx = X.index.intersection(y.index)
            if len(common_idx) < len(X) * 0.9:  # Allow 10% missing data
                raise ValueError(
                    f"Insufficient overlap between X and y indices. "
                    f"X: {len(X)}, y: {len(y)}, common: {len(common_idx)}"
                )
        
        return X
    
    def _generate_config_filename(self) -> str:
        """
        Generate config filename based on metadata fields.
        
        Returns
        -------
        str
            Generated filename in format: data/ensembles/{model_type}/{ticker}_{timeframe}_{direction}_{style}_{timestamp}.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.ticker.name}_{self.timeframe.name}_{self.direction.value}_{self.style.value}_{timestamp}.json"
        
        # Ensure data/ensembles/TradingEnsemble directory exists
        config_dir = os.path.join("data", "ensembles_staging", "TradingEnsemble")
        os.makedirs(config_dir, exist_ok=True)
        
        return os.path.join(config_dir, filename)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'TradingEnsemble':
        """
        Fit the ensemble model to training data.
        
        Calculates correlation coefficients between each feature and target,
        normalizes them to sum to 1, and computes exposure fractions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with 'annualized_volatility' column.
            All other columns should be binary (0/1) trading signals.
        y : pd.Series or np.ndarray
            Target values (continuous log returns)
            
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
        X = self._validate_input_data(X, y, method="fit")
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)
        
        # Get feature columns (excluding volatility)
        feature_cols = [col for col in X.columns if col != 'annualized_volatility']
        
        # Align data and drop NaN values
        df = pd.DataFrame({
            **{col: X[col] for col in feature_cols},
            'target': y
        }).dropna()
        
        if len(df) < 10:
            raise ValueError(f"Insufficient data after dropping NaN. Need at least 10 samples, got {len(df)}")
        
        # Calculate correlation coefficients
        correlations = {}
        for col in feature_cols:
            corr = df[col].corr(df['target'])
            if pd.isna(corr):  # Handle zero-variance features
                corr = 0.0
            correlations[col] = corr
        
        # Use absolute correlations for weights
        abs_correlations = {col: abs(corr) for col, corr in correlations.items()}
        
        # Normalize weights to sum to 1
        total_abs_corr = sum(abs_correlations.values())
        if total_abs_corr == 0:
            # If all correlations are zero, use equal weights
            self.weights_ = {col: 1.0 / len(feature_cols) for col in feature_cols}
        else:
            self.weights_ = {col: abs_corr / total_abs_corr 
                            for col, abs_corr in abs_correlations.items()}
        
        # Calculate exposure fractions (h_i)
        self.exposure_fractions_ = {}
        for col in feature_cols:
            feature_data = df[col]
            # Fraction of time feature == 1
            self.exposure_fractions_[col] = feature_data.mean()
        
        # Store feature names and set fitted status
        self.feature_names_ = feature_cols
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate ensemble predictions using fitted parameters.
        
        Applies the ensemble formula: r/v * Σ(w_i * X_i / √h_i)
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with 'annualized_volatility' column.
            Must contain the same feature columns as used in fit().
            
        Returns
        -------
        np.ndarray
            Array of ensemble predictions (position sizes)
            
        Raises
        ------
        ValueError
            If model not fitted or input validation fails
        """
        if not self.is_fitted_:
            raise ValueError(
                "Model must be fitted before calling predict(). "
                "Call fit() first or load a configuration file."
            )
        
        # Validate inputs
        X = self._validate_input_data(X, method="predict")
        
        # Check that all required feature columns are present
        missing_features = set(self.feature_names_) - set(X.columns)
        if missing_features:
            raise ValueError(
                f"Missing feature columns: {missing_features}. "
                f"Expected: {self.feature_names_}"
            )
        
        # Extract volatility and feature data
        volatility = X['annualized_volatility'].values
        feature_data = X[self.feature_names_]
        
        # Apply ensemble formula: r/v * Σ(w_i * X_i / √h_i)
        predictions = np.zeros(len(X))
        
        for col in self.feature_names_:
            w_i = self.weights_[col]
            h_i = self.exposure_fractions_[col]
            X_i = feature_data[col].values
            
            # Avoid division by zero for h_i
            sqrt_h_i = np.sqrt(max(h_i, 1e-8))
            
            # Add weighted contribution
            predictions += w_i * X_i / sqrt_h_i
        
        # Scale by risk target and volatility
        # Avoid division by zero for volatility
        safe_volatility = np.maximum(volatility, 1e-8)
        predictions = (self.r / safe_volatility) * predictions
        
        return predictions
    
    def save_config(self, filepath: Optional[str] = None) -> str:
        """
        Save fitted parameters to a JSON configuration file.
        
        If no filepath is provided, generates filename automatically based on metadata:
        {ticker}_{timeframe}_{direction}_{style}_ensemble_{timestamp}.json
        
        Parameters
        ----------
        filepath : str, optional
            Path to save configuration. If None, auto-generates filename.
            
        Returns
        -------
        str
            Path where configuration was saved
            
        Raises
        ------
        ValueError
            If model not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving configuration")
        
        if filepath is None:
            filepath = self._generate_config_filename()
        
        # Create configuration dictionary
        config = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'model_type': 'TradingEnsemble',
                'ticker': self.ticker.name,
                'timeframe': self.timeframe.name,
                'direction': self.direction.value,
                'style': self.style.value,
                'num_features': len(self.feature_names_),
                'target_risk': self.r
            },
            'weights': self.weights_,
            'exposure_fractions': self.exposure_fractions_,
            'feature_names': self.feature_names_,
            'parameters': {
                'ticker': self.ticker.name,
                'timeframe': self.timeframe.name,
                'direction': self.direction.value,
                'style': self.style.value,
                'r': self.r
            }
        }
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        return filepath
    
    def load_config(self, filepath: str) -> 'TradingEnsemble':
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
        required_keys = ['weights', 'exposure_fractions', 'feature_names']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Configuration file missing required keys: {missing_keys}")
        
        # Load parameters
        self.weights_ = config['weights']
        self.exposure_fractions_ = config['exposure_fractions']
        self.feature_names_ = config['feature_names']
        
        # Load model parameters if available
        if 'parameters' in config:
            params = config['parameters']
            if 'r' in params:
                self.r = params['r']
            # Load metadata fields if available
            if 'ticker' in params:
                self.ticker = Ticker[params['ticker']]
            if 'timeframe' in params:
                self.timeframe = TimeFrame[params['timeframe']]
            if 'direction' in params:
                self.direction = Direction(params['direction'])
            if 'style' in params:
                self.style = Style(params['style'])
        
        # Validate loaded data consistency
        if set(self.weights_.keys()) != set(self.feature_names_):
            raise ValueError("Inconsistent feature names between weights and feature_names")
        
        if set(self.exposure_fractions_.keys()) != set(self.feature_names_):
            raise ValueError("Inconsistent feature names between exposure_fractions and feature_names")
        
        # Set fitted status
        self.is_fitted_ = True
        
        return self
    
    def __repr__(self) -> str:
        """String representation of the ensemble model."""
        if self.is_fitted_:
            return (f"TradingEnsemble(r={self.r}, fitted=True, "
                   f"n_features={len(self.feature_names_)})")
        else:
            return f"TradingEnsemble(r={self.r}, fitted=False)"
    
    def __str__(self) -> str:
        """Detailed string description of the ensemble model."""
        if not self.is_fitted_:
            return f"TradingEnsemble (not fitted)\n  Target Risk: {self.r}"
        
        lines = [
            f"TradingEnsemble (fitted)",
            f"  Target Risk: {self.r}",
            f"  Features: {len(self.feature_names_)}",
            f"  Top 3 weights: {dict(sorted(self.weights_.items(), key=lambda x: x[1], reverse=True)[:3])}"
        ]
        
        return "\n".join(lines)
