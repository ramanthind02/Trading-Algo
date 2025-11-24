"""
Diversified Ensemble Class for Strategy Combination

This module implements a diversified ensemble class that combines multiple
binary trading strategies using intra-feature correlation for diversified weights
and applies risk-adjusted position sizing based on volatility and exposure fractions.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Union, Dict

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
        save_path: Optional[str] = None
    ):
        self.target_volatility = target_volatility
        self.instrument_weights = instrument_weights
        self.save_path = save_path
        
        # Fitted parameters (set during fit() or load_config())
        self.weights_ = None
        self.exposure_fractions_ = None
        self.feature_names_ = None
        self.target_volatility_ = None
        self.unique_tickers_ = None
        self.instrument_weights_ = None
        self.n_tickers_ = None
        self.is_fitted_ = False
        
        # Load configuration if provided
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
        instrument_weights: Optional[Dict[str, float]] = None
    ) -> 'DiversifiedEnsemble':
        """
        Fit the ensemble model to training data.
        
        Calculates diversified weights based on intra-feature correlations,
        computes exposure fractions, and sets up instrument weighting.
        The target y is required for API consistency but ignored in this implementation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with binary (0/1) trading signals.
            Should not contain 'annualized_volatility' column.
        ticker : pd.Series or np.ndarray
            Ticker symbols for each sample
        volatility : pd.Series or np.ndarray
            Volatility values for each sample (positive values)
        y : pd.Series or np.ndarray
            Target values (required but ignored in this implementation)
        instrument_weights : dict, optional
            Custom instrument weights mapping ticker -> weight
            
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
        X, ticker, volatility, y = self._validate_input_data(
            X, ticker, volatility, y, method="fit"
        )
        
        if ticker is None or volatility is None:
            raise ValueError("ticker and volatility parameters are required for fit()")
        
        if y is None:
            raise ValueError("y parameter is required for fit() (even though it's ignored)")
        
        # Get feature columns
        feature_cols = list(X.columns)
        
        # Align data and drop NaN values
        df = pd.DataFrame({
            **{col: X[col] for col in feature_cols},
            'ticker': ticker,
            'volatility': volatility,
            'target': y
        }).dropna()
        
        if len(df) < 10:
            raise ValueError(f"Insufficient data after dropping NaN. Need at least 10 samples, got {len(df)}")
        
        # Calculate diversified weights (ignoring target y)
        self.weights_ = self._calculate_diversified_weights(df[feature_cols])
        
        # Calculate exposure fractions (h_i)
        self.exposure_fractions_ = {}
        for col in feature_cols:
            feature_data = df[col]
            # Fraction of time feature == 1
            self.exposure_fractions_[col] = feature_data.mean()
        
        # Store feature names
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
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        ticker: Union[pd.Series, np.ndarray],
        volatility: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Generate ensemble predictions using fitted parameters.
        
        Applies the ensemble formula: Σ((τ × w_i) / (σ_i × √h_i)) × instrument_weight
        for each active feature, then multiplies by instrument weight.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with binary values.
            Must contain the same feature columns as used in fit().
        ticker : pd.Series or np.ndarray
            Ticker symbols for each sample
        volatility : pd.Series or np.ndarray  
            Volatility values for each sample (positive values)
            
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
        X, ticker, volatility, _ = self._validate_input_data(
            X, ticker, volatility, method="predict"
        )
        
        if ticker is None or volatility is None:
            raise ValueError("ticker and volatility parameters are required for predict()")
        
        # Check that all required feature columns are present
        missing_features = set(self.feature_names_) - set(X.columns)
        if missing_features:
            raise ValueError(
                f"Missing feature columns: {missing_features}. "
                f"Expected: {self.feature_names_}"
            )
        
        # Check for unseen tickers (raise error as requested)
        unseen_tickers = set(ticker.unique()) - set(self.unique_tickers_)
        if unseen_tickers:
            raise ValueError(
                f"Unseen ticker symbols in predict(): {unseen_tickers}. "
                f"Known tickers: {self.unique_tickers_}"
            )
        
        # Extract feature data
        feature_data = X[self.feature_names_]
        
        # Apply ensemble formula: Σ((τ × w_i) / (σ_i × √h_i)) × instrument_weight
        predictions = np.zeros(len(X))
        
        for row_idx in range(len(X)):
            row_forecast = 0.0
            vol = volatility.iloc[row_idx] if hasattr(volatility, 'iloc') else volatility[row_idx]
            tick = ticker.iloc[row_idx] if hasattr(ticker, 'iloc') else ticker[row_idx]
            
            # Calculate sum of active features
            for col in self.feature_names_:
                X_i = feature_data.iloc[row_idx, feature_data.columns.get_loc(col)]
                
                if X_i == 1:  # Feature is active
                    w_i = self.weights_[col]
                    h_i = self.exposure_fractions_[col]
                    
                    # Avoid division by zero for h_i
                    sqrt_h_i = np.sqrt(max(h_i, 1e-8))
                    
                    # Calculate percent forecast for this feature
                    percent_forecast_i = (self.target_volatility_ * w_i) / (vol * sqrt_h_i)
                    row_forecast += percent_forecast_i
            
            # Apply instrument weight
            instrument_weight = self.instrument_weights_[tick]
            predictions[row_idx] = row_forecast * instrument_weight
        
        return predictions
    
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
