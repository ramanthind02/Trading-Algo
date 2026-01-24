"""
Portfolio Class for Position Sizing and Instrument Allocation

This module provides a Portfolio class that applies instrument weighting,
Instrument Diversification Multiplier (IDM), and optional position capping
to combined forecasts from the WeightLayer.

The Portfolio is the final layer before Execution:
    WeightLayer.combine() -> Portfolio.predict() -> PositionSizer.calculate_positions()

Key responsibilities:
1. Apply instrument weights (equal weight or custom allocation)
2. Calculate and apply IDM from instrument return correlations
3. Apply optional position capping

Reference: Robert Carver's "Systematic Trading" and "Leveraged Trading"
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
from utils.enums import TimeFrame
from .weight_layer import WeightLayer

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio class for applying instrument weighting and IDM to combined forecasts.

    The Portfolio uses WeightLayer to combine forecasts from all base models across all ensembles.
    WeightLayer applies FDM (Forecast Diversification Multiplier) during combination.
    Portfolio then applies:
    1. Instrument weights (default: equal weight per instrument)
    2. Instrument Diversification Multiplier (IDM)
    3. Optional position capping

    Parameters
    ----------
    ensembles : List[DiversifiedEnsemble], optional
        List of ensembles for this portfolio (new API)
    trading_timeframe : TimeFrame, default=TimeFrame.D
        The timeframe this portfolio trades on
    target_volatility : float, optional
        Target volatility for position sizing (new API)
    max_position_pct : float, default=2.0
        Maximum position size per instrument (e.g., 2.0 = 200%)
    weight_layer : WeightLayer, optional
        Weight layer for combining forecasts. If None, creates default inverse correlation WeightLayer.
    instrument_weights : Dict[str, float], optional
        Custom weights per instrument. If None, equal weight.
    idm_max : float, default=2.5
        Maximum IDM value (Carver's recommendation)

    Attributes
    ----------
    ensembles : List[DiversifiedEnsemble]
        List of ensembles for this portfolio
    weight_layer : WeightLayer
        The WeightLayer for forecast combination (required, created if not provided)
    trading_timeframe : TimeFrame
        The trading timeframe
    max_position_pct : float or None
        Maximum position size
    instrument_weights : Dict[str, float] or None
        Custom instrument weights
    idm_max : float
        Maximum IDM value
    idm_ : float
        Fitted Instrument Diversification Multiplier
    mean_return_correlation_ : float
        Mean correlation between instrument returns (for diagnostics)
    instruments_ : List[str]
        List of instruments seen during fit
    is_fitted_ : bool
        Whether the Portfolio has been fitted

    Examples
    --------
    >>> # Using default WeightLayer
    >>> portfolio = Portfolio(
    ...     ensembles=[ensemble1, ensemble2],
    ...     trading_timeframe=TimeFrame.D,
    ...     max_position_pct=2.0
    ... )
    >>> portfolio.fit_from_candles(candles_df, target_data)
    >>> positions = portfolio.predict_from_candles(test_candles)
    >>> 
    >>> # Using custom WeightLayer
    >>> custom_weight_layer = WeightLayer(weight_method='inverse_correlation', fdm_max=2.0)
    >>> portfolio = Portfolio(
    ...     ensembles=[ensemble1, ensemble2],
    ...     weight_layer=custom_weight_layer,
    ...     trading_timeframe=TimeFrame.D
    ... )
    """
    
    def __init__(
        self,
        ensembles: Optional[List] = None,
        trading_timeframe: TimeFrame = TimeFrame.D,
        target_volatility: Optional[float] = None,
        max_position_pct: float = 2.0,
        weight_layer: Optional[WeightLayer] = None,
        instrument_weights: Optional[Dict[str, float]] = None,
        idm_max: float = 2.5
    ):
        """
        Initialize Portfolio.

        Parameters
        ----------
        ensembles : List[DiversifiedEnsemble], optional
            List of ensembles for this portfolio (new API)
        trading_timeframe : TimeFrame, default=TimeFrame.D
            The timeframe this portfolio trades on
        target_volatility : float, optional
            Target volatility for position sizing (new API)
        max_position_pct : float, default=2.0
            Maximum position size per instrument (e.g., 2.0 = 200%)
        weight_layer : WeightLayer, optional
            Weight layer for combining forecasts. If None, creates default inverse correlation WeightLayer.
        instrument_weights : Dict[str, float], optional
            Custom weights per instrument. If None, equal weight.
        idm_max : float, default=2.5
            Maximum IDM value (Carver's recommendation)
        """
        # New API: ensembles-based
        self.ensembles = ensembles if ensembles is not None else []
        self.trading_timeframe = trading_timeframe
        self.target_volatility = target_volatility
        
        # WeightLayer is required - create default if not provided
        if weight_layer is None:
            # Create default inverse correlation WeightLayer
            self.weight_layer = WeightLayer(
                weight_method='inverse_correlation',
                fdm_max=2.0
            )
        else:
            self.weight_layer = weight_layer
        
        # Common attributes
        self.max_position_pct = max_position_pct
        self.instrument_weights = instrument_weights
        self.idm_max = idm_max

        # Fitted attributes
        self.idm_: Optional[float] = None
        self.mean_return_correlation_: Optional[float] = None
        self.instruments_: Optional[List[str]] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        instrument_returns: pd.DataFrame,
        idm_override: Optional[float] = None
    ) -> 'Portfolio':
        """
        Fit IDM from historical instrument returns.

        The IDM (Instrument Diversification Multiplier) accounts for portfolio-level
        diversification benefits. Lower correlation between instruments = higher IDM.

        Formula:
            IDM = sqrt(1 / (mean_corr + epsilon))
            Capped at idm_max (typically 2.5)

        Parameters
        ----------
        instrument_returns : pd.DataFrame
            Historical returns for all instruments.
            Columns: instrument tickers, rows: time periods
        idm_override : float, optional
            If provided, use this IDM value instead of calculating from returns.
            Useful for testing or when using pre-calculated IDM.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If instrument_returns is empty or has insufficient data
        """
        if idm_override is not None:
            self.idm_ = min(idm_override, self.idm_max)
            self.mean_return_correlation_ = None
            self.instruments_ = list(instrument_returns.columns) if not instrument_returns.empty else []
            self.is_fitted_ = True
            return self

        if instrument_returns.empty:
            raise ValueError("instrument_returns DataFrame cannot be empty")

        if len(instrument_returns) < 2:
            raise ValueError("Need at least 2 time periods to calculate IDM")

        self.instruments_ = list(instrument_returns.columns)

        # Calculate IDM from return correlations
        self.idm_ = self._calculate_idm(instrument_returns)
        self.is_fitted_ = True

        return self

    def _calculate_idm(self, instrument_returns: pd.DataFrame) -> float:
        """
        Calculate Instrument Diversification Multiplier from instrument return correlations.
        
        Formula: IDM = sqrt(1 / (mean_correlation + epsilon))
        Where mean_correlation is calculated from the correlation matrix of instrument returns.
        
        Steps:
        1. Build correlation matrix of instrument returns (tickers as columns)
        2. Calculate mean absolute correlation: mean(|rho_ij|) for i != j
        3. Floor negative correlations at zero (Carver's recommendation)
        4. Calculate IDM: sqrt(1 / (mean_correlation + epsilon))
        5. Cap at idm_max (default 2.5)
        
        Parameters
        ----------
        instrument_returns : pd.DataFrame
            Historical returns for all instruments.
            Columns: instrument tickers, rows: time periods

        Returns
        -------
        float
            IDM value (capped at idm_max)
        """
        if instrument_returns.empty or len(instrument_returns.columns) < 2:
            # Single instrument or no data: IDM = 1.0
            self.mean_return_correlation_ = 1.0
            return 1.0
        
        # Build correlation matrix of instrument returns
        # Columns are tickers, rows are time periods
        corr_matrix = instrument_returns.corr()
        
        # Floor negative correlations at zero (Carver's recommendation)
        # This treats negative correlations as zero (no diversification benefit from negative correlation)
        corr_matrix = corr_matrix.clip(lower=0.0)
        
        # Calculate mean correlation (excluding diagonal)
        # Get upper triangle (excluding diagonal) and calculate mean
        # This gives us mean(|rho_ij|) for i != j as specified
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()
        
        if len(correlations) == 0:
            # No correlations to calculate (shouldn't happen with 2+ instruments)
            self.mean_return_correlation_ = 1.0
            return 1.0
        
        # Mean correlation: mean(|rho_ij|) for i != j
        # Since we've already floored at zero, this is effectively mean(|rho_ij|)
        mean_correlation = correlations.mean()
        self.mean_return_correlation_ = mean_correlation
        
        # Calculate IDM: sqrt(1 / (mean_correlation + epsilon))
        # Lower correlation = higher IDM (more diversification benefit)
        epsilon = 0.01  # Small epsilon to avoid division by zero
        idm = np.sqrt(1.0 / (mean_correlation + epsilon))
        
        # Cap at idm_max (Carver's recommendation: 2.5)
        idm = min(idm, self.idm_max)
        
        return idm

    def predict(
        self,
        combined_forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply instrument weighting, IDM, and optional capping to combined forecasts.

        Parameters
        ----------
        combined_forecasts : pd.DataFrame
            Combined forecasts from WeightLayer.
            Required columns: ['ticker', 'forecast_score']
            forecast_score should already be FDM-scaled
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - ticker: Instrument identifier
            - forecast_score: Original forecast from WeightLayer (passed through)
            - position_fraction: Position fraction after instrument weighting, IDM, and capping

        Raises
        ------
        ValueError
            If Portfolio has not been fitted or input is invalid
        """
        if not self.is_fitted_:
            raise ValueError(
                "Portfolio must be fitted before calling predict(). "
                "Call fit() with instrument returns first."
            )

        # Validate input
        if not isinstance(combined_forecasts, pd.DataFrame):
            raise ValueError(
                f"combined_forecasts must be pd.DataFrame, got {type(combined_forecasts)}"
            )

        required_cols = ['ticker', 'forecast_score']
        missing_cols = set(required_cols) - set(combined_forecasts.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if combined_forecasts.empty:
            return pd.DataFrame(columns=['ticker', 'forecast_score', 'position_fraction'])

        # Step 1: Apply instrument weights
        weighted = self._apply_instrument_weights(combined_forecasts)

        # Step 2: Apply IDM
        idm_scaled = self._apply_idm(weighted)

        # Step 3: Apply position cap (optional)
        result = self._apply_position_cap(idm_scaled)

        return result

    def _apply_instrument_weights(
        self,
        combined_forecasts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply instrument weights to combined forecasts.

        Parameters
        ----------
        combined_forecasts : pd.DataFrame
            Columns: ['ticker', 'forecast_score']

        Returns
        -------
        pd.DataFrame
            Columns: ['ticker', 'forecast_score', 'instrument_weight', 'position_weighted']
        """
        df = combined_forecasts.copy()

        # Determine instrument weights
        if self.instrument_weights is not None:
            # Use custom weights
            df['instrument_weight'] = df['ticker'].map(self.instrument_weights)

            # Handle missing weights (instruments not in custom weights)
            if df['instrument_weight'].isna().any():
                missing = df[df['instrument_weight'].isna()]['ticker'].unique()
                # Assign equal share of remaining weight
                n_missing = len(missing)
                used_weight = sum(self.instrument_weights.get(t, 0) for t in df['ticker'].unique() if t not in missing)
                remaining_weight = max(1.0 - used_weight, 0.0)
                fallback_weight = remaining_weight / n_missing if n_missing > 0 else 0.0
                df['instrument_weight'] = df['instrument_weight'].fillna(fallback_weight)
        else:
            # Equal weight per unique instrument
            unique_instruments = df['ticker'].nunique()
            equal_weight = 1.0 / unique_instruments if unique_instruments > 0 else 0.0
            df['instrument_weight'] = equal_weight

        # Calculate weighted position
        df['position_weighted'] = df['forecast_score'] * df['instrument_weight']

        return df

    def _apply_idm(
        self,
        positions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply IDM to account for portfolio-level diversification.

        Parameters
        ----------
        positions : pd.DataFrame
            Columns: ['ticker', 'forecast_score', 'instrument_weight', 'position_weighted']

        Returns
        -------
        pd.DataFrame
            Same columns plus 'position_idm'
        """
        df = positions.copy()
        df['position_idm'] = df['position_weighted'] * self.idm_
        return df

    def _apply_position_cap(
        self,
        positions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply position cap if specified.

        Parameters
        ----------
        positions : pd.DataFrame
            Columns include 'position_idm'

        Returns
        -------
        pd.DataFrame
            Columns: ['ticker', 'forecast_score', 'position_fraction']
        """
        df = positions.copy()

        if self.max_position_pct is not None:
            df['position_fraction'] = df['position_idm'].clip(
                lower=-self.max_position_pct,
                upper=self.max_position_pct
            )
        else:
            df['position_fraction'] = df['position_idm']

        return df[['ticker', 'forecast_score', 'position_fraction']]
    
    def fit_from_candles(
        self,
        candles_df: pd.DataFrame,
        target_data: Optional[pd.Series] = None
    ) -> 'Portfolio':
        """
        Fit all ensembles using candles DataFrame.
        
        This is the new DataFrame-based API for fitting portfolios.
        Routes candles to each ensemble, which routes to base models.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
            Should contain candles for the trading_timeframe of this portfolio
        target_data : pd.Series, optional
            Target values (returns) for training. If None, ensembles must be pre-fitted.
            
        Returns
        -------
        self
            Fitted portfolio
        """
        if not self.ensembles:
            raise ValueError("No ensembles provided. Cannot fit portfolio without ensembles.")
        
        # Filter candles for this portfolio's trading timeframe
        tf_candles = candles_df[candles_df['timeframe'] == self.trading_timeframe].copy()
        
        if tf_candles.empty:
            raise ValueError(
                f"No candles found for trading timeframe {self.trading_timeframe.name}. "
                f"Available timeframes: {candles_df['timeframe'].unique()}"
            )
        
        # Fit all ensembles
        for ensemble in self.ensembles:
            if target_data is not None:
                # Use the new fit_from_candles method if available
                if hasattr(ensemble, 'fit_from_candles'):
                    ensemble.fit_from_candles(tf_candles, target_data)
                else:
                    # Fallback: ensembles must be pre-fitted
                    pass
        
        # Fit IDM if we have return data
        if target_data is not None:
            # Calculate returns from candles for IDM calculation
            returns_df = self._calculate_returns_from_candles(tf_candles)
            
            # Debug logging
            logger.debug(
                f"IDM calculation: returns_df shape={returns_df.shape}, "
                f"columns={list(returns_df.columns) if not returns_df.empty else []}, "
                f"unique tickers in candles={sorted(tf_candles['ticker'].unique().tolist())}"
            )
            
            if not returns_df.empty and len(returns_df.columns) >= 2:
                # Need at least 2 instruments to calculate IDM
                # Calculate IDM from correlations
                try:
                    self.fit(returns_df)
                    logger.debug(
                        f"IDM calculated successfully: IDM={self.idm_}, "
                        f"mean_correlation={self.mean_return_correlation_}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error calculating IDM: {e}. Using default IDM=1.0",
                        exc_info=True
                    )
                    self.idm_ = 1.0
                    self.mean_return_correlation_ = 1.0
                
                # If IDM wasn't calculated (e.g., insufficient data), default to 1.0
                if self.idm_ is None:
                    logger.warning("IDM is None after fit(). Using default IDM=1.0")
                    self.idm_ = 1.0
                    self.mean_return_correlation_ = 1.0
            else:
                # Not enough instruments or empty returns - set default IDM
                if returns_df.empty:
                    logger.warning(
                        f"Returns DataFrame is empty after processing. "
                        f"Input candles: {len(tf_candles)} rows, "
                        f"unique tickers: {sorted(tf_candles['ticker'].unique().tolist())}. "
                        f"Cannot calculate IDM. Using default IDM=1.0"
                    )
                elif len(returns_df.columns) < 2:
                    logger.warning(
                        f"Only {len(returns_df.columns)} instrument(s) in returns DataFrame. "
                        f"Need at least 2 instruments to calculate IDM. "
                        f"Available tickers in candles: {sorted(tf_candles['ticker'].unique().tolist())}. "
                        f"Using default IDM=1.0"
                    )
                self.idm_ = 1.0
                self.mean_return_correlation_ = 1.0
                # Set instruments_ from candles if not already set
                if self.instruments_ is None:
                    self.instruments_ = sorted(tf_candles['ticker'].unique().tolist())
            
            # Fit WeightLayer (calculates weights and FDM from forecast correlations)
            self._fit_weight_layer(tf_candles)
        
        self.is_fitted_ = True
        return self
    
    def predict_from_candles(
        self,
        candles_df: pd.DataFrame,
        return_ensemble_predictions: bool = False,
        return_base_model_predictions: bool = False
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Generate position fractions using candles DataFrame.
        
        This is the new DataFrame-based API for prediction.
        Routes candles to ensembles, aggregates predictions, and applies risk management.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
            Should contain candles for the trading_timeframe of this portfolio
        return_ensemble_predictions : bool, default=False
            If True, return ensemble-level predictions in result dict
        return_base_model_predictions : bool, default=False
            If True, return base model-level predictions in result dict
            
        Returns
        -------
        pd.DataFrame or Dict[str, Any]
            If both flags are False: DataFrame with portfolio positions
            If either flag is True: Dict with structure:
            {
                'portfolio': pd.DataFrame,  # Portfolio-level positions
                'ensembles': Dict[str, pd.DataFrame],  # Only if return_ensemble_predictions=True
                'base_models': Dict[str, pd.DataFrame]  # Only if return_base_model_predictions=True
            }
        """
        if not self.ensembles:
            raise ValueError("No ensembles provided. Cannot predict without ensembles.")
        
        # Filter candles for this portfolio's trading timeframe
        tf_candles = candles_df[candles_df['timeframe'] == self.trading_timeframe].copy()
        
        if tf_candles.empty:
            empty_df = pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score', 'position_fraction'])
            if return_ensemble_predictions or return_base_model_predictions:
                result = {'portfolio': empty_df}
                if return_ensemble_predictions:
                    result['ensembles'] = {}
                if return_base_model_predictions:
                    result['base_models'] = {}
                return result
            return empty_df
        
        # Calculate volatility (from candles) - needed for all predictions
        volatility = self._calculate_volatility_from_candles(tf_candles)
        
        # Storage for granular predictions
        forecast_vectors = []  # For WeightLayer.combine()
        ensemble_predictions_dict = {}
        base_model_predictions_dict = {}
        
        # Get predictions from all ensembles
        for ensemble_idx, ensemble in enumerate(self.ensembles):
            # Use the new predict_from_candles method if available
            if hasattr(ensemble, 'predict_from_candles'):
                # Always get base model predictions to build forecast vectors for WeightLayer
                ensemble_result = ensemble.predict_from_candles(
                    tf_candles,
                    volatility=volatility,  # Pass volatility for proper scaling
                    return_base_model_predictions=True
                )
                
                if isinstance(ensemble_result, dict):
                    ensemble_pred = ensemble_result.get('ensemble')
                    base_models = ensemble_result.get('base_models', {})
                    
                    # Build forecast vectors for WeightLayer (one per ensemble)
                    # WeightLayer expects: ['ticker', 'model_name', 'forecast', 'signal']
                    ensemble_forecast_vector = []
                    for model_name, model_pred in base_models.items():
                        if isinstance(model_pred, pd.DataFrame) and 'forecast_score' in model_pred.columns:
                            # Convert to WeightLayer format
                            forecast_df = model_pred.copy()
                            forecast_df['model_name'] = model_name
                            forecast_df['forecast'] = forecast_df['forecast_score']
                            # Extract signal from forecast (1 if forecast > 0, else 0)
                            forecast_df['signal'] = (forecast_df['forecast'] > 0).astype(int)
                            # Keep only required columns
                            forecast_df = forecast_df[['ticker', 'datetime', 'model_name', 'forecast', 'signal']]
                            ensemble_forecast_vector.append(forecast_df)
                    
                    if ensemble_forecast_vector:
                        # Combine all base models from this ensemble into one forecast vector
                        ensemble_vector_df = pd.concat(ensemble_forecast_vector, ignore_index=True)
                        forecast_vectors.append(ensemble_vector_df)
                    
                    # Store base model predictions with ensemble prefix (for granular output)
                    if return_base_model_predictions:
                        for model_name, model_pred in base_models.items():
                            ensemble_name = f"ensemble_{ensemble_idx}"
                            full_model_name = f"{ensemble_name}::{model_name}"
                            # Convert to position fractions (forecasts already volatility-adjusted)
                            base_model_positions = self._apply_risk_management_to_forecasts(
                                model_pred, volatility, tf_candles
                            )
                            base_model_predictions_dict[full_model_name] = base_model_positions
                    
                    # Store ensemble-level predictions if requested
                    if return_ensemble_predictions:
                        ensemble_name = f"ensemble_{ensemble_idx}"
                        # Convert to position fractions
                        ensemble_positions = self._apply_risk_management_to_forecasts(
                            ensemble_pred, volatility, tf_candles
                        )
                        ensemble_predictions_dict[ensemble_name] = ensemble_positions
                else:
                    # If ensemble doesn't return dict, it's already aggregated
                    ensemble_pred = ensemble_result
                    if return_ensemble_predictions:
                        ensemble_name = f"ensemble_{ensemble_idx}"
                        ensemble_positions = self._apply_risk_management_to_forecasts(
                            ensemble_pred, volatility, tf_candles
                        )
                        ensemble_predictions_dict[ensemble_name] = ensemble_positions
            else:
                # Fallback: use regular predict (requires features, not implemented here)
                pass
        
        if not forecast_vectors:
            empty_df = pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score', 'position_fraction'])
            if return_ensemble_predictions or return_base_model_predictions:
                result = {'portfolio': empty_df}
                if return_ensemble_predictions:
                    result['ensembles'] = {}
                if return_base_model_predictions:
                    result['base_models'] = {}
                return result
            return empty_df
        
        # Use WeightLayer to combine forecasts from all base models across all ensembles
        if self.weight_layer.is_fitted_:
            # WeightLayer is fitted - use it to combine forecasts
            combined_forecasts = self.weight_layer.combine(forecast_vectors)
            # Convert to format expected by _apply_risk_management (needs datetime column)
            # WeightLayer returns ['ticker', 'forecast_score'], but we need datetime
            # We'll merge with candles to get datetime alignment
            forecast_scores_df = self._align_forecasts_with_candles(combined_forecasts, tf_candles)
        else:
            # WeightLayer not fitted - fallback to simple averaging
            # This should not happen if fit_from_candles was called, but handle gracefully
            forecast_scores_df = self._aggregate_ensembles_fallback(forecast_vectors, tf_candles)
        
        # Apply risk management for portfolio-level
        positions_df = self._apply_risk_management(
            forecast_scores_df, volatility, tf_candles
        )
        
        # Return structure based on flags
        if return_ensemble_predictions or return_base_model_predictions:
            result = {'portfolio': positions_df}
            if return_ensemble_predictions:
                result['ensembles'] = ensemble_predictions_dict
            if return_base_model_predictions:
                result['base_models'] = base_model_predictions_dict
            return result
        
        return positions_df
    
    def _aggregate_ensembles(
        self,
        ensemble_predictions: List[pd.DataFrame],
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate predictions from multiple ensembles.
        
        Aggregates across ensembles (if multiple), but preserves ticker-level information.
        Each ticker keeps its own forecast_score.
        
        Parameters
        ----------
        ensemble_predictions : List[pd.DataFrame]
            List of prediction DataFrames from each ensemble (columns: ticker, datetime, forecast_score)
        candles_df : pd.DataFrame
            Candles DataFrame for alignment
            
        Returns
        -------
        pd.DataFrame
            Aggregated forecast scores with columns: ticker, datetime, forecast_score
            (aggregated across ensembles, but preserving ticker-level granularity)
        """
        if not ensemble_predictions:
            return pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score'])
        
        # Combine all predictions (DataFrames)
        combined_df = pd.concat(ensemble_predictions, ignore_index=True)
        
        # Group by (datetime, ticker) and take mean of forecast_score across ensembles
        # This preserves ticker-level information while aggregating across ensembles
        if 'ticker' in combined_df.columns and 'datetime' in combined_df.columns:
            # Group by (datetime, ticker) and take mean across ensembles
            aggregated = combined_df.groupby(['datetime', 'ticker'])['forecast_score'].mean().reset_index()
        elif 'datetime' in combined_df.columns:
            # If no ticker column, just group by datetime
            aggregated = combined_df.groupby('datetime')['forecast_score'].mean().reset_index()
            # Add dummy ticker column if needed (shouldn't happen in normal flow)
            if 'ticker' not in aggregated.columns:
                aggregated['ticker'] = candles_df['ticker'].iloc[0] if len(candles_df) > 0 else None
        else:
            # Fallback: use index if datetime is index
            if combined_df.index.name == 'datetime' or isinstance(combined_df.index, pd.DatetimeIndex):
                aggregated = combined_df.groupby(combined_df.index)['forecast_score'].mean().reset_index()
                aggregated.columns = ['datetime', 'forecast_score']
            else:
                # Last resort: try to use index as-is
                aggregated = combined_df.set_index('datetime')['forecast_score'].groupby(level=0).mean().reset_index()
                aggregated.columns = ['datetime', 'forecast_score']
            
            # Add ticker column if missing
            if 'ticker' not in aggregated.columns:
                aggregated['ticker'] = candles_df['ticker'].iloc[0] if len(candles_df) > 0 else None
        
        # Ensure columns are in correct order
        if 'ticker' in aggregated.columns and 'datetime' in aggregated.columns:
            aggregated = aggregated[['ticker', 'datetime', 'forecast_score']]
        
        # Note: FDM is now applied by WeightLayer.combine(), not here
        # This method is only used as fallback when WeightLayer is not fitted
        
        return aggregated
    
    def _calculate_volatility_from_candles(
        self,
        candles_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate blended volatility from EWSD bias nodes.
        
        Creates EWSD (Exponentially Weighted Standard Deviation) nodes for each ticker
        and streams candles to build up volatility estimates. EWSD nodes implement
        Carver's blended volatility:
        - 70% EWMA-32 (short-run estimate)
        - 30% long-run historical average (10-year window)
        
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
    
    def _calculate_returns_from_candles(
        self,
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate returns DataFrame from candles for IDM calculation.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame
            
        Returns
        -------
        pd.DataFrame
            Returns DataFrame with tickers as columns, datetime as index
        """
        returns_dict = {}
        date_ranges_dict = {}  # Store date ranges for logging
        
        for ticker in candles_df['ticker'].unique():
            ticker_candles = candles_df[candles_df['ticker'] == ticker].copy()
            
            # Ensure datetime column is properly formatted as datetime
            ticker_candles['datetime'] = pd.to_datetime(ticker_candles['datetime'])
            ticker_candles = ticker_candles.sort_values('datetime')
            
            # Calculate returns
            ticker_candles['returns'] = ticker_candles['close'].pct_change()
            
            # Set datetime as index (already datetime type)
            ticker_candles = ticker_candles.set_index('datetime')
            # Drop NaN from individual ticker (first row will be NaN from pct_change)
            ticker_returns = ticker_candles['returns'].dropna()
            
            # Convert ticker to string for dictionary key (handle both enum and string)
            if hasattr(ticker, 'name'):
                ticker_key = ticker.name  # Ticker enum
            elif hasattr(ticker, 'value'):
                ticker_key = str(ticker.value)  # Fallback
            else:
                ticker_key = str(ticker)  # String or other
            
            if len(ticker_returns) > 0:
                # Ensure datetime index is properly formatted as DatetimeIndex
                if not isinstance(ticker_returns.index, pd.DatetimeIndex):
                    ticker_returns.index = pd.to_datetime(ticker_returns.index)
                
                # Normalize datetime index to remove timezone and time components for alignment
                # This ensures all tickers align on the same dates
                ticker_returns.index = ticker_returns.index.normalize()
                
                returns_dict[ticker_key] = ticker_returns
                date_ranges_dict[ticker_key] = (ticker_returns.index.min(), ticker_returns.index.max())
                
                logger.debug(
                    f"Added returns for ticker {ticker_key}: {len(ticker_returns)} rows, "
                    f"date range: {ticker_returns.index.min()} to {ticker_returns.index.max()}"
                )
            else:
                logger.warning(f"No valid returns for ticker {ticker_key} after dropna()")
                date_ranges_dict[ticker_key] = (None, None)
        
        if not returns_dict:
            logger.warning("No returns calculated for any ticker")
            return pd.DataFrame()
        
        # Create DataFrame with all ticker returns
        # This will align by datetime index (union of all datetimes)
        # Ensure all indices are normalized and the same type before creating DataFrame
        normalized_returns_dict = {}
        for ticker_key, ticker_returns in returns_dict.items():
            # Create a copy to avoid modifying original
            normalized_returns = ticker_returns.copy()
            
            # Ensure index is DatetimeIndex and normalized (date-only, no time)
            if not isinstance(normalized_returns.index, pd.DatetimeIndex):
                normalized_returns.index = pd.to_datetime(normalized_returns.index)
            
            # Normalize to remove time components (ensures alignment)
            normalized_returns.index = normalized_returns.index.normalize()
            
            normalized_returns_dict[ticker_key] = normalized_returns
        
        returns_df = pd.DataFrame(normalized_returns_dict)
        
        logger.debug(
            f"Returns DataFrame created: shape={returns_df.shape}, "
            f"columns={list(returns_df.columns)}, "
            f"index type: {type(returns_df.index)}, "
            f"NaN count per column: {returns_df.isna().sum().to_dict()}, "
            f"Sample index values: {returns_df.index[:5].tolist() if len(returns_df) > 0 else 'empty'}"
        )
        
        # Only drop rows where we have fewer than 2 tickers with valid returns
        # We need at least 2 tickers for correlation calculation
        # Use dropna with thresh=2 to keep rows with at least 2 non-NaN values
        if len(returns_df.columns) >= 2:
            rows_before = len(returns_df)
            # Count non-NaN values per row
            non_nan_per_row = returns_df.notna().sum(axis=1)
            rows_with_2plus = (non_nan_per_row >= 2).sum()
            
            logger.debug(
                f"Before dropna(thresh=2): {rows_before} rows, "
                f"{rows_with_2plus} rows have 2+ non-NaN values"
            )
            
            returns_df = returns_df.dropna(thresh=2)
            rows_after = len(returns_df)
            
            if rows_after == 0:
                # Get date ranges from stored dict (before DataFrame creation)
                date_ranges = [(col, date_ranges_dict.get(col, (None, None))[0], date_ranges_dict.get(col, (None, None))[1]) 
                              for col in returns_df.columns]
                
                logger.warning(
                    f"Returns DataFrame is empty after dropna(thresh=2). "
                    f"Input: {rows_before} rows, {len(returns_df.columns)} columns. "
                    f"Only {rows_with_2plus} rows had 2+ non-NaN values. "
                    f"This suggests tickers have no overlapping datetime indices. "
                    f"Date ranges from individual tickers: {date_ranges}"
                )
            else:
                logger.debug(
                    f"After dropna(thresh=2): {rows_before} -> {rows_after} rows, "
                    f"columns={list(returns_df.columns)}"
                )
        else:
            # If we have fewer than 2 tickers, return empty
            logger.warning(f"Only {len(returns_df.columns)} ticker(s) in returns DataFrame")
            return pd.DataFrame()
        
        return returns_df
    
    def _fit_weight_layer(
        self,
        candles_df: pd.DataFrame
    ) -> None:
        """
        Fit WeightLayer from forecast vectors and signals.
        
        Collects forecast vectors from all ensembles, extracts binary signals,
        and fits the WeightLayer (which calculates weights and FDM).
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Candles DataFrame for generating forecasts
        """
        if not self.ensembles:
            return
        
        # Calculate volatility for forecast generation
        volatility = self._calculate_volatility_from_candles(candles_df)
        
        # Collect forecast vectors and signals from all ensembles
        forecast_vectors = []
        all_signals_list = []
        
        for ensemble in self.ensembles:
            if not hasattr(ensemble, 'predict_from_candles'):
                continue
            
            try:
                # Get ensemble predictions with base model granularity
                ensemble_result = ensemble.predict_from_candles(
                    candles_df,
                    volatility=volatility,
                    return_base_model_predictions=True
                )
                
                if isinstance(ensemble_result, dict):
                    base_models = ensemble_result.get('base_models', {})
                    ensemble_forecast_vector = []
                    
                    for model_name, model_pred in base_models.items():
                        if isinstance(model_pred, pd.DataFrame) and 'forecast_score' in model_pred.columns:
                            # Convert to WeightLayer format: ['ticker', 'model_name', 'forecast', 'signal']
                            forecast_df = model_pred.copy()
                            forecast_df['model_name'] = model_name
                            forecast_df['forecast'] = forecast_df['forecast_score']
                            # Extract signal from forecast (1 if forecast > 0, else 0)
                            forecast_df['signal'] = (forecast_df['forecast'] > 0).astype(int)
                            # Keep only required columns
                            forecast_df = forecast_df[['ticker', 'datetime', 'model_name', 'forecast', 'signal']]
                            ensemble_forecast_vector.append(forecast_df)
                            
                            # Extract signals for weight calculation (pivot by model_name)
                            signal_series = forecast_df.set_index('datetime')['signal']
                            all_signals_list.append((model_name, signal_series))
                    
                    if ensemble_forecast_vector:
                        # Combine all base models from this ensemble into one forecast vector
                        ensemble_vector_df = pd.concat(ensemble_forecast_vector, ignore_index=True)
                        forecast_vectors.append(ensemble_vector_df)
            except Exception:
                # Skip this ensemble if prediction fails
                continue
        
        if len(forecast_vectors) == 0:
            # Cannot fit WeightLayer without forecast vectors
            return
        
        # Build signals DataFrame for weight calculation
        # Pivot signals by model_name (columns) and datetime (index)
        if all_signals_list:
            # Find common datetime index
            common_index = all_signals_list[0][1].index
            for _, signal_series in all_signals_list[1:]:
                common_index = common_index.intersection(signal_series.index)
            
            if len(common_index) >= 2:
                signals_df = pd.DataFrame(index=common_index)
                for model_name, signal_series in all_signals_list:
                    aligned_signal = signal_series.reindex(common_index)
                    signals_df[model_name] = aligned_signal
                
                # Drop rows with any NaN
                signals_df = signals_df.dropna()
                
                if len(signals_df) >= 2 and len(signals_df.columns) >= 1:
                    # Fit WeightLayer
                    try:
                        self.weight_layer.fit(forecast_vectors, signals_df)
                    except Exception:
                        # If fitting fails, WeightLayer will remain unfitted
                        # It will use defaults when combine() is called
                        pass
    
    def _align_forecasts_with_candles(
        self,
        combined_forecasts: pd.DataFrame,
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align combined forecasts (from WeightLayer) with candles DataFrame.
        
        WeightLayer.combine() returns ['ticker', 'datetime', 'forecast_score'] if datetime is available.
        If datetime is missing, we merge with candles to add it.
        
        Parameters
        ----------
        combined_forecasts : pd.DataFrame
            Combined forecasts from WeightLayer with columns: ['ticker', 'forecast_score'] or ['ticker', 'datetime', 'forecast_score']
        candles_df : pd.DataFrame
            Candles DataFrame for datetime alignment
            
        Returns
        -------
        pd.DataFrame
            Forecast scores with columns: ['ticker', 'datetime', 'forecast_score']
        """
        # If datetime is already in combined_forecasts, use it directly
        if 'datetime' in combined_forecasts.columns:
            # Ensure datetime is datetime type
            combined_forecasts = combined_forecasts.copy()
            combined_forecasts['datetime'] = pd.to_datetime(combined_forecasts['datetime'])
            return combined_forecasts[['ticker', 'datetime', 'forecast_score']]
        
        # Fallback: datetime not in combined_forecasts, merge with candles
        # Group candles by ticker and get unique datetimes
        # For each ticker, assign forecast_score to all its datetimes
        results = []
        for ticker in combined_forecasts['ticker'].unique():
            ticker_candles = candles_df[candles_df['ticker'] == ticker].copy()
            ticker_forecast = combined_forecasts[combined_forecasts['ticker'] == ticker]
            
            if len(ticker_forecast) > 0:
                forecast_score = ticker_forecast.iloc[0]['forecast_score']
            else:
                forecast_score = 0.0
            
            # Assign same forecast_score to all datetimes for this ticker
            for _, row in ticker_candles.iterrows():
                results.append({
                    'ticker': ticker,
                    'datetime': pd.to_datetime(row['datetime']),
                    'forecast_score': forecast_score
                })
        
        return pd.DataFrame(results)
    
    def _aggregate_ensembles_fallback(
        self,
        forecast_vectors: List[pd.DataFrame],
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fallback aggregation when WeightLayer is not fitted.
        
        Simple averaging across all base models.
        
        Parameters
        ----------
        forecast_vectors : List[pd.DataFrame]
            List of forecast vectors from all ensembles
        candles_df : pd.DataFrame
            Candles DataFrame for alignment
            
        Returns
        -------
        pd.DataFrame
            Aggregated forecast scores with columns: ['ticker', 'datetime', 'forecast_score']
        """
        if not forecast_vectors:
            return pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score'])
        
        # Combine all forecast vectors
        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        
        # Group by (datetime, ticker) and take mean of forecast
        if 'ticker' in all_forecasts.columns and 'datetime' in all_forecasts.columns:
            aggregated = all_forecasts.groupby(['datetime', 'ticker'])['forecast'].mean().reset_index()
            aggregated.columns = ['datetime', 'ticker', 'forecast_score']
            aggregated = aggregated[['ticker', 'datetime', 'forecast_score']]
        else:
            # Fallback
            aggregated = pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score'])
        
        return aggregated
    
    def _apply_risk_management_to_forecasts(
        self,
        forecasts_df: pd.DataFrame,
        volatility: Dict[str, float],
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply risk management to forecast DataFrame (for ensemble/base model level).
        
        Similar to _apply_risk_management but works with forecast DataFrames
        that have ticker and datetime columns.
        Note: Forecasts are already volatility-adjusted from Ensemble layer.
        
        Parameters
        ----------
        forecasts_df : pd.DataFrame
            Forecast scores with columns: ticker, datetime, forecast_score
            (already volatility-adjusted from Ensemble)
        volatility : Dict[str, float]
            Volatility per ticker (used for alignment, not scaling)
        candles_df : pd.DataFrame
            Candles DataFrame for alignment
            
        Returns
        -------
        pd.DataFrame
            Position fractions with columns: ticker, datetime, forecast_score, position_fraction
        """
        results = []
        
        # Merge forecasts with candles to align by ticker and datetime
        for _, row in candles_df.iterrows():
            dt = pd.to_datetime(row['datetime'])
            ticker = row['ticker']
            
            # Find matching forecast
            matching = forecasts_df[
                (forecasts_df['ticker'] == ticker) &
                (pd.to_datetime(forecasts_df['datetime']) == dt)
            ]
            
            if len(matching) > 0:
                forecast_score = matching.iloc[0]['forecast_score']
            else:
                # Try to find nearest datetime for this ticker
                ticker_forecasts = forecasts_df[forecasts_df['ticker'] == ticker].copy()
                if len(ticker_forecasts) > 0:
                    ticker_forecasts['datetime'] = pd.to_datetime(ticker_forecasts['datetime'])
                    ticker_forecasts = ticker_forecasts.set_index('datetime')
                    if dt in ticker_forecasts.index:
                        forecast_score = ticker_forecasts.loc[dt, 'forecast_score']
                    else:
                        # Find nearest
                        nearest_idx = ticker_forecasts.index.get_indexer([dt], method='nearest')[0]
                        if nearest_idx >= 0:
                            forecast_score = ticker_forecasts.iloc[nearest_idx]['forecast_score']
                        else:
                            forecast_score = 0.0
                else:
                    forecast_score = 0.0
            
            # Forecasts are already volatility-adjusted from Ensemble and FDM-scaled from aggregation
            # Apply IDM (Instrument Diversification Multiplier)
            idm_value = self.idm_ if self.idm_ is not None else 1.0
            scaled_forecast = forecast_score * idm_value
            
            # Apply instrument weights
            if self.instrument_weights is not None:
                inst_weight = self.instrument_weights.get(ticker, 1.0)
            else:
                # Equal weight
                unique_tickers = candles_df['ticker'].nunique()
                inst_weight = 1.0 / unique_tickers if unique_tickers > 0 else 1.0
            
            position_fraction = scaled_forecast * inst_weight
            
            # Apply position cap
            if self.max_position_pct is not None:
                position_fraction = np.clip(
                    position_fraction,
                    -self.max_position_pct,
                    self.max_position_pct
                )
            
            results.append({
                'ticker': ticker,
                'datetime': dt,
                'forecast_score': forecast_score,
                'position_fraction': position_fraction
            })
        
        return pd.DataFrame(results)
    
    def _apply_risk_management(
        self,
        forecast_scores_df: pd.DataFrame,
        volatility: Dict[str, float],
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply risk management to forecast scores.
        
        Applies DM (diversification multiplier) and instrument weights.
        Note: Forecasts are already volatility-adjusted from Ensemble layer.
        
        Parameters
        ----------
        forecast_scores_df : pd.DataFrame
            Forecast scores with columns: ticker, datetime, forecast_score
            (already volatility-adjusted from Ensemble)
        volatility : Dict[str, float]
            Volatility per ticker (used for alignment, not scaling)
        candles_df : pd.DataFrame
            Candles DataFrame for alignment
            
        Returns
        -------
        pd.DataFrame
            Position fractions with columns: ticker, datetime, forecast_score, position_fraction
        """
        results = []
        
        # Merge forecast scores with candles to align by ticker and datetime
        for _, row in candles_df.iterrows():
            dt = pd.to_datetime(row['datetime'])
            ticker = row['ticker']
            
            # Find matching forecast score for this ticker and datetime
            matching = forecast_scores_df[
                (forecast_scores_df['ticker'] == ticker) &
                (pd.to_datetime(forecast_scores_df['datetime']) == dt)
            ]
            
            if len(matching) > 0:
                forecast_score = matching.iloc[0]['forecast_score']
            else:
                # Try to find nearest datetime for this ticker
                ticker_forecasts = forecast_scores_df[forecast_scores_df['ticker'] == ticker].copy()
                if len(ticker_forecasts) > 0:
                    ticker_forecasts['datetime'] = pd.to_datetime(ticker_forecasts['datetime'])
                    ticker_forecasts = ticker_forecasts.set_index('datetime')
                    if dt in ticker_forecasts.index:
                        forecast_score = ticker_forecasts.loc[dt, 'forecast_score']
                    else:
                        # Find nearest
                        nearest_idx = ticker_forecasts.index.get_indexer([dt], method='nearest')[0]
                        if nearest_idx >= 0:
                            forecast_score = ticker_forecasts.iloc[nearest_idx]['forecast_score']
                        else:
                            forecast_score = 0.0
                else:
                    forecast_score = 0.0
            
            # Forecasts are already volatility-adjusted from Ensemble and FDM-scaled from aggregation
            # Apply IDM (Instrument Diversification Multiplier)
            idm_value = self.idm_ if self.idm_ is not None else 1.0
            scaled_forecast = forecast_score * idm_value
            
            # Apply instrument weights
            if self.instrument_weights is not None:
                inst_weight = self.instrument_weights.get(ticker, 1.0)
            else:
                # Equal weight
                unique_tickers = candles_df['ticker'].nunique()
                inst_weight = 1.0 / unique_tickers if unique_tickers > 0 else 1.0
            
            position_fraction = scaled_forecast * inst_weight
            
            # Apply position cap
            if self.max_position_pct is not None:
                position_fraction = np.clip(
                    position_fraction,
                    -self.max_position_pct,
                    self.max_position_pct
                )
            
            results.append({
                'ticker': ticker,
                'datetime': dt,
                'forecast_score': forecast_score,
                'position_fraction': position_fraction
            })
        
        return pd.DataFrame(results)

    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about the fitted Portfolio.

        Returns
        -------
        dict
            Dictionary containing:
            - is_fitted: Whether Portfolio has been fitted
            - idm: Instrument Diversification Multiplier
            - mean_return_correlation: Mean correlation between instrument returns
            - fdm: Forecast Diversification Multiplier (from WeightLayer)
            - mean_forecast_correlation: Mean correlation between forecast values (from WeightLayer)
            - n_instruments: Number of instruments
            - instruments: List of instrument tickers
            - idm_max: Maximum allowed IDM
            - fdm_max: Maximum allowed FDM (from WeightLayer)
            - max_position_pct: Position cap (if any)
            - weight_layer: Full WeightLayer diagnostics dict
        """
        weight_layer_diag = self.weight_layer.get_diagnostics() if self.weight_layer else {}
        
        # Flatten weight layer diagnostics for easier access
        diagnostics = {
            'is_fitted': self.is_fitted_,
            'idm': self.idm_,
            'mean_return_correlation': self.mean_return_correlation_,
            'fdm': weight_layer_diag.get('fdm') if isinstance(weight_layer_diag, dict) else None,
            'mean_forecast_correlation': weight_layer_diag.get('mean_forecast_correlation') if isinstance(weight_layer_diag, dict) else None,
            'n_instruments': len(self.instruments_) if self.instruments_ else 0,
            'instruments': self.instruments_,
            'idm_max': self.idm_max,
            'fdm_max': weight_layer_diag.get('fdm_max') if isinstance(weight_layer_diag, dict) else None,
            'max_position_pct': self.max_position_pct,
            'trading_timeframe': self.trading_timeframe.name if self.trading_timeframe else None,
            'weight_layer': weight_layer_diag  # Full weight layer diagnostics
        }
        
        return diagnostics
    
    def print_diagnostics(self) -> None:
        """
        Print comprehensive diagnostic information in a readable format.
        
        This method prints IDM, FDM, weight layer information, and other
        portfolio diagnostics in a formatted way for easy inspection.
        """
        print("=" * 60)
        print("PORTFOLIO DIAGNOSTICS")
        print("=" * 60)
        
        print(f"\n📊 Portfolio Status:")
        print(f"  Fitted: {self.is_fitted_}")
        print(f"  Trading Timeframe: {self.trading_timeframe.name if self.trading_timeframe else 'N/A'}")
        print(f"  Number of Instruments: {len(self.instruments_) if self.instruments_ else 0}")
        print(f"  Instruments: {self.instruments_ if self.instruments_ else 'N/A'}")
        
        print(f"\n🎯 Instrument Diversification Multiplier (IDM):")
        print(f"  IDM: {self.idm_ if self.idm_ is not None else 'Not calculated'}")
        print(f"  IDM Max: {self.idm_max}")
        print(f"  Mean Return Correlation: {self.mean_return_correlation_ if self.mean_return_correlation_ is not None else 'N/A'}")
        
        if self.weight_layer:
            weight_layer_diag = self.weight_layer.get_diagnostics()
            print(f"\n🔮 Forecast Diversification Multiplier (FDM):")
            print(f"  FDM: {weight_layer_diag.get('fdm', 'Not calculated')}")
            print(f"  FDM Max: {weight_layer_diag.get('fdm_max', 'N/A')}")
            print(f"  Mean Forecast Correlation: {weight_layer_diag.get('mean_forecast_correlation', 'N/A')}")
            
            print(f"\n⚖️  Weight Layer:")
            print(f"  Fitted: {weight_layer_diag.get('is_fitted', False)}")
            print(f"  Weight Method: {weight_layer_diag.get('weight_method', 'N/A')}")
            print(f"  Number of Models: {weight_layer_diag.get('n_models', 0)}")
            
            weights = weight_layer_diag.get('weights')
            if weights and isinstance(weights, dict):
                print(f"  Model Weights (top 5):")
                # Filter out NaN weights and sort
                valid_weights = {k: v for k, v in weights.items() if not pd.isna(v)}
                if valid_weights:
                    sorted_weights = sorted(valid_weights.items(), key=lambda x: x[1], reverse=True)[:5]
                    for model_name, weight in sorted_weights:
                        print(f"    {model_name}: {weight:.4f}")
                else:
                    print(f"    (All weights are NaN - check signal correlations)")
        else:
            print(f"\n🔮 Forecast Diversification Multiplier (FDM):")
            print(f"  WeightLayer: Not initialized")
        
        print(f"\n📏 Position Sizing:")
        print(f"  Max Position %: {self.max_position_pct}")
        print(f"  Instrument Weights: {'Custom' if self.instrument_weights else 'Equal weight'}")
        
        print("=" * 60)

    def __repr__(self) -> str:
        """String representation of the portfolio."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"Portfolio({fitted_str}, timeframe={self.trading_timeframe.name}, idm={self.idm_})"

    def __str__(self) -> str:
        """Detailed string description of the portfolio."""
        lines = [
            "Portfolio",
            f"  Trading Timeframe: {self.trading_timeframe.name}",
            f"  Is Fitted: {self.is_fitted_}",
            f"  IDM: {self.idm_}",
            f"  IDM Max: {self.idm_max}",
            f"  Max Position: {self.max_position_pct}",
            f"  Instruments: {self.instruments_}"
        ]
        return "\n".join(lines)
