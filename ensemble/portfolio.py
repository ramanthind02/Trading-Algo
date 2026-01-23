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
from typing import Dict, List, Optional, Union, Any
from utils.enums import TimeFrame
from .weight_layer import WeightLayer


class Portfolio:
    """
    Portfolio class for applying instrument weighting and IDM to combined forecasts.

    The Portfolio receives combined forecasts from the WeightLayer (already FDM-scaled)
    and applies:
    1. Instrument weights (default: equal weight per instrument)
    2. Instrument Diversification Multiplier (IDM)
    3. Optional position capping

    Parameters
    ----------
    weight_layer : WeightLayer
        The WeightLayer that combines forecasts from all ensembles
    trading_timeframe : TimeFrame
        The timeframe this portfolio trades on (e.g., TimeFrame.D for daily)
    max_position_pct : float, optional
        Maximum position size per instrument. If None, no capping.
    instrument_weights : Dict[str, float], optional
        Weight for each instrument. If None, equal weight.
        Weights should sum to 1.0 (will be normalized if not)
    idm_max : float, default=2.5
        Maximum IDM value (capped to prevent excessive leverage)

    Attributes
    ----------
    weight_layer : WeightLayer
        The WeightLayer for forecast combination
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
    >>> weight_layer = WeightLayer()
    >>> weight_layer.fit(forecast_vectors, signals)
    >>> portfolio = Portfolio(
    ...     weight_layer=weight_layer,
    ...     trading_timeframe=TimeFrame.D,
    ...     max_position_pct=2.0
    ... )
    >>> portfolio.fit(instrument_returns)
    >>> positions = portfolio.predict(combined_forecasts)
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
            Weight layer for combining forecasts (legacy API)
        instrument_weights : Dict[str, float], optional
            Custom weights per instrument. If None, equal weight.
        idm_max : float, default=2.5
            Maximum IDM value (Carver's recommendation)
        """
        # New API: ensembles-based
        self.ensembles = ensembles if ensembles is not None else []
        self.trading_timeframe = trading_timeframe
        self.target_volatility = target_volatility
        
        # Legacy API: weight_layer-based
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
        Calculate Instrument Diversification Multiplier from return correlations.
        
        Formula: IDM = sqrt(1 / (mean_correlation + epsilon))
        Where mean_correlation is calculated from the correlation matrix of instrument returns.
        
        Steps:
        1. Build correlation matrix of instrument returns
        2. Calculate mean correlation: mean(|rho_ij|) for i != j
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
        
        # Build correlation matrix
        corr_matrix = instrument_returns.corr()
        
        # Floor negative correlations at zero (Carver's recommendation)
        corr_matrix = corr_matrix.clip(lower=0.0)
        
        # Calculate mean correlation (excluding diagonal)
        # Get upper triangle (excluding diagonal) and calculate mean
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()
        
        if len(correlations) == 0:
            # No correlations to calculate (shouldn't happen with 2+ instruments)
            self.mean_return_correlation_ = 1.0
            return 1.0
        
        mean_correlation = correlations.mean()
        self.mean_return_correlation_ = mean_correlation
        
        # Calculate IDM: sqrt(1 / (mean_correlation + epsilon))
        epsilon = 0.01  # Small epsilon to avoid division by zero
        idm = np.sqrt(1.0 / (mean_correlation + epsilon))
        
        # Cap at idm_max
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
            if not returns_df.empty:
                # Calculate IDM from correlations
                self.fit(returns_df)
                # If IDM wasn't calculated (e.g., insufficient data), default to 1.0
                if self.idm_ is None:
                    self.idm_ = 1.0
                    self.mean_return_correlation_ = 1.0
        
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
        ensemble_predictions_list = []
        ensemble_predictions_dict = {}
        base_model_predictions_dict = {}
        
        # Get predictions from all ensembles
        for ensemble_idx, ensemble in enumerate(self.ensembles):
            # Use the new predict_from_candles method if available
            if hasattr(ensemble, 'predict_from_candles'):
                # Determine if we need base model predictions
                need_base_models = return_base_model_predictions
                
                if need_base_models:
                    # Get ensemble predictions with base model granularity
                    ensemble_result = ensemble.predict_from_candles(
                        tf_candles,
                        volatility=volatility,  # Pass volatility for proper scaling
                        return_base_model_predictions=True
                    )
                    
                    if isinstance(ensemble_result, dict):
                        ensemble_pred = ensemble_result.get('ensemble')
                        base_models = ensemble_result.get('base_models', {})
                        
                        # Store base model predictions with ensemble prefix
                        for model_name, model_pred in base_models.items():
                            ensemble_name = f"ensemble_{ensemble_idx}"
                            full_model_name = f"{ensemble_name}::{model_name}"
                            # Convert to position fractions (forecasts already volatility-adjusted)
                            base_model_positions = self._apply_risk_management_to_forecasts(
                                model_pred, volatility, tf_candles
                            )
                            base_model_predictions_dict[full_model_name] = base_model_positions
                    else:
                        ensemble_pred = ensemble_result
                else:
                    ensemble_pred = ensemble.predict_from_candles(tf_candles, volatility=volatility)
                
                # Always append the ensemble prediction DataFrame (not dict) for aggregation
                if isinstance(ensemble_pred, pd.DataFrame):
                    ensemble_predictions_list.append(ensemble_pred)
                else:
                    # If we got a dict but didn't request base models, extract ensemble
                    if isinstance(ensemble_pred, dict):
                        ensemble_pred = ensemble_pred.get('ensemble', pd.DataFrame())
                        ensemble_predictions_list.append(ensemble_pred)
                
                # Store ensemble-level predictions if requested
                if return_ensemble_predictions:
                    ensemble_name = f"ensemble_{ensemble_idx}"
                    # Get the ensemble prediction (might be from dict or direct)
                    if isinstance(ensemble_pred, dict):
                        ensemble_pred_df = ensemble_pred.get('ensemble')
                    else:
                        ensemble_pred_df = ensemble_pred
                    # Convert to position fractions
                    ensemble_positions = self._apply_risk_management_to_forecasts(
                        ensemble_pred_df, volatility, tf_candles
                    )
                    ensemble_predictions_dict[ensemble_name] = ensemble_positions
            else:
                # Fallback: use regular predict (requires features, not implemented here)
                pass
        
        if not ensemble_predictions_list:
            empty_df = pd.DataFrame(columns=['ticker', 'datetime', 'forecast_score', 'position_fraction'])
            if return_ensemble_predictions or return_base_model_predictions:
                result = {'portfolio': empty_df}
                if return_ensemble_predictions:
                    result['ensembles'] = {}
                if return_base_model_predictions:
                    result['base_models'] = {}
                return result
            return empty_df
        
        # Aggregate ensemble predictions for portfolio-level
        forecast_scores_df = self._aggregate_ensembles(ensemble_predictions_list, tf_candles)
        
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
        import logging
        
        logger = logging.getLogger(__name__)
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
        
        for ticker in candles_df['ticker'].unique():
            ticker_candles = candles_df[candles_df['ticker'] == ticker].copy()
            ticker_candles = ticker_candles.sort_values('datetime')
            ticker_candles['returns'] = ticker_candles['close'].pct_change()
            
            # Set datetime as index
            ticker_candles = ticker_candles.set_index('datetime')
            returns_dict[ticker] = ticker_candles['returns']
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        return returns_df
    
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
            
            # Forecasts are already volatility-adjusted from Ensemble
            # No need to apply volatility scaling again
            # Just apply IDM (Instrument Diversification Multiplier)
            # Use calculated idm_ if available, otherwise default to 1.0
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
            
            # Forecasts are already volatility-adjusted from Ensemble
            # No need to apply volatility scaling again
            # Just apply IDM (Instrument Diversification Multiplier)
            # Use calculated idm_ if available, otherwise default to 1.0
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
            - n_instruments: Number of instruments
            - instruments: List of instrument tickers
            - idm_max: Maximum allowed IDM
            - max_position_pct: Position cap (if any)
        """
        return {
            'is_fitted': self.is_fitted_,
            'idm': self.idm_,
            'mean_return_correlation': self.mean_return_correlation_,
            'n_instruments': len(self.instruments_) if self.instruments_ else 0,
            'instruments': self.instruments_,
            'idm_max': self.idm_max,
            'max_position_pct': self.max_position_pct,
            'trading_timeframe': self.trading_timeframe.name if self.trading_timeframe else None
        }

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
