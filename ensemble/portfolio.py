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
from typing import Dict, List, Optional
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
        weight_layer: Optional[WeightLayer] = None,
        trading_timeframe: TimeFrame = TimeFrame.D,
        max_position_pct: Optional[float] = None,
        instrument_weights: Optional[Dict[str, float]] = None,
        idm_max: float = 2.5
    ):
        """
        Initialize Portfolio.

        Parameters
        ----------
        weight_layer : WeightLayer, optional
            Weight layer for combining forecasts. If None, Portfolio can still
            be used for instrument weighting and IDM calculation.
        trading_timeframe : TimeFrame, default=TimeFrame.D
            The timeframe this portfolio trades on
        max_position_pct : float, optional
            Maximum position size per instrument (e.g., 2.0 = 200%)
        instrument_weights : Dict[str, float], optional
            Custom weights per instrument. If None, equal weight.
        idm_max : float, default=2.5
            Maximum IDM value (Carver's recommendation)
        """
        self.weight_layer = weight_layer
        self.trading_timeframe = trading_timeframe
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
        # Handle single instrument case
        if len(instrument_returns.columns) <= 1:
            self.mean_return_correlation_ = 1.0
            return 1.0

        # Calculate correlation matrix of instrument returns
        corr_matrix = instrument_returns.corr().abs()

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()

        if len(correlations) == 0:
            self.mean_return_correlation_ = 0.5
            return min(np.sqrt(1.0 / 0.51), self.idm_max)

        # Calculate mean correlation
        mean_corr = correlations.mean()

        # Floor negative correlations at zero (Carver's recommendation)
        mean_corr = max(mean_corr, 0.0)

        self.mean_return_correlation_ = mean_corr

        # Calculate IDM: sqrt(1 / (mean_corr + epsilon))
        epsilon = 0.01  # Small value to avoid division by zero
        idm = np.sqrt(1.0 / (mean_corr + epsilon))

        # Cap at idm_max
        return min(idm, self.idm_max)

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
