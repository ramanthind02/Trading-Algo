"""
PortfolioManager - Simple Wrapper for Multiple Portfolios

This module provides a PortfolioManager class that coordinates multiple Portfolio
instances (one per trading timeframe). Each Portfolio can be fitted and used for
prediction using DataFrames of candles, making experimentation and testing straightforward.

The PortfolioManager is a lightweight coordinator that:
1. Routes candles to appropriate portfolios by timeframe
2. Collects and combines outputs from all portfolios
3. Optionally converts position fractions to contracts via PositionSizer

Author: Trading Research Team
Date: 2025-01-10
"""

import logging
from typing import Dict, Optional

import pandas as pd

from execution.position_sizer import PositionSizer
from utils.enums import TimeFrame
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Simple wrapper class for managing multiple portfolios.
    
    PortfolioManager coordinates multiple Portfolio instances (one per trading timeframe).
    Each Portfolio can be fitted and used for prediction using DataFrames of candles.
    
    Parameters
    ----------
    portfolios : Dict[TimeFrame, Portfolio]
        Portfolio instances keyed by their trading timeframe.
        Example: {
            TimeFrame.D: daily_portfolio,
            TimeFrame.W: weekly_portfolio
        }
    position_sizer : PositionSizer, optional
        If provided, converts position fractions to contracts
        
    Attributes
    ----------
    portfolios : Dict[TimeFrame, Portfolio]
        Portfolio instances by timeframe
    position_sizer : PositionSizer or None
        Position sizer for contract conversion
        
    Examples
    --------
    >>> # Create portfolios
    >>> daily_portfolio = Portfolio(
    ...     ensembles=[ensemble1, ensemble2],
    ...     trading_timeframe=TimeFrame.D,
    ...     target_volatility=0.20,
    ...     dm=2.0
    ... )
    >>> 
    >>> weekly_portfolio = Portfolio(
    ...     ensembles=[ensemble3],
    ...     trading_timeframe=TimeFrame.W,
    ...     target_volatility=0.15,
    ...     dm=1.5
    ... )
    >>> 
    >>> # Create manager
    >>> manager = PortfolioManager(
    ...     portfolios={
    ...         TimeFrame.D: daily_portfolio,
    ...         TimeFrame.W: weekly_portfolio
    ...     },
    ...     position_sizer=my_sizer
    ... )
    >>> 
    >>> # Fit and predict
    >>> manager.fit(candles_df, target_data)
    >>> positions = manager.predict(candles_df)
    """
    
    def __init__(
        self,
        portfolios: Dict[TimeFrame, Portfolio],
        position_sizer: Optional[PositionSizer] = None
    ):
        """
        Initialize PortfolioManager with multiple portfolios.
        
        Parameters
        ----------
        portfolios : Dict[TimeFrame, Portfolio]
            Portfolio instances keyed by their trading timeframe.
            Example: {
                TimeFrame.D: daily_portfolio,
                TimeFrame.W: weekly_portfolio
            }
        position_sizer : PositionSizer, optional
            If provided, converts position fractions to contracts
        """
        if not portfolios:
            raise ValueError("portfolios cannot be empty")
        
        self.portfolios: Dict[TimeFrame, Portfolio] = portfolios
        self.position_sizer: Optional[PositionSizer] = position_sizer
        
        logger.info(
            f"PortfolioManager initialized with {len(portfolios)} portfolios: "
            f"{[tf.name for tf in portfolios.keys()]}"
        )
    
    def fit(
        self,
        candles_df: pd.DataFrame,
        target_data: Optional[pd.Series] = None
    ) -> None:
        """
        Fit all portfolios using candles DataFrame.
        
        Routes candles to appropriate portfolio based on timeframe column.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
        target_data : pd.Series, optional
            Target values (returns) for training. If None, portfolios must be pre-fitted.
        """
        if candles_df.empty:
            logger.warning("candles_df is empty. Nothing to fit.")
            return
        
        # Validate required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'ticker', 'timeframe']
        missing_cols = set(required_cols) - set(candles_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Group candles by timeframe and route to appropriate portfolio
        for tf, portfolio in self.portfolios.items():
            tf_candles = candles_df[candles_df['timeframe'] == tf].copy()
            
            if len(tf_candles) > 0:
                try:
                    # Use new fit_from_candles method if available
                    if hasattr(portfolio, 'fit_from_candles'):
                        portfolio.fit_from_candles(tf_candles, target_data)
                    else:
                        # Fallback: use regular fit (requires different input format)
                        logger.warning(
                            f"Portfolio for {tf.name} does not support fit_from_candles(). "
                            f"Skipping fit for this portfolio."
                        )
                except Exception as e:
                    logger.error(
                        f"Error fitting portfolio for {tf.name}: {e}",
                        exc_info=True
                    )
        
        logger.info("PortfolioManager fit completed")
    
    def predict(
        self,
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate positions from all portfolios using candles DataFrame.
        
        Routes candles to appropriate portfolios, collects outputs, and optionally
        converts to contracts.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
            
        Returns
        -------
        pd.DataFrame
            Combined positions from all portfolios with columns:
            - ticker, datetime, timeframe, forecast_score, position_fraction
            - contracts, notional_value, notional_pct (if position_sizer enabled)
        """
        if candles_df.empty:
            return pd.DataFrame(columns=[
                'ticker', 'datetime', 'timeframe', 'forecast_score', 'position_fraction'
            ])
        
        # Validate required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'ticker', 'timeframe']
        missing_cols = set(required_cols) - set(candles_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        all_positions = []
        
        # Get predictions from each portfolio
        for tf, portfolio in self.portfolios.items():
            tf_candles = candles_df[candles_df['timeframe'] == tf].copy()
            
            if len(tf_candles) > 0:
                try:
                    # Use new predict_from_candles method if available
                    if hasattr(portfolio, 'predict_from_candles'):
                        positions_df = portfolio.predict_from_candles(tf_candles)
                        positions_df['timeframe'] = tf
                        all_positions.append(positions_df)
                    else:
                        # Fallback: use regular predict (requires different input format)
                        logger.warning(
                            f"Portfolio for {tf.name} does not support predict_from_candles(). "
                            f"Skipping prediction for this portfolio."
                        )
                except Exception as e:
                    logger.error(
                        f"Error predicting with portfolio for {tf.name}: {e}",
                        exc_info=True
                    )
        
        # Combine all positions
        if not all_positions:
            return pd.DataFrame(columns=[
                'ticker', 'datetime', 'timeframe', 'forecast_score', 'position_fraction'
            ])
        
        combined_df = pd.concat(all_positions, ignore_index=True)
        
        # Optionally convert to contracts
        if self.position_sizer:
            try:
                contracts_df = self.position_sizer.calculate_positions(combined_df)
                return contracts_df
            except Exception as e:
                logger.error(
                    f"PositionSizer failed: {e}. Returning fractions only.",
                    exc_info=True
                )
                return combined_df
        
        return combined_df
    
    def get_portfolio(self, timeframe: TimeFrame) -> Optional[Portfolio]:
        """
        Get portfolio for a specific timeframe.
        
        Parameters
        ----------
        timeframe : TimeFrame
            Trading timeframe
            
        Returns
        -------
        Portfolio or None
            Portfolio instance for the timeframe, or None if not found
        """
        return self.portfolios.get(timeframe)
    
    def __repr__(self) -> str:
        """String representation of the portfolio manager."""
        timeframes = [tf.name for tf in self.portfolios.keys()]
        return f"PortfolioManager(portfolios={timeframes}, has_sizer={self.position_sizer is not None})"
    
    def __str__(self) -> str:
        """Detailed string description of the portfolio manager."""
        lines = [
            "PortfolioManager",
            f"  Portfolios: {len(self.portfolios)}",
            f"  Timeframes: {[tf.name for tf in self.portfolios.keys()]}",
            f"  Position Sizer: {'Enabled' if self.position_sizer else 'Disabled'}"
        ]
        return "\n".join(lines)
