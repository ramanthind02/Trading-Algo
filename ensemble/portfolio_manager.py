"""Lightweight coordinator for multiple timeframe-specific portfolios."""

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Dict, Optional

import pandas as pd

from execution.position_sizer import PositionSizer
from ensemble.portfolio import Portfolio, PortfolioCacheQuery
from utils.core.enums import TimeFrame

logger = logging.getLogger(__name__)

_REQUIRED_CANDLE_COLUMNS = (
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "ticker",
    "timeframe",
)
_POSITION_COLUMNS = (
    "ticker",
    "datetime",
    "timeframe",
    "forecast_score",
    "position_fraction",
)
_UNSUPPORTED = object()


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

    @staticmethod
    def _empty_positions() -> pd.DataFrame:
        return pd.DataFrame(columns=list(_POSITION_COLUMNS))

    @staticmethod
    def _validate_candles_df(candles_df: pd.DataFrame) -> None:
        missing_cols = set(_REQUIRED_CANDLE_COLUMNS) - set(candles_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _iter_timeframe_candles(
        self,
        candles_df: pd.DataFrame,
    ) -> Iterator[tuple[TimeFrame, Portfolio, pd.DataFrame]]:
        self._validate_candles_df(candles_df)
        for timeframe, portfolio in self.portfolios.items():
            timeframe_candles = candles_df[candles_df["timeframe"] == timeframe].copy()
            if not timeframe_candles.empty:
                yield timeframe, portfolio, timeframe_candles

    def _combine_positions(
        self,
        positions: Iterable[tuple[TimeFrame, pd.DataFrame]],
    ) -> pd.DataFrame:
        frames = [
            frame.assign(timeframe=timeframe)
            for timeframe, frame in positions
            if isinstance(frame, pd.DataFrame) and not frame.empty
        ]
        if not frames:
            return self._empty_positions()

        combined_df = pd.concat(frames, ignore_index=True)
        if self.position_sizer is None:
            return combined_df
        try:
            return self.position_sizer.calculate_positions(combined_df)
        except Exception as exc:
            logger.error(
                "PositionSizer failed: %s. Returning fractions only.",
                exc,
                exc_info=True,
            )
            return combined_df

    @staticmethod
    def _call_if_supported(
        portfolio: Portfolio,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any | None:
        method = getattr(portfolio, method_name, None)
        if not callable(method):
            return _UNSUPPORTED
        return method(*args, **kwargs)

    @staticmethod
    def _log_unsupported_method(timeframe: TimeFrame, method_name: str) -> None:
        logger.warning(
            "Portfolio for %s does not support %s(). Skipping.",
            timeframe.name,
            method_name,
        )

    @staticmethod
    def _safe_portfolio_call(
        timeframe: TimeFrame,
        action: str,
        fn: Callable[[], Any],
    ) -> Any | None:
        try:
            return fn()
        except Exception as exc:
            logger.error(
                "Error %s portfolio for %s: %s",
                action,
                timeframe.name,
                exc,
                exc_info=True,
            )
            return None
    
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

        for timeframe, portfolio, timeframe_candles in self._iter_timeframe_candles(candles_df):
            call = lambda: self._call_if_supported(
                portfolio,
                "fit_from_candles",
                timeframe_candles,
                target_data,
            )
            result = self._safe_portfolio_call(timeframe, "fitting", call)
            if result is _UNSUPPORTED:
                self._log_unsupported_method(timeframe, "fit_from_candles")

        logger.info("PortfolioManager fit completed")

    def fit_from_cache(
        self,
        query: PortfolioCacheQuery,
        target_data: Optional[pd.Series] = None,
    ) -> None:
        """Fit portfolios from cache-native query inputs."""
        for tf, portfolio in self.portfolios.items():
            if hasattr(portfolio, "fit_from_cache"):
                portfolio.fit_from_cache(query.for_timeframe(tf), target_data=target_data)

    def predict_from_cache(self, query: PortfolioCacheQuery) -> pd.DataFrame:
        """Predict from cache-native query inputs."""
        return self._combine_positions(
            (
                timeframe,
                self._safe_portfolio_call(
                    timeframe,
                    "predicting from cache",
                    lambda timeframe=timeframe, portfolio=portfolio: self._call_if_supported(
                        portfolio,
                        "predict_from_cache",
                        query.for_timeframe(timeframe),
                    ),
                ),
            )
            for timeframe, portfolio in self.portfolios.items()
        )
    
    def predict(
        self,
        candles_df: pd.DataFrame,
        daily_volatility_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate positions from all portfolios using candles DataFrame.
        
        Routes candles to appropriate portfolios, collects outputs, and optionally
        converts to contracts.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe
        daily_volatility_df : pd.DataFrame, optional
            Daily EWSD volatility DataFrame with columns:
            ['datetime', 'ticker', 'ewsd_annual_vol'].
            
        Returns
        -------
        pd.DataFrame
            Combined positions from all portfolios with columns:
            - ticker, datetime, timeframe, forecast_score, position_fraction
            - contracts, notional_value, notional_pct (if position_sizer enabled)
        """
        if candles_df.empty:
            return self._empty_positions()

        volatility_df = pd.DataFrame() if daily_volatility_df is None else daily_volatility_df
        positions = []
        for timeframe, portfolio, timeframe_candles in self._iter_timeframe_candles(candles_df):
            result = self._safe_portfolio_call(
                timeframe,
                "predicting with",
                lambda timeframe_candles=timeframe_candles, portfolio=portfolio: self._call_if_supported(
                    portfolio,
                    "predict_from_candles",
                    timeframe_candles,
                    daily_volatility_df=volatility_df,
                ),
            )
            if result is _UNSUPPORTED:
                self._log_unsupported_method(timeframe, "predict_from_candles")
                continue
            positions.append((timeframe, result))

        return self._combine_positions(positions)
    
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
