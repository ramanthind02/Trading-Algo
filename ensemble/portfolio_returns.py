from __future__ import annotations

from typing import Dict

import pandas as pd

from .ensemble_utils import normalize_candles_datetime_column, normalize_ticker_key
from .weight_layer import _correlation_multiplier_from_corr_matrix


def _ticker_return_series(ticker_candles: pd.DataFrame) -> pd.Series:
    """Build a normalized daily return series for one ticker."""
    ordered = ticker_candles.copy()
    ordered["datetime"] = pd.to_datetime(ordered["datetime"])
    ordered = ordered.sort_values("datetime")
    ordered["returns"] = ordered["close"].pct_change()
    returns = ordered.set_index("datetime")["returns"].dropna()
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    returns.index = returns.index.normalize()
    return returns


def _filter_returns_for_idm(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with enough overlapping ticker returns for correlation estimation."""
    if len(returns_df.columns) < 2:
        return returns_df
    return returns_df.dropna(thresh=2)


def calculate_returns_from_candles(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Convert candles into a ticker-column return matrix used for IDM fitting."""
    normalized_candles = normalize_candles_datetime_column(candles_df)

    returns_dict: Dict[str, pd.Series] = {}
    for ticker in normalized_candles["ticker"].unique():
        ticker_candles = normalized_candles[normalized_candles["ticker"] == ticker].copy()
        ticker_returns = _ticker_return_series(ticker_candles)
        if ticker_returns.empty:
            continue
        returns_dict[normalize_ticker_key(ticker)] = ticker_returns

    if not returns_dict:
        return pd.DataFrame()

    returns_df = pd.DataFrame(returns_dict)
    return _filter_returns_for_idm(returns_df)


def calculate_idm_from_returns(
    instrument_returns: pd.DataFrame,
    idm_max: float,
) -> tuple[float, float]:
    """Calculate mean correlation and IDM from an instrument return matrix."""
    if instrument_returns.empty or len(instrument_returns.columns) < 2:
        return 1.0, 1.0

    corr_matrix = instrument_returns.corr()
    return _correlation_multiplier_from_corr_matrix(
        corr_matrix,
        cap=idm_max,
    )
