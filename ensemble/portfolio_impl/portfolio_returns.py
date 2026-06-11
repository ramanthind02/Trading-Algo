from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import pandas as pd

from ensemble.ensemble_utils import normalize_candles_datetime_column, normalize_ticker_key
from ensemble.weight_layer import _correlation_multiplier_from_corr_matrix


@lru_cache(maxsize=128)
def _load_ratio_close_series(ticker: str) -> Optional[pd.Series]:
    """Load the RATIO (proportional) back-adjusted close for *ticker*.

    The ratio series preserves %-returns exactly within each contract and never
    goes negative, so ``close.pct_change()`` on it gives faithful daily returns
    for the IDM/weight correlation estimate (additive back-adjustment deflates
    deep-history %-returns and can flip sign on crude). Returns a date-indexed
    Series, or ``None`` when the file does not exist (non-futures, or
    pre-regeneration state) — callers then fall back to the candle close.
    """
    try:
        from cache.runtime.cache_paths import project_root as _project_root

        path = (
            _project_root()
            / "data"
            / "ohlc_data"
            / ticker
            / f"D_{ticker}_ratio.parquet"
        )
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if "date" in df.columns:
            idx = pd.to_datetime(df["date"])
        elif "datetime" in df.columns:
            idx = pd.to_datetime(df["datetime"])
        else:
            idx = pd.to_datetime(df.index)
        close = pd.Series(df["close"].astype("float64").to_numpy(), index=idx.normalize())
        return close[~close.index.duplicated(keep="last")].sort_index()
    except Exception:
        # Any read/parse problem must NOT break IDM fitting — fall back silently.
        return None


def _ticker_return_series(ticker_candles: pd.DataFrame) -> pd.Series:
    """Build a normalized daily return series for one ticker.

    Uses the RATIO back-adjusted close (faithful %-returns) when that series is
    available on disk, aligned to the candle dates; otherwise falls back to
    ``candle.close.pct_change()`` (the additive series carried on the candles).
    """
    ordered = ticker_candles.copy()
    ordered["datetime"] = pd.to_datetime(ordered["datetime"])
    ordered = ordered.sort_values("datetime")

    ratio_returns = _ratio_returns_for_candles(ordered)
    if ratio_returns is not None:
        return ratio_returns

    ordered["returns"] = ordered["close"].pct_change()
    returns = ordered.set_index("datetime")["returns"].dropna()
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    returns.index = returns.index.normalize()
    return returns


def _ratio_returns_for_candles(ordered: pd.DataFrame) -> Optional[pd.Series]:
    """Return ratio-close %-returns aligned to the candle dates, or None.

    Returns None (so the caller falls back to the additive close) when the
    ticker is unknown, the ratio file is missing, or the overlap with the candle
    window is too small to be the IDM denominator.
    """
    if "ticker" not in ordered.columns or ordered["ticker"].empty:
        return None
    ticker = str(ordered["ticker"].iloc[0])
    ratio_close = _load_ratio_close_series(ticker)
    if ratio_close is None or ratio_close.empty:
        return None

    dates = pd.DatetimeIndex(ordered["datetime"]).normalize()
    aligned = ratio_close.reindex(dates)
    # Require near-complete coverage of the candle window; otherwise the ratio
    # store is stale/partial for this ticker and the additive close is safer.
    if aligned.notna().mean() < 0.95:
        return None
    returns = aligned.pct_change().dropna()
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
