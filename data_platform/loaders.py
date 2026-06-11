"""Unified bar-data loaders for the data platform.

Single home for the read-side of the canonical stores:
  - load_data(ticker, tf)              futures + TLT, keyed by the Ticker enum
  - load_data_multi_ticker(tickers,tf) many tickers stacked with a ticker column
  - load_stock_data(symbol, tf, adj)   Norgate US stocks, keyed by string symbol

Both stores return a DataFrame indexed by unix timestamp (int64) with a
``datetime`` column; float32/int32 on disk are upcast to float64/int64 for
pipeline compatibility. Accepted on-disk schemas: a plain ``date`` column, or
a ``date``/``datetime``-named index (DatetimeIndex or string-named). The legacy
``datetime``-column schema has been removed (no production file uses it).

``utils.core.helpers`` and ``utils.core.stock_helpers`` re-export these so existing
imports keep working; new code should import from ``data_platform.loaders``.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from data_platform.providers.norgate.stocks import StockAdjustment, price_path
from lib.core.enums import Ticker, TimeFrame


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[1],
    )


def ohlc_data_dir() -> Path:
    return _repo_root() / "data" / "ohlc_data"


# In-process cache for OHLC parquet reads (same ticker, timeframe, range, file).
# Feature research and permutation call the loaders many times per run with
# identical arguments; the bias-node parquet cache is separate.
_LOAD_DATA_CACHE: dict[tuple[str, int, int, int], pd.DataFrame] = {}


def _load_data_cache_key(
    file_path: Path, start: datetime, end: datetime, *, mtime_ns: int,
) -> tuple[str, int, int, int]:
    return (
        str(file_path.resolve()),
        int(pd.Timestamp(start).value),
        int(pd.Timestamp(end).value),
        mtime_ns,
    )


def _nautilus_research_candles_enabled() -> bool:
    """Whether to route candle loads through the Nautilus research adapter.

    Controlled by the ``NAUTILUS_RESEARCH_CANDLES`` env var (``"1"`` = ON).
    Default OFF -> legacy provider-parquet path (no behaviour change). When ON,
    ``load_data`` / ``load_data_multi_ticker`` delegate to
    ``data_platform.nautilus.candles`` (WP-2 Unit-2, Option B).
    """
    return os.environ.get("NAUTILUS_RESEARCH_CANDLES") == "1"


def _normalize_loaded_frame(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Shared schema handling for both the futures and stock stores."""
    if "date" in df.columns:
        # Insert at position 0 to match the index-restoring branch below, so
        # date32-contract files (plain 'date' column) and legacy files (date as
        # pandas index) produce identical column order.
        datetimes = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"]).copy()
        df.insert(0, "datetime", datetimes)
    elif isinstance(df.index, pd.DatetimeIndex) or (
        hasattr(df.index, "name") and df.index.name in ("date", "datetime")
    ):
        df = df.reset_index()
        col = "date" if "date" in df.columns else df.columns[0]
        df = df.rename(columns={col: "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].astype("float64")
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype("int64")

    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    df = df[mask].copy()

    df["timestamp"] = df["datetime"].astype("int64") // 10**9
    return df.set_index("timestamp").sort_index()


def load_data(
    ticker: Ticker,
    timeframe: TimeFrame,
    start: datetime = datetime(1990, 1, 1),
    end: datetime = datetime(2099, 12, 31),
) -> pd.DataFrame:
    """Load OHLC data for one Ticker from ``data/ohlc_data/{TICKER}/``.

    Returns a DataFrame indexed by unix timestamp (int64) with a ``datetime``
    column. Handles both legacy and current parquet schemas.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    from lib.core.research_feed import LEGACY_FEED, feed_for_ticker

    _feed = feed_for_ticker(ticker)
    if _feed != LEGACY_FEED:
        # Non-futures research feed (cfd / spliced). Reuse _normalize_loaded_frame
        # for an output shape identical to the futures path (timestamp index,
        # float64 OHLC, int64 volume, start/end filtered).
        from data_platform.providers.mt5.cfd_candles import load_research_candles_raw

        raw = load_research_candles_raw(ticker, timeframe, _feed)
        return _normalize_loaded_frame(raw, start, end)

    if _nautilus_research_candles_enabled():
        from data_platform.nautilus.candles import load_data_nautilus

        return load_data_nautilus(ticker, timeframe, start=start, end=end)

    file_path = ohlc_data_dir() / ticker.name / f"{timeframe.name}_{ticker.name}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    mtime_ns = file_path.stat().st_mtime_ns
    cache_key = _load_data_cache_key(file_path, start, end, mtime_ns=mtime_ns)
    cached = _LOAD_DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    df = pd.read_parquet(file_path, engine="pyarrow")
    df = _normalize_loaded_frame(df, start, end)

    _LOAD_DATA_CACHE[cache_key] = df.copy()
    return df.copy()


def load_data_multi_ticker(
    tickers: List[Ticker],
    timeframe: TimeFrame,
    start: datetime = datetime(1990, 1, 1),
    end: datetime = datetime(2099, 12, 31),
    use_millisecond_offset: bool = False,
) -> pd.DataFrame:
    """Load multiple tickers and stack rows with ``ticker`` / ``timeframe`` columns.

    ``use_millisecond_offset`` is deprecated and ignored; the primary key is
    (datetime, ticker) with no per-ticker offset.
    """
    if _nautilus_research_candles_enabled():
        from data_platform.nautilus.candles import load_data_multi_ticker_nautilus

        return load_data_multi_ticker_nautilus(
            tickers,
            timeframe,
            start=start,
            end=end,
            use_millisecond_offset=use_millisecond_offset,
        )

    all_dfs = []
    for ticker in tickers:
        ticker_df = load_data(ticker, timeframe, start=start, end=end).reset_index()
        ticker_df["ticker"] = ticker
        ticker_df["timeframe"] = timeframe
        all_dfs.append(ticker_df)
    combined = pd.concat(all_dfs, axis=0, ignore_index=True)
    return combined.sort_values("datetime").reset_index(drop=True)


def load_stock_data(
    symbol: str,
    timeframe: TimeFrame,
    adjustment: StockAdjustment = StockAdjustment.TOTAL_RETURN,
    start: datetime = datetime(1990, 1, 1),
    end: datetime = datetime(2099, 12, 31),
) -> pd.DataFrame:
    """Load one stock's OHLCV from the partitioned ``data/stock_data/`` store.

    String-keyed (bypasses the futures ``Ticker`` enum). ``symbol`` is the raw
    Norgate symbol (``"AAPL"``, ``"AGM.A"``, ``"AABA-201910"`` for delisted).

    Raises:
        FileNotFoundError: If the requested symbol/timeframe/adjustment parquet is absent.
    """
    file_path: Path = price_path(symbol, timeframe.name, adjustment)
    if not file_path.exists():
        raise FileNotFoundError(
            f"No stock data for symbol={symbol!r} timeframe={timeframe.name} "
            f"adjustment={adjustment.value} at {file_path}"
        )
    df = pd.read_parquet(file_path)
    return _normalize_loaded_frame(df, start, end)
