"""Ingest raw research OHLC candles into the Nautilus ``ParquetDataCatalog``.

WP-2 Unit-2, Option B. For each ``(ticker, timeframe)`` this reads the **exact
same** provider parquet the legacy ``data_platform.loaders.load_data`` reads
(``data/ohlc_data/{TICKER}/{TF}_{TICKER}.parquet``), upcasts float32->float64 /
int32->int64 the way :func:`data_platform.loaders._normalize_loaded_frame` does,
and writes one :class:`ResearchCandle` per row to the catalog.

Byte-exactness contract
-----------------------
The stored ``open/high/low/close`` are ``float(np.float64(<on-disk float32>))``
-- i.e. the identical float64 the legacy normalizer produces via ``.astype
("float64")``. ``volume`` is ``int(np.int64(...))``. ``ts_event == ts_init`` is
the bar's tz-naive date-label as unix nanoseconds (``Timestamp.value``), so the
read-back adapter reconstructs the legacy ``datetime`` exactly.

TLT is a :class:`~lib.core.enums.Ticker` member and lives under the same
``data/ohlc_data/TLT/`` store, so no special stock-store path is required here --
it flows through the identical loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from nautilus_trader.persistence.catalog import ParquetDataCatalog

from data_platform.loaders import ohlc_data_dir
from data_platform.nautilus.catalog import get_catalog
from data_platform.nautilus.research_candle import (
    ResearchCandle,
    research_instrument_id,
)
from lib.core.enums import Ticker, TimeFrame


@dataclass(frozen=True)
class ResearchIngestResult:
    """Counts of candle records written for one ``(ticker, timeframe)``."""

    ticker: str
    timeframe: str
    instrument_id: str
    candles_written: int


def _ohlc_parquet_path(ticker: Ticker, timeframe: TimeFrame) -> Path:
    """The same path :func:`data_platform.loaders.load_data` reads."""
    return ohlc_data_dir() / ticker.name / f"{timeframe.name}_{ticker.name}.parquet"


def _read_provider_frame(path: Path) -> pd.DataFrame:
    """Read provider parquet exactly as the legacy loader does (fastparquet).

    Returns a frame with a tz-naive ``datetime`` column (the date-label) plus
    float64 OHLC and int64 volume -- mirroring the legacy normalizer's casts so
    stored bytes match what the loader yields downstream.
    """
    df = pd.read_parquet(path, engine="fastparquet")

    # Recover the date-label column regardless of on-disk schema (index `date`,
    # `datetime`/`date` column), matching `_normalize_loaded_frame` semantics.
    if "datetime" in df.columns:
        datetime_col = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        datetime_col = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.DatetimeIndex) or (
        getattr(df.index, "name", None) in ("date", "datetime")
    ):
        reset = df.reset_index()
        col = "date" if "date" in reset.columns else reset.columns[0]
        datetime_col = pd.to_datetime(reset[col])
        df = reset
    else:  # pragma: no cover - defensive; provider always carries a date label
        raise ValueError(f"No date label found in provider parquet: {path}")

    out = pd.DataFrame({"datetime": datetime_col.reset_index(drop=True)})
    for col in ("open", "high", "low", "close"):
        out[col] = df[col].to_numpy().astype(np.float64)
    out["volume"] = df["volume"].to_numpy().astype(np.int64)
    return out


def _frame_to_candles(
    df: pd.DataFrame, ticker_name: str, timeframe_name: str
) -> list[ResearchCandle]:
    iid = research_instrument_id(ticker_name, timeframe_name)
    # Date-label -> unix nanoseconds (tz-naive midnight for daily bars).
    ts_ns = df["datetime"].astype("int64").to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    volumes = df["volume"].to_numpy()
    return [
        ResearchCandle(
            instrument_id=iid,
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=int(volumes[i]),
            ts_event=int(ts_ns[i]),
            ts_init=int(ts_ns[i]),
        )
        for i in range(len(df))
    ]


def ingest_research_candles_for(
    ticker: Ticker,
    timeframe: TimeFrame,
    catalog: ParquetDataCatalog,
) -> ResearchIngestResult:
    """Ingest one ``(ticker, timeframe)`` series into *catalog*."""
    path = _ohlc_parquet_path(ticker, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")

    frame = _read_provider_frame(path)
    candles = _frame_to_candles(frame, ticker.name, timeframe.name)
    if candles:
        catalog.write_data(candles)

    return ResearchIngestResult(
        ticker=ticker.name,
        timeframe=timeframe.name,
        instrument_id=str(research_instrument_id(ticker.name, timeframe.name)),
        candles_written=len(candles),
    )


def ingest_research_candles(
    tickers: Iterable[Ticker],
    timeframes: Iterable[TimeFrame],
    catalog: ParquetDataCatalog | None = None,
) -> list[ResearchIngestResult]:
    """Ingest ``ResearchCandle`` records for every ``(ticker, timeframe)`` pair.

    Skips ``(ticker, timeframe)`` pairs whose provider parquet is absent (some
    tickers lack every timeframe) and records nothing for them.
    """
    cat = catalog if catalog is not None else get_catalog()
    results: list[ResearchIngestResult] = []
    for ticker in tickers:
        for timeframe in timeframes:
            path = _ohlc_parquet_path(ticker, timeframe)
            if not path.exists():
                continue
            results.append(ingest_research_candles_for(ticker, timeframe, cat))
    return results
