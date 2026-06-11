"""Norgate market-series scraper — indices, cash commodities, forex spot.

These Norgate databases are flat single-series instruments (no back-adjustment,
no expiries, no stock price-adjustment variants), so they need neither the
futures contract machinery (``fetch_contracts``) nor the stock multi-adjustment
machinery (``stocks``). One ``price_timeseries`` call per symbol, stored as-is.

Captured databases (all of which a US Stocks+Futures subscription exposes):

    US Indices        (~1,615)  e.g. $SPX, $RUI, sector/industry indices
    World Indices     (~31)     e.g. $DAX, $N225, $FT100
    Cash Commodities  (~100)    e.g. $BCOM family, $GSR, spot metals ratios
    Forex Spot        (~57)     e.g. EURUSD, USDJPY, $USDX, XAUUSD

Storage layout
--------------
    data/norgate/market_series/
      {CATEGORY}/                  us_indices | world_indices | cash_commodities | forex_spot
        {SAFE_SYMBOL}.parquet      daily OHLC (+ Volume/Turnover where present)

``{SAFE_SYMBOL}`` renders Norgate symbols filesystem-safe (``$SPX`` -> ``_SPX``,
``$BCOMAG`` -> ``_BCOMAG``); the original symbol is stamped into parquet metadata
under ``norgate_raw_symbol`` so it round-trips losslessly.

Schema (per file)
-----------------
    date    date32 (index, name "date")
    open    float32
    high    float32
    low     float32
    close   float32
    volume  int64    (only when Norgate returns Volume; indices)
    turnover float64 (only when Norgate returns Turnover; indices)

Incremental: a symbol whose parquet already exists is skipped (``--overwrite``
refetches). Per-symbol errors are caught and counted; the run continues.

CLI
---
    python -m data_platform.providers.norgate.market_series                # all four categories
    python -m data_platform.providers.norgate.market_series --category forex_spot
    python -m data_platform.providers.norgate.market_series --category us_indices --limit 50
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    import norgatedata
except ImportError:
    norgatedata = None  # type: ignore[assignment]
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ._constants import PARQUET_COMPRESSION, PARQUET_COMPRESSION_LEVEL
from ._paths import norgate_root
from .fetch_continuous import ensure_norgate_running


class MarketCategory(Enum):
    """A flat Norgate market-series database and its on-disk folder name."""

    US_INDICES = ("us_indices", "US Indices")
    WORLD_INDICES = ("world_indices", "World Indices")
    CASH_COMMODITIES = ("cash_commodities", "Cash Commodities")
    FOREX_SPOT = ("forex_spot", "Forex Spot")

    def __init__(self, folder: str, database: str) -> None:
        self.folder = folder
        self.database = database


_RAW_SYMBOL_META_KEY = b"norgate_raw_symbol"


# ── paths ────────────────────────────────────────────────────────────────────


def market_series_root() -> Path:
    return norgate_root() / "market_series"


def safe_symbol(symbol: str) -> str:
    """Render a Norgate symbol as a filesystem-safe token (``$SPX`` -> ``_SPX``)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_") or "_"


def series_path(category: MarketCategory, symbol: str) -> Path:
    return market_series_root() / category.folder / f"{safe_symbol(symbol)}.parquet"


# ── result type ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SeriesResult:
    symbol: str
    rows: int
    start: str
    end: str


@dataclass(frozen=True)
class CategorySummary:
    category: str
    requested: int
    written: int
    skipped_existing: int
    errored: int


# ── normalisation ─────────────────────────────────────────────────────────


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Norgate price frame -> canonical daily schema, preserving optional columns.

    OHLC are float32; Volume (when present) is int64 (index volumes overflow
    int32); Turnover (when present) is float64.
    """
    out = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Turnover": "turnover",
        }
    ).copy()
    out.index = pd.to_datetime(out.index).normalize().tz_localize(None)
    out.index.name = "date"
    out = out.sort_index().loc[~out.index.duplicated(keep="last")]

    keep = ["open", "high", "low", "close"]
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype("float32")
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0).astype("int64")
        keep.append("volume")
    if "turnover" in out.columns:
        out["turnover"] = out["turnover"].astype("float64")
        keep.append("turnover")
    return out[keep]


def _write(df: pd.DataFrame, path: Path, raw_symbol: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata(
        {**existing, _RAW_SYMBOL_META_KEY: raw_symbol.encode("utf-8")}
    )
    pq.write_table(
        table, path, compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


def read_raw_symbol(path: Path) -> str | None:
    """Recover the original Norgate symbol stamped into a market-series file."""
    meta = pq.read_schema(path).metadata or {}
    value = meta.get(_RAW_SYMBOL_META_KEY)
    return value.decode("utf-8") if value is not None else None


# ── fetch ────────────────────────────────────────────────────────────────────


def fetch_symbol(category: MarketCategory, symbol: str) -> SeriesResult | None:
    try:
        df = norgatedata.price_timeseries(
            symbol,
            interval="D",
            timeseriesformat="pandas-dataframe",
            padding_setting=norgatedata.PaddingType.NONE,
        )
    except Exception as exc:
        print(f"    ERR {symbol}: {exc}")
        return None
    if df is None or df.empty:
        return None
    daily = _normalize(df)
    if daily.empty:
        return None
    _write(daily, series_path(category, symbol), raw_symbol=symbol)
    return SeriesResult(
        symbol=symbol,
        rows=len(daily),
        start=str(daily.index.min().date()),
        end=str(daily.index.max().date()),
    )


def already_fetched(category: MarketCategory, symbol: str) -> bool:
    return series_path(category, symbol).exists()


def scrape_category(
    category: MarketCategory, *, overwrite: bool = False, limit: int | None = None
) -> CategorySummary:
    symbols = sorted(norgatedata.database_symbols(category.database))
    if limit is not None:
        symbols = symbols[:limit]

    print(f"\n=== {category.database} ({len(symbols)} symbols) -> "
          f"{market_series_root() / category.folder} ===")

    written = skipped = errored = 0
    for i, symbol in enumerate(symbols, start=1):
        if not overwrite and already_fetched(category, symbol):
            skipped += 1
            continue
        result = fetch_symbol(category, symbol)
        if result is None:
            errored += 1
            continue
        written += 1
        if written % 50 == 0 or len(symbols) <= 60:
            print(f"  [{i}/{len(symbols)}] OK {symbol:<12} "
                  f"rows={result.rows:>6} ({result.start}..{result.end})")
    summary = CategorySummary(
        category=category.folder, requested=len(symbols),
        written=written, skipped_existing=skipped, errored=errored,
    )
    print(f"  -> written={summary.written} skipped={summary.skipped_existing} "
          f"errored={summary.errored}")
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape Norgate market series (indices, cash commodities, forex spot)."
    )
    p.add_argument(
        "--category",
        choices=[c.folder for c in MarketCategory],
        default=None,
        help="Single category to scrape (default: all four).",
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Cap symbols per category (for testing).")
    p.add_argument("--overwrite", action="store_true",
                   help="Refetch symbols even if their files already exist.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    ensure_norgate_running()

    if args.category:
        categories = [c for c in MarketCategory if c.folder == args.category]
    else:
        categories = list(MarketCategory)

    summaries = [
        scrape_category(c, overwrite=args.overwrite, limit=args.limit)
        for c in categories
    ]
    total_written = sum(s.written for s in summaries)
    total_err = sum(s.errored for s in summaries)
    print(f"\nDone. {len(categories)} categories, "
          f"{total_written} series written, {total_err} errored.")


if __name__ == "__main__":
    main()
