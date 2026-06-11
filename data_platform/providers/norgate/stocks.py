"""Norgate US-stock scraper — survivorship-bias-free, multi-adjustment, partitioned.

This adapter is a SEPARATE store from the futures pipeline. It does NOT touch the
``Ticker`` enum, ``data/ohlc_data/`` or ``bootstrap_source_candles``. Stocks are
keyed by their raw Norgate symbol string (e.g. ``AAPL``, ``AGM.A``, ``AABA-201910``
for delisted) and stored under ``data/stock_data/``.

Storage layout
--------------
Symbols are partitioned by their first alphanumeric character (uppercased) into a
"bucket" folder, because ~35k symbols in one directory is pathological on Windows::

    data/stock_data/
      {BUCKET}/                         first alnum char of symbol, else "_"
        {SAFE_SYMBOL}/                  filename-safe rendering of the symbol
          D_TR_{SAFE_SYMBOL}.parquet    daily, TOTAL_RETURN  (primary)
          W_TR_{SAFE_SYMBOL}.parquet    weekly, TOTAL_RETURN
          M_TR_{SAFE_SYMBOL}.parquet    monthly, TOTAL_RETURN
          D_CAP_{SAFE_SYMBOL}.parquet   daily, CAPITAL (splits + reconstructions)
          W_CAP_{SAFE_SYMBOL}.parquet   weekly, CAPITAL
          M_CAP_{SAFE_SYMBOL}.parquet   monthly, CAPITAL
          D_UNADJ_{SAFE_SYMBOL}.parquet daily, UNADJUSTED (raw traded close)
          W_UNADJ_{SAFE_SYMBOL}.parquet weekly, UNADJUSTED
          M_UNADJ_{SAFE_SYMBOL}.parquet monthly, UNADJUSTED
          membership_{SAFE_SYMBOL}.parquet  per-index 0/1 constituent timeseries

``{ADJ}`` tokens are ``TR`` (TOTAL_RETURN), ``CAP`` (CAPITAL), ``UNADJ`` (NONE).
``{SAFE_SYMBOL}`` replaces filesystem-hostile characters (``. - / \\ : space``)
with ``_`` so symbols like ``AGM.A`` or ``AABA-201910`` produce valid paths; the
*original* symbol is preserved as parquet metadata and is the loader key.

Price schema (every D/W/M file), matching the futures store exactly
------------------------------------------------------------------
    date    date32 (parquet native DATE)   ← index, name "date"
    open    float32
    high    float32
    low     float32
    close   float32
    volume  int32

The UNADJ series stores the raw traded close in all OHLC slots' close (Norgate's
adjusted frame only exposes ``Unadjusted Close``, not unadjusted O/H/L), so the
UNADJ open/high/low equal the unadjusted close; use it as the σ denominator / raw
price reference, not for intrabar range.

Membership schema
-----------------
    date    date32   ← index, name "date"
    {index_col}  int8 (0/1)   one column per scraped index (e.g. sp500, russell3000)

CLI
---
    python -m data_platform.providers.norgate.stocks --universe both
    python -m data_platform.providers.norgate.stocks --universe active --limit 20
    python -m data_platform.providers.norgate.stocks --watchlist "S&P 500" --limit 10

Incremental: a symbol whose D_TR file already exists is skipped (use --overwrite
to refetch). Symbols that error (delisted/access-denied/empty) are skipped and the
run continues; a summary count prints at the end.
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

from data_platform.storage import write_norgate_bars
from ._constants import PARQUET_COMPRESSION, PARQUET_COMPRESSION_LEVEL
from .fetch_continuous import ensure_norgate_running

# ── universe / adjustment vocab ───────────────────────────────────────────


class Universe(Enum):
    """Which Norgate equity database(s) to enumerate."""

    ACTIVE = "active"
    DELISTED = "delisted"
    BOTH = "both"


_UNIVERSE_DATABASES: dict[Universe, tuple[str, ...]] = {
    Universe.ACTIVE: ("US Equities",),
    Universe.DELISTED: ("US Equities Delisted",),
    Universe.BOTH: ("US Equities", "US Equities Delisted"),
}


class StockAdjustment(Enum):
    """Stored price-adjustment series for a stock and its on-disk file token."""

    TOTAL_RETURN = "TR"
    CAPITAL = "CAP"
    UNADJUSTED = "UNADJ"


_ADJ_TO_NORGATE = (
    {
        StockAdjustment.TOTAL_RETURN: norgatedata.StockPriceAdjustmentType.TOTALRETURN,
        StockAdjustment.CAPITAL: norgatedata.StockPriceAdjustmentType.CAPITAL,
        StockAdjustment.UNADJUSTED: norgatedata.StockPriceAdjustmentType.NONE,
    }
    if norgatedata is not None
    else {}
)

# Indices captured as per-stock membership timeseries. Column names are
# filesystem/identifier-safe slugs of the watchlist name.
INDEX_MEMBERSHIP_WATCHLISTS: tuple[str, ...] = (
    "S&P 500",
    "S&P MidCap 400",
    "S&P SmallCap 600",
    "Russell 1000",
    "Russell 2000",
    "Russell 3000",
    "Nasdaq 100",
    "Dow Jones Industrial Average",
)


# ── paths ──────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[3],
    )


def stock_data_root() -> Path:
    return _repo_root() / "data" / "stock_data"


def safe_symbol(symbol: str) -> str:
    """Render a Norgate symbol as a filesystem-safe token (``AGM.A`` -> ``AGM_A``)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_") or "_"


def symbol_bucket(symbol: str) -> str:
    """Partition bucket = first alphanumeric char (uppercased), else ``_``."""
    for ch in symbol:
        if ch.isalnum():
            return ch.upper()
    return "_"


def stock_dir(symbol: str) -> Path:
    return stock_data_root() / symbol_bucket(symbol) / safe_symbol(symbol)


def price_path(symbol: str, timeframe: str, adjustment: StockAdjustment) -> Path:
    safe = safe_symbol(symbol)
    return stock_dir(symbol) / f"{timeframe}_{adjustment.value}_{safe}.parquet"


def membership_path(symbol: str) -> Path:
    return stock_dir(symbol) / f"membership_{safe_symbol(symbol)}.parquet"


# ── result type ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StockFetchResult:
    symbol: str
    daily_rows: int
    start: str | None
    end: str | None
    indices: int  # number of membership columns written


# ── schema helpers ──────────────────────────────────────────────────────────


def _ohlcv_from_close(close: pd.Series) -> pd.DataFrame:
    """Build an OHLCV frame from a single close series (for the UNADJ series).

    Norgate's adjusted price frame exposes ``Unadjusted Close`` only, so the
    unadjusted O/H/L are set equal to the unadjusted close and volume to 0.
    """
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 0.0},
        index=close.index,
    )
    return frame


def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Norgate price frame -> canonical daily schema (date32 index, float32, int32)."""
    out = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    ).copy()
    out.index = pd.to_datetime(out.index).normalize().tz_localize(None)
    out.index.name = "date"
    if "volume" not in out.columns:
        out["volume"] = 0
    out = out[["open", "high", "low", "close", "volume"]].copy()
    out = out.sort_index().loc[~out.index.duplicated(keep="last")]
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype("float32")
    out["volume"] = out["volume"].fillna(0).astype("int32")
    return out


def _aggregate(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = (
        daily.resample(freq)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open"])
    )
    agg.index.name = "date"
    for col in ("open", "high", "low", "close"):
        agg[col] = agg[col].astype("float32")
    agg["volume"] = agg["volume"].astype("int32")
    return agg


_RAW_SYMBOL_META_KEY = b"norgate_raw_symbol"


def _write(df: pd.DataFrame, path: Path, *, raw_symbol: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=True)
    if raw_symbol is not None:
        # Stamp the original Norgate symbol into file-level metadata so the safe
        # filename (e.g. AGM_A) can be losslessly mapped back to the raw symbol.
        existing = table.schema.metadata or {}
        table = table.replace_schema_metadata(
            {**existing, _RAW_SYMBOL_META_KEY: raw_symbol.encode("utf-8")}
        )
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


def read_raw_symbol(path: Path) -> str | None:
    """Recover the original Norgate symbol stamped into a stock parquet file."""
    meta = pq.read_schema(path).metadata or {}
    value = meta.get(_RAW_SYMBOL_META_KEY)
    return value.decode("utf-8") if value is not None else None


# ── Norgate fetch ────────────────────────────────────────────────────────────


def _fetch_price_frame(
    symbol: str, adjustment: StockAdjustment
) -> pd.DataFrame | None:
    try:
        return norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting=_ADJ_TO_NORGATE[adjustment],
            padding_setting=norgatedata.PaddingType.NONE,
            timeseriesformat="pandas-dataframe",
        )
    except Exception as exc:  # delisted/access-denied/unknown symbol
        print(f"    ERR price {symbol} [{adjustment.value}]: {exc}")
        return None


def _index_slug(indexname: str) -> str:
    """Watchlist name -> safe membership column slug (``S&P 500`` -> ``sp_500``)."""
    slug = re.sub(r"[^a-z0-9]+", "_", indexname.lower()).strip("_")
    return slug.replace("s_p", "sp")


def _fetch_membership(symbol: str, index_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-index 0/1 constituent timeseries aligned to the symbol's daily dates."""
    cols: dict[str, pd.Series] = {}
    for indexname in INDEX_MEMBERSHIP_WATCHLISTS:
        try:
            ics = norgatedata.index_constituent_timeseries(
                symbol,
                indexname,
                padding_setting=norgatedata.PaddingType.NONE,
                timeseriesformat="pandas-dataframe",
            )
        except Exception:
            continue
        if ics is None or ics.empty or "Index Constituent" not in ics.columns:
            continue
        series = ics["Index Constituent"]
        series.index = pd.to_datetime(series.index).normalize().tz_localize(None)
        if (series != 0).any():
            cols[_index_slug(indexname)] = series
    frame = pd.DataFrame(cols, index=index_dates)
    frame = frame.reindex(index_dates).fillna(0).astype("int8")
    frame.index.name = "date"
    return frame


# ── per-symbol scrape ────────────────────────────────────────────────────────


def fetch_symbol(symbol: str, *, overwrite: bool = False) -> StockFetchResult | None:
    """Fetch all three adjustment series + membership flags for one symbol; write parquet.

    Returns ``None`` if the symbol could not be fetched (and should be counted as
    a skip/error by the caller). Incremental: returns ``None`` early-skip is
    handled by the caller via :func:`already_fetched`.
    """
    tr_raw = _fetch_price_frame(symbol, StockAdjustment.TOTAL_RETURN)
    if tr_raw is None or tr_raw.empty:
        return None

    daily_tr = _normalize_daily(tr_raw)
    if daily_tr.empty:
        return None

    # Unadjusted close lives as a column on the (adjusted) TR frame.
    unadj_close = None
    if "Unadjusted Close" in tr_raw.columns:
        uc = tr_raw["Unadjusted Close"].copy()
        uc.index = pd.to_datetime(uc.index).normalize().tz_localize(None)
        unadj_close = uc.astype("float32")

    cap_raw = _fetch_price_frame(symbol, StockAdjustment.CAPITAL)
    daily_cap = _normalize_daily(cap_raw) if cap_raw is not None and not cap_raw.empty else None

    # Build the UNADJ daily frame from the Unadjusted Close column (preferred) or
    # fall back to a dedicated NONE fetch.
    if unadj_close is not None:
        daily_unadj = _normalize_daily(_ohlcv_from_close(unadj_close))
    else:
        none_raw = _fetch_price_frame(symbol, StockAdjustment.UNADJUSTED)
        daily_unadj = (
            _normalize_daily(none_raw) if none_raw is not None and not none_raw.empty else None
        )

    series_by_adj: dict[StockAdjustment, pd.DataFrame | None] = {
        StockAdjustment.TOTAL_RETURN: daily_tr,
        StockAdjustment.CAPITAL: daily_cap,
        StockAdjustment.UNADJUSTED: daily_unadj,
    }

    _meta = {_RAW_SYMBOL_META_KEY: symbol.encode("utf-8")}
    for adjustment, daily in series_by_adj.items():
        if daily is None or daily.empty:
            continue
        write_norgate_bars(daily, price_path(symbol, "D", adjustment), store="stock_data", extra_metadata=_meta)
        write_norgate_bars(_aggregate(daily, "W-SUN"), price_path(symbol, "W", adjustment), store="stock_data", extra_metadata=_meta)
        write_norgate_bars(_aggregate(daily, "ME"), price_path(symbol, "M", adjustment), store="stock_data", extra_metadata=_meta)

    membership = _fetch_membership(symbol, daily_tr.index)
    n_indices = int(membership.shape[1])
    if n_indices > 0:
        _write(membership, membership_path(symbol), raw_symbol=symbol)

    return StockFetchResult(
        symbol=symbol,
        daily_rows=len(daily_tr),
        start=str(daily_tr.index.min().date()),
        end=str(daily_tr.index.max().date()),
        indices=n_indices,
    )


def already_fetched(symbol: str) -> bool:
    """Incremental guard: the primary (D_TR) file exists."""
    return price_path(symbol, "D", StockAdjustment.TOTAL_RETURN).exists()


# ── universe enumeration ─────────────────────────────────────────────────────


def list_symbols(universe: Universe) -> list[str]:
    symbols: list[str] = []
    for db in _UNIVERSE_DATABASES[universe]:
        symbols.extend(norgatedata.database_symbols(db))
    # Preserve order, drop dupes (a symbol could appear in both DBs in theory).
    seen: set[str] = set()
    return [s for s in symbols if not (s in seen or seen.add(s))]


def list_watchlist_symbols(watchlist: str) -> list[str]:
    return list(norgatedata.watchlist_symbols(watchlist))


# ── batch scrape ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScrapeSummary:
    requested: int
    written: int
    skipped_existing: int
    errored: int


def _scrape_one(symbol: str, overwrite: bool) -> tuple[str, str]:
    """Worker entrypoint: fetch one symbol, return (symbol, status).

    Status is one of 'written' | 'skipped' | 'errored'. Runs in a separate
    process; ``norgatedata`` initialises per-process (multiprocessing-safe per
    the package docs). Never raises — the batch must survive any single failure.
    """
    try:
        if not overwrite and already_fetched(symbol):
            return symbol, "skipped"
        result = fetch_symbol(symbol, overwrite=overwrite)
        return symbol, ("written" if result is not None else "errored")
    except Exception:
        return symbol, "errored"


def scrape_parallel(
    symbols: list[str], *, overwrite: bool = False, workers: int = 6,
    progress_every: int = 200,
) -> ScrapeSummary:
    """Parallel batch scrape across a process pool.

    Each worker process holds its own NDU connection. Writes target distinct
    per-symbol files, so there is no write contention. Use for the full-universe
    run (35k symbols) where the sequential ~3s/symbol is prohibitive.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    total = len(symbols)
    counts = {"written": 0, "skipped": 0, "errored": 0}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_scrape_one, sym, overwrite): sym for sym in symbols
        }
        for fut in as_completed(futures):
            _, status = fut.result()
            counts[status] += 1
            done += 1
            if done % progress_every == 0 or done == total:
                print(
                    f"  [{done}/{total}] written={counts['written']} "
                    f"skipped={counts['skipped']} errored={counts['errored']}",
                    flush=True,
                )
    summary = ScrapeSummary(
        requested=total, written=counts["written"],
        skipped_existing=counts["skipped"], errored=counts["errored"],
    )
    print(
        f"\nDone (parallel, {workers} workers). requested={summary.requested}  "
        f"written={summary.written}  skipped_existing={summary.skipped_existing}  "
        f"errored={summary.errored}"
    )
    return summary


def scrape(symbols: list[str], *, overwrite: bool = False) -> ScrapeSummary:
    written = skipped = errored = 0
    total = len(symbols)
    for i, symbol in enumerate(symbols, start=1):
        if not overwrite and already_fetched(symbol):
            skipped += 1
            continue
        try:
            result = fetch_symbol(symbol, overwrite=overwrite)
        except Exception as exc:  # never let one symbol kill the batch
            print(f"  [{i}/{total}] ERR  {symbol}: {exc}")
            errored += 1
            continue
        if result is None:
            print(f"  [{i}/{total}] SKIP {symbol}: no data")
            errored += 1
            continue
        written += 1
        print(
            f"  [{i}/{total}] OK   {symbol:<14} D={result.daily_rows:>5}"
            f"  idx={result.indices}  ({result.start} -> {result.end})"
        )
    summary = ScrapeSummary(
        requested=total, written=written, skipped_existing=skipped, errored=errored
    )
    print(
        f"\nDone. requested={summary.requested}  written={summary.written}"
        f"  skipped_existing={summary.skipped_existing}  errored={summary.errored}"
    )
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape Norgate US stocks (TR/CAP/UNADJ + index membership) "
        "to data/stock_data/.",
    )
    p.add_argument(
        "--universe",
        choices=[u.value for u in Universe],
        default=Universe.ACTIVE.value,
        help="Which equity database(s) to enumerate (default: active).",
    )
    p.add_argument(
        "--watchlist",
        default=None,
        help="Scrape only this index/watchlist's members (overrides --universe).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of symbols (for testing on a sample).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Refetch symbols even if their files already exist.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default 1 = sequential). Use 6-8 for "
             "the full-universe run; each worker holds its own NDU connection.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    ensure_norgate_running()

    if args.watchlist:
        symbols = list_watchlist_symbols(args.watchlist)
        label = f"watchlist '{args.watchlist}'"
    else:
        universe = Universe(args.universe)
        symbols = list_symbols(universe)
        label = f"universe '{universe.value}'"

    if args.limit is not None:
        symbols = symbols[: args.limit]

    print(f"=== Norgate stock scrape: {label} ({len(symbols)} symbols) "
          f"workers={args.workers} ===")
    if args.workers > 1:
        scrape_parallel(symbols, overwrite=args.overwrite, workers=args.workers)
    else:
        scrape(symbols, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
