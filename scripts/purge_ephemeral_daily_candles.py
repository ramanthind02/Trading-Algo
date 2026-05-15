#!/usr/bin/env python3
"""Remove session / partial-style daily rows from the central candle cache.

Historically, the personal live profile upserted a synthetic \"today\" daily
row into the central cache before the official close. That behavior is gone;
this tool scrubs persisted rows so the next ``enigma_*_forecast`` run can
refill from IB or bootstrap.

Examples::

    # Preview removing NY-today rows for default live tickers
    python scripts/purge_ephemeral_daily_candles.py --dry-run --drop-today

    # Apply: drop NY-today and refresh monthly bars from remaining dailies
    python scripts/purge_ephemeral_daily_candles.py --drop-today

    # Aggressive: drop all daily rows on or after 2025-01-01 (then refetch via live script)
    python scripts/purge_ephemeral_daily_candles.py --drop-on-or-after 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.cache.runtime.central_cache import CentralCacheStore
from utils.cache.runtime.central_cache_models import ArtifactScope
from utils.core.enums import TimeFrame, Ticker

_DEFAULT_TICKERS: tuple[str, ...] = ("ES", "NQ", "YM", "RTY", "GC", "TLT")


def _ny_today_normalized() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=ZoneInfo("America/New_York")).date())


def _parse_cutoff(value: str) -> pd.Timestamp:
    return pd.Timestamp(date.fromisoformat(value)).normalize()


def _trim_daily_index(
    frame: pd.DataFrame,
    *,
    drop_today_ny: bool,
    drop_on_or_after: pd.Timestamp | None,
) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame, 0
    idx_norm = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    keep = pd.Series(True, index=frame.index)
    if drop_today_ny:
        today = _ny_today_normalized()
        keep &= idx_norm != today
    if drop_on_or_after is not None:
        cut = drop_on_or_after.normalize()
        keep &= idx_norm < cut
    trimmed = frame.loc[keep].sort_index()
    return trimmed, int(len(frame) - len(trimmed))


def _monthly_from_daily_indexed(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    g = daily.sort_index()
    monthly = g.resample("ME").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    monthly.index.name = "datetime"
    return monthly


def _process_ticker(
    store: CentralCacheStore,
    ticker: Ticker,
    *,
    drop_today_ny: bool,
    drop_on_or_after: pd.Timestamp | None,
    dry_run: bool,
    skip_monthly: bool,
) -> tuple[int, int]:
    """Return (daily_removed, monthly_rewritten)."""
    record = store.describe_candle(ticker, TimeFrame.D)
    if record is None or record.coverage.start is None or record.coverage.end is None:
        print(f"  {ticker.name} D: no candle record, skip")
        return 0, 0

    daily = store.query_candles(
        ticker,
        TimeFrame.D,
        start=record.coverage.start,
        end=record.coverage.end,
    )
    trimmed, removed = _trim_daily_index(
        daily,
        drop_today_ny=drop_today_ny,
        drop_on_or_after=drop_on_or_after,
    )
    if removed == 0:
        print(f"  {ticker.name} D: no rows matched filters")
        return 0, 0

    print(f"  {ticker.name} D: would remove {removed} row(s), {len(trimmed)} remain")
    if dry_run:
        return removed, 0

    store.set_candles(ticker, TimeFrame.D, trimmed, scope=ArtifactScope.LIVE)
    monthly_written = 0
    if not skip_monthly and not trimmed.empty:
        monthly = _monthly_from_daily_indexed(trimmed)
        if not monthly.empty:
            store.set_candles(ticker, TimeFrame.M, monthly, scope=ArtifactScope.LIVE)
            monthly_written = 1
            print(f"  {ticker.name} M: rewritten from trimmed daily ({len(monthly)} rows)")
    return removed, monthly_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=list(_DEFAULT_TICKERS),
        help=f"Ticker symbols (default: {' '.join(_DEFAULT_TICKERS)})",
    )
    parser.add_argument(
        "--drop-today",
        action="store_true",
        help="Remove rows whose calendar date is today in America/New_York",
    )
    parser.add_argument(
        "--drop-on-or-after",
        metavar="YYYY-MM-DD",
        help="Remove all daily rows on or after this calendar date (naive compare on index)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not write",
    )
    parser.add_argument(
        "--skip-monthly",
        action="store_true",
        help="Do not rebuild monthly candles after trimming daily",
    )
    args = parser.parse_args()

    if not args.drop_today and not args.drop_on_or_after:
        parser.error("Specify at least one of --drop-today or --drop-on-or-after")

    drop_cut = _parse_cutoff(args.drop_on_or_after) if args.drop_on_or_after else None

    tickers = tuple(dict.fromkeys(str(t).upper() for t in args.tickers))
    store = CentralCacheStore.get_instance()

    total_d = 0
    total_m = 0
    for name in tickers:
        try:
            t_enum = Ticker[name]
        except KeyError:
            print(f"  Skip unknown ticker {name}")
            continue
        removed, m_w = _process_ticker(
            store,
            t_enum,
            drop_today_ny=args.drop_today,
            drop_on_or_after=drop_cut,
            dry_run=args.dry_run,
            skip_monthly=args.skip_monthly,
        )
        total_d += removed
        total_m += m_w

    mode = "dry-run" if args.dry_run else "applied"
    print(f"Done ({mode}): daily rows matched={total_d}, monthly series rewritten={total_m}")


if __name__ == "__main__":
    main()
    sys.exit(0)
