#!/usr/bin/env python3
r"""
Incremental MT5 data scraper — optimised for maximum throughput.

Key design choices (from benchmarking):
  - Broker round-trip latency is ~10-18s per chunk regardless of chunk size.
    Maximise ticks-per-call by using the largest window that stays under the
    200k tick cap. For FX (~150k ticks/week) that's 7 days; for low-volume
    instruments it can be much larger.
  - Once fetched, data is cached locally by the MT5 terminal. Re-reads
    are ~300x faster (500M ticks/min vs 1.7M ticks/min cold).
  - Use copy_rates_range (not copy_rates_from / copy_rates_from_pos) for
    bars — the only API call that triggers a broker download.

Directory layout
----------------
data/mt5_data/
  <SYMBOL>/
    bars_M1/year=YYYY/part.parquet
    ticks/year=YYYY/part.parquet

Usage
-----
# Daily incremental (M1 bars, all symbols):
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper

# Tick bootstrap for specific symbols — fetches full history:
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --ticks --symbols EURUSD XAUUSD SP500

# Tick bootstrap from a specific date:
.\.venv\Scripts\python.exe -m data_platform.providers.mt5.scraper --ticks --from 2020-01-01 --symbols EURUSD
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _HERE.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
    _HERE.parents[3],
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts._bootstrap  # noqa: F401

import MetaTrader5 as mt5
from lib.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MT5_DATA_DIR = _REPO_ROOT / "data" / "mt5_data"

# MT5 hard cap: 200,000 ticks per copy_ticks_range call.
# We target ~150k ticks/call to stay safely under the cap.
# Chunk size is computed adaptively per symbol; this is the fallback.
TICK_CAP          = 200_000
TICK_TARGET       = 150_000   # aim for this many ticks per call
BARS_CHUNK_DAYS   = 30        # bars have no cap issue; 30-day windows are fine

# pyarrow schemas
_BARS_SCHEMA = pa.schema([
    ("time",        pa.timestamp("s", tz="UTC")),
    ("open",        pa.float32()),
    ("high",        pa.float32()),
    ("low",         pa.float32()),
    ("close",       pa.float32()),
    ("tick_volume", pa.int32()),
    ("spread",      pa.int16()),
    ("real_volume", pa.int64()),
])

_TICKS_SCHEMA = pa.schema([
    ("time_msc", pa.int64()),
    ("bid",      pa.float64()),
    ("ask",      pa.float64()),
    ("last",     pa.float64()),
    ("volume",   pa.int64()),
    ("time",     pa.timestamp("s", tz="UTC")),
    ("flags",    pa.int32()),
])


# ---------------------------------------------------------------------------
# MT5 connection
# ---------------------------------------------------------------------------

def connect() -> bool:
    """Attach to the already-running MT5 terminal (no args = no IPC conflict)."""
    if not mt5.initialize():
        logger.error("mt5.initialize() failed: %s", mt5.last_error())
        return False
    acct = mt5.account_info()
    info = mt5.terminal_info()
    logger.info("MT5 attached: build=%s login=%s server=%s",
                info.build, acct.login if acct else "N/A", acct.server if acct else "N/A")
    return True


# ---------------------------------------------------------------------------
# Adaptive chunk sizing
# ---------------------------------------------------------------------------

def probe_ticks_per_day(symbol: str, near_date: datetime) -> int:
    """
    Fetch one week of ticks near *near_date* to measure tick density.
    Returns estimated ticks/day.  Falls back to a conservative 100k/day.
    """
    end = near_date + timedelta(days=7)
    ticks = mt5.copy_ticks_range(symbol, near_date, end, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return 100_000  # safe fallback
    # Divide by 5 trading days
    return max(1, len(ticks) // 5)


def optimal_chunk_days(ticks_per_day: int) -> int:
    """
    Largest chunk window that keeps ticks/call under TICK_TARGET.
    Capped at 30 days.
    """
    if ticks_per_day <= 0:
        return 7
    days = max(1, TICK_TARGET // ticks_per_day)
    return min(days, 30)


# ---------------------------------------------------------------------------
# Parquet storage helpers
# ---------------------------------------------------------------------------

def _bars_path(symbol: str, year: int) -> Path:
    return MT5_DATA_DIR / symbol / "bars_M1" / f"year={year}" / "part.parquet"


def _ticks_path(symbol: str, year: int) -> Path:
    return MT5_DATA_DIR / symbol / "ticks" / f"year={year}" / "part.parquet"


def _last_stored_ts(symbol: str, kind: str) -> Optional[datetime]:
    """Return the latest timestamp already stored (UTC-aware), or None."""
    base = MT5_DATA_DIR / symbol / kind
    if not base.exists():
        return None
    part_files = sorted(base.glob("year=*/part.parquet"))
    if not part_files:
        return None
    try:
        tbl = pq.read_table(part_files[-1], columns=["time"])
        vals = tbl.column("time").to_pylist()
        if not vals:
            return None
        ts = vals[-1]
        if hasattr(ts, "as_py"):
            ts = ts.as_py()
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return pd.Timestamp(ts).to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception as exc:
        logger.warning("Could not read last ts from %s/%s: %s", symbol, kind, exc)
        return None


def _write_ticks(symbol: str, raw: "np.ndarray") -> None:
    """Convert MT5 tick array → DataFrame → year-partitioned parquet.

    The new chunk is sorted/deduped on its own (cheap — it's small). When an
    existing year file is present we merge memory-efficiently:

    - Incremental append (new ticks strictly newer than what's stored — the
      common case, since fetches resume from last_stored_ts + 1ms): zero-copy
      ``concat_tables`` + streaming write, so peak memory stays ~O(file).
    - Overlap/backfill (rare): merge and dedup in Arrow via ``sort_by``.

    The previous implementation always did a pandas ``drop_duplicates`` +
    ``sort_values`` on the *combined* old+new frame, which deep-copies the
    whole file several times over and OOMed once year files reached ~40M rows.
    """
    if raw is None or len(raw) == 0:
        return
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["year"] = df["time"].dt.year

    for year, grp in df.groupby("year"):
        path = _ticks_path(symbol, int(year))
        path.parent.mkdir(parents=True, exist_ok=True)
        grp = grp.drop(columns=["year"]).drop_duplicates("time_msc").sort_values("time_msc")
        tbl_new = pa.Table.from_pandas(
            grp[["time_msc", "bid", "ask", "last", "volume", "time", "flags"]],
            schema=_TICKS_SCHEMA, preserve_index=False,
        )
        if path.exists():
            tbl_old = pq.read_table(path, schema=_TICKS_SCHEMA)
            old_max = pc.max(tbl_old.column("time_msc")).as_py()
            new_min = pc.min(tbl_new.column("time_msc")).as_py()
            if old_max is not None and new_min is not None and new_min > old_max:
                # Pure append: no cross-overlap, both sides already sorted/deduped.
                combined = pa.concat_tables([tbl_old, tbl_new])
            else:
                # Overlap/backfill (rare): re-scraping dates already on disk.
                # Fall back to the pandas merge+dedup.
                pdf = (
                    pa.concat_tables([tbl_old, tbl_new])
                    .to_pandas()
                    .drop_duplicates("time_msc")
                    .sort_values("time_msc")
                )
                combined = pa.Table.from_pandas(
                    pdf, schema=_TICKS_SCHEMA, preserve_index=False
                )
            pq.write_table(combined, path, compression="zstd")
        else:
            pq.write_table(tbl_new, path, compression="zstd")


def _write_bars(symbol: str, raw: "np.ndarray") -> None:
    if raw is None or len(raw) == 0:
        return
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["year"] = df["time"].dt.year

    for year, grp in df.groupby("year"):
        path = _bars_path(symbol, int(year))
        path.parent.mkdir(parents=True, exist_ok=True)
        grp = grp.drop(columns=["year"])
        tbl_new = pa.Table.from_pandas(
            grp[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]],
            schema=_BARS_SCHEMA, preserve_index=False,
        )
        if path.exists():
            tbl_old = pq.read_table(path, schema=_BARS_SCHEMA)
            combined = pa.concat_tables([tbl_old, tbl_new]).to_pandas()
            combined = combined.drop_duplicates("time").sort_values("time")
            tbl_new = pa.Table.from_pandas(combined, schema=_BARS_SCHEMA, preserve_index=False)
        pq.write_table(tbl_new, path, compression="zstd")


# ---------------------------------------------------------------------------
# Tick fetch — adaptive chunking
# ---------------------------------------------------------------------------

def fetch_ticks(symbol: str, from_dt: datetime, to_dt: datetime) -> int:
    """
    Fetch ticks for *symbol* in [from_dt, to_dt) using adaptive chunk sizing.

    Chunk size is calibrated against the symbol's tick density so each
    broker round-trip carries ~150k ticks (near the 200k cap).  This
    minimises the number of broker calls, which dominate latency.
    """
    if not mt5.symbol_select(symbol, True):
        logger.warning("symbol_select(%s) failed — skipping ticks", symbol)
        return 0

    # Probe density on a recent week to size chunks
    probe_start = max(from_dt, to_dt - timedelta(days=30))
    tpd = probe_ticks_per_day(symbol, probe_start)
    chunk_days = optimal_chunk_days(tpd)
    logger.info("%s: ~%d ticks/day  ->  %d-day chunks", symbol, tpd, chunk_days)

    total = 0
    cursor = from_dt
    chunk_td = timedelta(days=chunk_days)
    n_chunks = max(1, int((to_dt - from_dt).days / chunk_days))
    done = 0

    t_start = time.perf_counter()

    while cursor < to_dt:
        chunk_end = min(cursor + chunk_td, to_dt)
        ticks = mt5.copy_ticks_range(symbol, cursor, chunk_end, mt5.COPY_TICKS_ALL)
        n = len(ticks) if ticks is not None else 0

        if n >= TICK_CAP:
            # Hit the cap — halve the chunk size for remaining work
            chunk_days = max(1, chunk_days // 2)
            chunk_td = timedelta(days=chunk_days)
            logger.debug("%s: hit 200k cap, halving chunk to %d days", symbol, chunk_days)

        if n > 0:
            _write_ticks(symbol, ticks)
            total += n

        done += 1
        elapsed = time.perf_counter() - t_start
        rate_m = (total / elapsed / 1e6 * 60) if elapsed > 0 else 0
        logger.debug(
            "%s chunk %d/%d (%s..%s): %d ticks  total=%d  %.1f M/min",
            symbol, done, n_chunks,
            cursor.date(), chunk_end.date(), n, total, rate_m,
        )

        cursor = chunk_end

    elapsed = time.perf_counter() - t_start
    rate_m = (total / elapsed / 1e6 * 60) if elapsed > 0 else 0
    logger.info("%s ticks done: %d ticks in %.1fs = %.1f M ticks/min", symbol, total, elapsed, rate_m)
    return total


# ---------------------------------------------------------------------------
# Bar fetch
# ---------------------------------------------------------------------------

def fetch_bars(symbol: str, from_dt: datetime, to_dt: datetime) -> int:
    if not mt5.symbol_select(symbol, True):
        logger.warning("symbol_select(%s) failed — skipping bars", symbol)
        return 0

    total = 0
    cursor = from_dt
    while cursor < to_dt:
        chunk_end = min(cursor + timedelta(days=BARS_CHUNK_DAYS), to_dt)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, cursor, chunk_end)
        if rates is not None and len(rates) > 0:
            _write_bars(symbol, rates)
            total += len(rates)
        cursor = chunk_end
    return total


# ---------------------------------------------------------------------------
# Per-symbol orchestration
# ---------------------------------------------------------------------------

def update_symbol(
    symbol: str,
    *,
    fetch_tick_data: bool,
    default_from: datetime,
    to_dt: datetime,
) -> dict:
    result = {"symbol": symbol, "bars": 0, "ticks": 0, "error": None}
    try:
        # Bars
        last_bar = _last_stored_ts(symbol, "bars_M1")
        bar_from = (last_bar + timedelta(minutes=1)) if last_bar else default_from
        if bar_from < to_dt:
            result["bars"] = fetch_bars(symbol, bar_from, to_dt)

        # Ticks
        if fetch_tick_data:
            last_tick = _last_stored_ts(symbol, "ticks")
            tick_from = (last_tick + timedelta(milliseconds=1)) if last_tick else default_from
            if tick_from < to_dt:
                result["ticks"] = fetch_ticks(symbol, tick_from, to_dt)

    except Exception as exc:
        logger.error("Error updating %s: %s", symbol, exc, exc_info=True)
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental MT5 data scraper")
    p.add_argument("--symbols", nargs="*", metavar="SYM",
                   help="Symbols to fetch (default: all in terminal)")
    p.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD", default=None,
                   help="Earliest date if no local data exists (default: 2 years ago)")
    p.add_argument("--ticks", action="store_true",
                   help="Also fetch raw tick data (slow on first run)")
    p.add_argument("--to", dest="to_date", metavar="YYYY-MM-DD", default=None,
                   help="End date (default: now UTC)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    to_dt = (
        datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.to_date else datetime.now(timezone.utc)
    )
    default_from = (
        datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.from_date else to_dt - timedelta(days=365 * 2)
    )

    logger.info("MT5 scrape start | to=%s | ticks=%s", to_dt.date(), args.ticks)

    if not connect():
        logger.error("Could not connect to MT5 — aborting")
        sys.exit(1)

    try:
        symbols = args.symbols if args.symbols else [s.name for s in (mt5.symbols_get() or [])]
        if not symbols:
            logger.error("No symbols found"); sys.exit(1)

        logger.info("Symbols: %d", len(symbols))
        MT5_DATA_DIR.mkdir(parents=True, exist_ok=True)

        results = []
        for sym in symbols:
            r = update_symbol(sym, fetch_tick_data=args.ticks,
                              default_from=default_from, to_dt=to_dt)
            results.append(r)
            status = "OK" if r["error"] is None else f"ERR({r['error'][:50]})"
            logger.info("  %-20s bars=%-8d ticks=%-10d %s",
                        sym, r["bars"], r["ticks"], status)
    finally:
        mt5.shutdown()

    ok  = sum(1 for r in results if r["error"] is None)
    err = len(results) - ok
    logger.info("Done: %d OK, %d errors", ok, err)

    print(f"\n{'Symbol':<24} {'Bars':>10} {'Ticks':>12} {'Status'}")
    print("-" * 55)
    for r in results:
        print(f"{r['symbol']:<24} {r['bars']:>10} {r['ticks']:>12}   {'OK' if r['error'] is None else 'ERROR'}")


if __name__ == "__main__":
    main()
