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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from lib.core.runtime_bootstrap import bootstrap_runtime

bootstrap_runtime(_REPO_ROOT)  # UTF-8 console + .env load (was: import scripts._bootstrap)

import MetaTrader5 as mt5
from lib.core.logger import get_logger
from data_platform.storage.contracts import MT5_BARS_SCHEMA, MT5_TICKS_SCHEMA
from data_platform.storage import write_mt5_bars, write_mt5_ticks

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Registry helpers (resilience: every write is try/except; scrape never fails
# because metadata recording failed)
# ---------------------------------------------------------------------------

def _broker_time_anchor() -> Optional[str]:
    """Return broker wall-clock as ISO string, or None on any error.

    Uses deployment.live.runtime.rollover_market.broker_now() which reads
    the EURUSD tick.time directly from the already-connected MT5 terminal.
    Wrapped in try/except so failure (e.g. terminal not yet initialised) is
    silent — the scrape must never abort because of this.
    """
    try:
        from deployment.live.runtime.rollover_market import broker_now
        bt = broker_now()
        return bt.isoformat() if bt is not None else None
    except Exception:  # noqa: BLE001
        return None


def _record_symbol_blobs_safe(
    reg_conn: "sqlite3.Connection | None",
    symbol: str,
    *,
    fetch_ticks: bool,
) -> None:
    """Write (or refresh) blob_manifest rows for *symbol*'s M1 bars and ticks.

    Resilience wrapper: any exception is logged as a warning and swallowed.
    The scrape result must never be affected by registry failures.
    """
    if reg_conn is None:
        return
    try:
        import sqlite3
        import json as _json
        from data_platform.registry import writer as _reg_writer, db as _reg_db
        from data_platform.registry.rebuild import _extract_coverage, _mtime_iso

        now_iso = datetime.now(timezone.utc).isoformat()

        with _reg_db.transaction(reg_conn):
            # ── bars_M1 ──────────────────────────────────────────────────────
            m1_dir = MT5_DATA_DIR / symbol / "bars_M1"
            if m1_dir.exists():
                for year_dir in sorted(m1_dir.iterdir()):
                    if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                        continue
                    year = year_dir.name.split("=", 1)[1]
                    for pq_file in sorted(year_dir.glob("*.parquet")):
                        key = _json.dumps({"symbol": symbol, "year": year}, sort_keys=True)
                        cov_start, cov_end, rows = _extract_coverage(pq_file, "time")
                        _reg_writer.record_blob(
                            reg_conn,
                            _reg_writer.BlobRecord(
                                store="mt5_m1",
                                key_json=key,
                                relative_path=str(pq_file.relative_to(_REPO_ROOT)),
                                written_at=_mtime_iso(pq_file, now_iso),
                                broker="darwinex",
                                timezone="broker_eet_as_utc",
                                engine="pyarrow",
                                rows=rows,
                                coverage_start=cov_start,
                                coverage_end=cov_end,
                            ),
                        )

            # ── ticks ─────────────────────────────────────────────────────────
            if fetch_ticks:
                ticks_dir = MT5_DATA_DIR / symbol / "ticks"
                if ticks_dir.exists():
                    for year_dir in sorted(ticks_dir.iterdir()):
                        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                            continue
                        year = year_dir.name.split("=", 1)[1]
                        for pq_file in sorted(year_dir.glob("*.parquet")):
                            key = _json.dumps({"symbol": symbol, "year": year}, sort_keys=True)
                            cov_start, cov_end, rows = _extract_coverage(pq_file, "time")
                            _reg_writer.record_blob(
                                reg_conn,
                                _reg_writer.BlobRecord(
                                    store="mt5_ticks",
                                    key_json=key,
                                    relative_path=str(pq_file.relative_to(_REPO_ROOT)),
                                    written_at=_mtime_iso(pq_file, now_iso),
                                    broker="darwinex",
                                    timezone="broker_eet_as_utc",
                                    engine="pyarrow",
                                    rows=rows,
                                    coverage_start=cov_start,
                                    coverage_end=cov_end,
                                ),
                            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry: could not record blobs for %s: %s", symbol, exc)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MT5_DATA_DIR = _REPO_ROOT / "data" / "mt5_data"

# MT5 hard cap: 200,000 ticks per copy_ticks_range call.
# We target ~150k ticks/call to stay safely under the cap.
# Chunk size is computed adaptively per symbol; this is the fallback.
TICK_CAP          = 200_000
TICK_TARGET       = 180_000   # aim for this many ticks per call (closer to cap = fewer broker round-trips)
BARS_CHUNK_DAYS   = 30        # bars have no cap issue; 30-day windows are fine

# pyarrow schemas — aliases for the canonical contracts (data_platform.storage.contracts)
_BARS_SCHEMA = MT5_BARS_SCHEMA
_TICKS_SCHEMA = MT5_TICKS_SCHEMA

# ADR-7: data/mt5_data is the Darwinex-only store; per-broker stores come later.
# All connect() calls into this module must be bound to the Darwinex terminal.
_EXPECTED_BROKER = "darwinex"

# Per-file write lock: keyed by resolved path string.
# MT5 read calls (copy_ticks_range etc.) are thread-safe; parquet writes are not.
# Threads writing to different symbols never contend; same symbol same year do.
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_WRITE_LOCKS_MUTEX = threading.Lock()


def _file_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _WRITE_LOCKS_MUTEX:
        if key not in _WRITE_LOCKS:
            _WRITE_LOCKS[key] = threading.Lock()
        return _WRITE_LOCKS[key]


# ---------------------------------------------------------------------------
# MT5 connection
# ---------------------------------------------------------------------------

def _assert_terminal_broker(expected: str) -> None:
    """Verify the already-initialised MT5 terminal belongs to ``expected``.

    Calls mt5.shutdown() and raises RuntimeError if:
    - account_info() returns None (terminal not logged in), or
    - the server name does not contain ``expected`` (wrong broker).

    This is the ADR-7 store-contamination guard: native symbol names collide
    across brokers, so a scrape from the wrong terminal silently corrupts the
    per-broker stores (data/mt5_data is the Darwinex one).
    """
    acct = mt5.account_info()
    if acct is None:
        mt5.shutdown()
        raise RuntimeError(
            "mt5.account_info() returned None after successful mt5.initialize() — "
            f"the terminal may not be logged in. Ensure the {expected} terminal is "
            "running and authenticated before scraping."
        )
    if expected.lower() not in acct.server.lower():
        mt5.shutdown()
        raise RuntimeError(
            f"Wrong MT5 terminal: connected to server {acct.server!r} but "
            f"expected the {expected!r} terminal (ADR-7 broker guard)."
        )


def _assert_darwinex_terminal() -> None:
    """ADR-7 guard for writers into data/mt5_data (the Darwinex-only store)."""
    _assert_terminal_broker(_EXPECTED_BROKER)


def connect() -> bool:
    """Bind to the MT5 terminal at $MT5_PATH (Darwinex).

    MT5_PATH **must** be set (via .env or the environment). The no-arg
    ``mt5.initialize()`` fallback has been removed: with several terminals
    installed (Darwinex live + FTMO/FundedNext demos) it attaches
    nondeterministically and would silently contaminate data/mt5_data with
    the wrong broker's native symbols (ADR-7).

    Raises RuntimeError if MT5_PATH is unset or the attached terminal is not
    the Darwinex terminal. Returns False only if mt5.initialize() itself fails.
    """
    path = os.environ.get("MT5_PATH")
    if not path:
        raise RuntimeError(
            "MT5_PATH is not set. Set it in .env or the environment to the "
            "Darwinex terminal executable path before running the scraper. "
            "data/mt5_data is the Darwinex-only store (ADR-7)."
        )
    if not mt5.initialize(path):
        logger.error("mt5.initialize(%s) failed: %s", path, mt5.last_error())
        return False
    _assert_darwinex_terminal()
    acct = mt5.account_info()
    info = mt5.terminal_info()
    logger.info("MT5 attached: build=%s login=%s server=%s",
                info.build, acct.login, acct.server)
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
        with _file_lock(path):
            if path.exists():
                tbl_old = pq.read_table(path, schema=_TICKS_SCHEMA)
                old_max = pc.max(tbl_old.column("time_msc")).as_py()
                new_min = pc.min(tbl_new.column("time_msc")).as_py()
                if old_max is not None and new_min is not None and new_min > old_max:
                    combined = pa.concat_tables([tbl_old, tbl_new])
                else:
                    pdf = (
                        pa.concat_tables([tbl_old, tbl_new])
                        .to_pandas()
                        .drop_duplicates("time_msc")
                        .sort_values("time_msc")
                    )
                    combined = pa.Table.from_pandas(
                        pdf, schema=_TICKS_SCHEMA, preserve_index=False
                    )
                write_mt5_ticks(combined, path)
            else:
                write_mt5_ticks(tbl_new, path)


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
        with _file_lock(path):
            if path.exists():
                tbl_old = pq.read_table(path, schema=_BARS_SCHEMA)
                combined = pa.concat_tables([tbl_old, tbl_new]).to_pandas()
                combined = combined.drop_duplicates("time").sort_values("time")
                tbl_new = pa.Table.from_pandas(combined, schema=_BARS_SCHEMA, preserve_index=False)
            write_mt5_bars(tbl_new, path)


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
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="Parallel threads for fetching multiple symbols simultaneously. "
                        "MT5 read calls are thread-safe within one process. "
                        "Recommended: 3–5 for overnight bootstrap runs. Default: 1 (sequential).")
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

    logger.info("MT5 scrape start | to=%s | ticks=%s | workers=%d",
                to_dt.date(), args.ticks, args.workers)

    if not connect():
        logger.error("Could not connect to MT5 — aborting")
        sys.exit(1)

    # ── Registry: start job_run ───────────────────────────────────────────────
    import json as _json
    _reg_conn = None
    _job_run_id = None
    try:
        from data_platform.registry import db as _reg_db, writer as _reg_writer
        _reg_conn = _reg_db.connect()
        _args_json = _json.dumps({
            "symbols": args.symbols,
            "ticks": args.ticks,
            "from_date": args.from_date,
            "to_date": args.to_date,
            "workers": args.workers,
        })
        with _reg_db.transaction(_reg_conn):
            _job_run_id = _reg_writer.record_job_run(
                _reg_conn,
                _reg_writer.JobRun(
                    job_name="mt5_scrape",
                    started_at=datetime.now(timezone.utc).isoformat(),
                    args_json=_args_json,
                    broker_time_anchor=_broker_time_anchor(),
                ),
            )
    except Exception as _exc:  # noqa: BLE001
        logger.warning("registry: could not start job_run: %s", _exc)

    results: list[dict] = []
    _exit_code = 0
    try:
        symbols = args.symbols if args.symbols else [s.name for s in (mt5.symbols_get() or [])]
        if not symbols:
            logger.error("No symbols found")
            _exit_code = 1
            sys.exit(1)

        logger.info("Symbols: %d  workers: %d", len(symbols), args.workers)
        MT5_DATA_DIR.mkdir(parents=True, exist_ok=True)

        kwargs = dict(fetch_tick_data=args.ticks, default_from=default_from, to_dt=to_dt)

        if args.workers <= 1:
            # Sequential — original behaviour
            for sym in symbols:
                r = update_symbol(sym, **kwargs)
                results.append(r)
                status = "OK" if r["error"] is None else f"ERR({r['error'][:50]})"
                logger.info("  %-20s bars=%-8d ticks=%-10d %s",
                            sym, r["bars"], r["ticks"], status)
                if r["error"] is None:
                    _record_symbol_blobs_safe(_reg_conn, sym, fetch_ticks=args.ticks)
        else:
            # Parallel — one thread per symbol, all sharing the same MT5 IPC connection.
            # MT5 copy_ticks_range / copy_rates_range are thread-safe for concurrent reads.
            # Writes are serialised per output file via _file_lock().
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                future_to_sym = {
                    pool.submit(update_symbol, sym, **kwargs): sym
                    for sym in symbols
                }
                for fut in as_completed(future_to_sym):
                    r = fut.result()
                    results.append(r)
                    status = "OK" if r["error"] is None else f"ERR({r['error'][:50]})"
                    logger.info("  %-20s bars=%-8d ticks=%-10d %s",
                                r["symbol"], r["bars"], r["ticks"], status)
                    if r["error"] is None:
                        _record_symbol_blobs_safe(
                            _reg_conn, r["symbol"], fetch_ticks=args.ticks
                        )
    except Exception:
        _exit_code = 1
        raise
    finally:
        mt5.shutdown()
        # ── Registry: finish job_run ──────────────────────────────────────────
        if _reg_conn is not None and _job_run_id is not None:
            try:
                from data_platform.registry import db as _reg_db, writer as _reg_writer
                _ok  = sum(1 for r in results if r["error"] is None)
                _err = len(results) - _ok
                _bars  = sum(r["bars"]  for r in results)
                _ticks = sum(r["ticks"] for r in results)
                _max_bar_time: str | None = None
                for _r in results:
                    if _r.get("error") is None:
                        _last = _last_stored_ts(_r["symbol"], "bars_M1")
                        if _last:
                            _t = _last.isoformat()
                            if _max_bar_time is None or _t > _max_bar_time:
                                _max_bar_time = _t
                _cov_json = _json.dumps({
                    "symbols": len(results),
                    "ok": _ok,
                    "errors": _err,
                    "max_bar_time": _max_bar_time,
                })
                with _reg_db.transaction(_reg_conn):
                    _reg_writer.finish_job_run(
                        _reg_conn, _job_run_id,
                        exit_code=_exit_code,
                        rows_written=_bars + _ticks,
                        coverage_json=_cov_json,
                        error_text=None,
                    )
            except Exception as _exc:  # noqa: BLE001
                logger.warning("registry: could not finish job_run: %s", _exc)
            try:
                _reg_conn.close()
            except Exception:  # noqa: BLE001
                pass

    ok  = sum(1 for r in results if r["error"] is None)
    err = len(results) - ok
    logger.info("Done: %d OK, %d errors", ok, err)

    print(f"\n{'Symbol':<24} {'Bars':>10} {'Ticks':>12} {'Status'}")
    print("-" * 55)
    for r in results:
        print(f"{r['symbol']:<24} {r['bars']:>10} {r['ticks']:>12}   {'OK' if r['error'] is None else 'ERROR'}")


if __name__ == "__main__":
    main()
