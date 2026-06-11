"""
Full-universe M1 bar backfill — the canonical 1-minute store for everything.

M1 bars are the backtest *spine* (see docs/library/Data/hybrid_tick_backtest.md):
signals, the order schedule, and coarse marking all run on them, and ticks are
fetched on demand only inside execution windows. So we want full M1 history for
the whole Darwinex universe, once.

This is a deep, resumable, parallel backfill distinct from the incremental
`scraper.py` (which defaults to the last 2 years and is the daily scheduled job):

  * Floor at a deep date (default 2000-01-01); MT5 returns whatever M1 exists
    (FX from ~2011, metals ~2018, stocks/ETFs ~2021 — the broker's M1 depth).
  * Calendar-year chunks bound each `copy_rates_range` call to ~370k bars so no
    single call is unboundedly large.
  * Resumable: skips years already stored (via the scraper's `_last_stored_ts`);
    re-runs only fetch what's missing.
  * Parallel: one thread per symbol (MT5 reads are thread-safe; writes serialise
    per-file via the scraper's `_file_lock`).
  * Priority-ordered: FX / indices / commodities first, then ETFs, then the ~692
    US stocks — so the most-used instruments land first.
  * Heartbeat: rewrites data/mt5_data/_m1_backfill_progress.json after each
    symbol so an external monitor can report progress.

Storage (unchanged — same as the incremental scraper):
    data/mt5_data/{SYMBOL}/bars_M1/year=YYYY/part.parquet

Run:
    python -m data_platform.providers.mt5.m1_backfill                      # all symbols, from 2000
    python -m data_platform.providers.mt5.m1_backfill --from 2010-01-01 --workers 4
    python -m data_platform.providers.mt5.m1_backfill --symbols EURUSD NDX XAUUSD
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5

# Reuse the scraper's tested connection, data root, and the proven per-symbol
# update (resume + 30-day chunked copy_rates_range + thread-safe write). Using
# update_symbol verbatim keeps the deep backfill and the incremental daily job
# byte-identical and avoids re-deriving the cold-download chunking that the
# scraper already gets right (small chunks land immediately; large ranges the
# terminal hasn't downloaded return empty).
from data_platform.providers.mt5.scraper import (
    MT5_DATA_DIR,
    connect,
    _last_stored_ts,
    _write_bars,
)

from lib.core.logger import get_logger

logger = get_logger(__name__)

UTC = timezone.utc
PROGRESS_PATH = MT5_DATA_DIR / "_m1_backfill_progress.json"

# Full-history M1 is fetched with copy_rates_from(now, BIG_COUNT) — the only call
# that triggers the deep broker download for a fresh symbol (copy_rates_range over
# old dates returns empty until the chart history base is populated). BIG_COUNT
# must exceed the deepest symbol's bar count: GBPUSD has ~9.8M M1 bars back to 1993.
_BIG_COUNT = 30_000_000
_COLD_RETRIES = 40     # poll until the progressive deep download settles (count stable)
_COLD_SLEEP_S = 4.0
# Resume: skip a symbol already deep + current (newest within this many days of now
# AND oldest older than 1 year — distinguishes a real backfill from a shallow remnant).
_FRESH_DAYS = 3
_DEEP_DAYS = 365

# Symbol-class priority for ordering (lower = fetched first). Classified from the
# MT5 symbol `path` (e.g. "Forex\\EURUSD", "Stocks\\US\\Nasdaq\\AAPL").
_PRIORITY = [
    ("Forex", 0),
    ("Indices", 1),
    ("Commodities", 2),
    ("ETFs", 3),
    ("Stocks", 4),
]


def _priority(path: str) -> int:
    for prefix, rank in _PRIORITY:
        if path.startswith(prefix):
            return rank
    return 9


def _ordered_symbols(requested: list[str] | None) -> list[str]:
    """Return symbols ordered by class priority then name."""
    infos = mt5.symbols_get() or []
    by_name = {s.name: s for s in infos}
    names = requested or [s.name for s in infos]
    return sorted(names, key=lambda n: (_priority(getattr(by_name.get(n), "path", "")), n))


def _stored_oldest(symbol: str) -> datetime | None:
    """Oldest stored M1 timestamp (UTC), or None — used to detect a real deep backfill."""
    import pyarrow.parquet as pq
    base = MT5_DATA_DIR / symbol / "bars_M1"
    parts = sorted(base.glob("year=*/part.parquet")) if base.exists() else []
    if not parts:
        return None
    try:
        vals = pq.read_table(parts[0], columns=["time"]).column("time").to_pylist()
        if not vals:
            return None
        ts = vals[0]
        ts = ts.as_py() if hasattr(ts, "as_py") else ts
        return ts if getattr(ts, "tzinfo", None) else ts.replace(tzinfo=UTC)
    except Exception:
        return None


def _already_deep_current(symbol: str, to_dt: datetime) -> bool:
    """True if this symbol's stored M1 is already deep AND current (skip on resume)."""
    newest = _last_stored_ts(symbol, "bars_M1")
    if newest is None or (to_dt - newest).days > _FRESH_DAYS:
        return False
    oldest = _stored_oldest(symbol)
    return oldest is not None and (to_dt - oldest).days > _DEEP_DAYS


def backfill_symbol(symbol: str, default_from: datetime, to_dt: datetime) -> dict:
    """Fetch + store the FULL M1 history for one symbol. Never raises.

    Uses copy_rates_from(now, BIG_COUNT) — the call that triggers the deep broker
    download (copy_rates_range over old dates returns empty on a fresh base). The
    cold download takes ~30-110s for deep symbols, so we retry until data lands.
    Writes via the scraper's year-partitioned, thread-safe, dedup `_write_bars`.

    Resumable: a symbol already deep + current (per `_already_deep_current`) is
    skipped, so an interrupted run picks up where it left off (bars=-1 sentinel).
    """
    try:
        if _already_deep_current(symbol, to_dt):
            return {"symbol": symbol, "bars": -1, "error": None}   # skipped (done)
        if not mt5.symbol_select(symbol, True):
            return {"symbol": symbol, "bars": 0, "error": "symbol_select failed"}

        # Poll until the count STABILISES, not until it's merely non-zero. The
        # first call returns the recent ~100k buffer immediately and only
        # *triggers* the deep broker download in the background; subsequent calls
        # return progressively more until the download settles. Breaking on the
        # first non-empty result would store recent-only data (the original bug).
        rates = None
        prev_n = -1
        for _ in range(_COLD_RETRIES):
            rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, to_dt, _BIG_COUNT)
            n = len(rates) if rates is not None else 0
            if n > 0 and n == prev_n:
                break                       # count unchanged since last poll → settled
            prev_n = n
            time.sleep(_COLD_SLEEP_S)

        if rates is None or len(rates) == 0:
            return {"symbol": symbol, "bars": 0, "error": "no M1 returned"}

        # Honour the floor: drop bars older than default_from (memory + scope).
        floor_s = int(default_from.timestamp())
        if rates[0]["time"] < floor_s:
            rates = rates[rates["time"] >= floor_s]
        _write_bars(symbol, rates)
        return {"symbol": symbol, "bars": len(rates), "error": None}
    except Exception as exc:
        return {"symbol": symbol, "bars": 0, "error": str(exc)[:100]}


def _write_progress(state: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = datetime.now(UTC).isoformat()
    tmp = PROGRESS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(PROGRESS_PATH)


def _verify_terminal(expect_login: int | None, any_terminal: bool) -> None:
    """Abort if the bound terminal isn't the expected account.

    `mt5.initialize()` attaches to whatever MT5 terminal is running/at MT5_PATH.
    With several terminals installed (Darwinex live + FTMO/FundedNext demos), the
    wrong one can bind — yielding empty/failed fetches across the whole run. We
    guard on the account *login* (exact, unambiguous) rather than the server,
    because MT5 reports the broker display name ('Darwinex-Live') which differs
    from the connection address in $MT5_SERVER ('liveUK-mt5.darwinex.com').
    """
    acct = mt5.account_info()
    login = getattr(acct, "login", None) if acct else None
    server = getattr(acct, "server", None) if acct else None
    logger.info("Bound terminal: login=%s server=%s", login, server)
    if any_terminal or not expect_login:
        return
    if login != expect_login:
        logger.error(
            "Bound terminal login %s != expected %s (server=%s). The wrong MT5 "
            "terminal is attached. Foreground the correct terminal or fix MT5_PATH, "
            "or pass --any-terminal to override.", login, expect_login, server,
        )
        mt5.shutdown()
        sys.exit(2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-universe MT5 M1 backfill")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="Subset (default: all terminal symbols, priority-ordered)")
    p.add_argument("--from", dest="from_date", default="1970-01-01", metavar="YYYY-MM-DD",
                   help="Deep floor; MT5 returns whatever M1 exists (default: 1970-01-01 = full history)")
    p.add_argument("--workers", type=int, default=4, help="Parallel threads (default: 4)")
    p.add_argument("--expect-login", type=int,
                   default=int(os.environ["MT5_USERNAME"]) if os.environ.get("MT5_USERNAME") else None,
                   help="Required account login of the bound terminal "
                        "(default: $MT5_USERNAME). Guards against the wrong terminal.")
    p.add_argument("--any-terminal", action="store_true",
                   help="Skip the bound-terminal account check.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    default_from = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=UTC)
    to_dt = datetime.now(UTC)

    if not connect():
        logger.error("Could not connect to MT5 — aborting"); sys.exit(1)
    _verify_terminal(args.expect_login, args.any_terminal)

    try:
        symbols = _ordered_symbols(args.symbols)
        if not symbols:
            logger.error("No symbols found"); sys.exit(1)

        logger.info("M1 backfill | %d symbols | from=%s | workers=%d",
                    len(symbols), default_from.date(), args.workers)
        state = {
            "started_utc": datetime.now(UTC).isoformat(),
            "symbols_total": len(symbols),
            "symbols_done": 0,
            "total_bars": 0,
            "skipped": 0,
            "errors": 0,
            "current": None,
            "from": default_from.date().isoformat(),
            "done": False,
        }
        _write_progress(state)

        t0 = time.perf_counter()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(backfill_symbol, s, default_from, to_dt): s for s in symbols}
            for fut in as_completed(futures):
                r = fut.result()
                done += 1
                state["symbols_done"] = done
                state["total_bars"] += max(0, r["bars"])
                if r["bars"] == -1:
                    state["skipped"] += 1
                if r["error"]:
                    state["errors"] += 1
                state["current"] = f"{r['symbol']} bars={r['bars']}"
                _write_progress(state)
                if r["bars"] == -1:
                    status = "SKIP (already deep+current)"
                elif r["error"]:
                    status = f"ERR({r['error'][:40]})"
                else:
                    status = "OK"
                logger.info("  [%d/%d] %-22s bars=%-9d %s",
                            done, len(symbols), r["symbol"], r["bars"], status)

        state["done"] = True
        state["current"] = "complete"
        _write_progress(state)
        elapsed = time.perf_counter() - t0
        logger.info("M1 backfill complete: %d bars across %d symbols (%d errors) in %.0fs",
                    state["total_bars"], len(symbols), state["errors"], elapsed)
    finally:
        mt5.shutdown()

    print(f"\nM1 backfill done: {state['total_bars']:,} bars, "
          f"{state['symbols_done']}/{state['symbols_total']} symbols, "
          f"{state['errors']} errors.")


if __name__ == "__main__":
    main()
