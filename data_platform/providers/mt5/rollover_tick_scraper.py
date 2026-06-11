"""
Rollover-window tick scraper.

Captures two discrete windows around the daily financing rollover, which sits at
**00:00 broker time** (= 17:00 New York). ALL MT5 timestamps are broker time
(EET/EEST) mislabelled UTC — `copy_ticks_range` also treats the wall-clock you
pass as broker time — so every window below is expressed in **broker wall-clock**.
See docs/library/Data/mt5_timezones.md.

  EXIT window:   23:00–23:59 broker (16:00–16:59 NY) — last hour of the session.
                 Exit limits rest here; spread widens approaching 00:00.

  ENTRY window:  01:00–02:00 broker (18:00–19:00 NY) — first hour after reopen.
                 Re-entry limits rest here; spread is initially wide then
                 tightens as liquidity returns.

The dead zone 00:00–01:00 broker (17:00–18:00 NY) has no quotes on Darwinex and
is intentionally not captured.

Why two separate windows and not one block?
  - The economics of a swap-avoidance overlay are distinct per leg:
    * exit_leg:  swap_saved - spread_paid_to_exit - missed_fill_risk
    * entry_leg: cost_to_reenter - spread_paid_to_enter
  - Loading only the exit or only the entry window for backtesting is clean
    and avoids carrying the dead-zone gap in the data.

Storage layout
--------------
data/mt5_data/{SYMBOL}/ticks_rollover_exit/year=YYYY/part.parquet
data/mt5_data/{SYMBOL}/ticks_rollover_entry/year=YYYY/part.parquet

Schema (zstd): time_msc (int64), bid (f64), ask (f64), last (f64),
               volume (int64), time (UTC ts), flags (int32)

Scheduling
----------
Registered by deployment/ops/setup_scheduled_task.ps1 as
TradingAlgo/MT5RolloverTickScrape.
Fires at 4:05 PM PT (7:05 PM ET) — both windows have fully closed by then.

Run manually:
  # Today's windows (default: most recently completed rollover)
  python -m data_platform.providers.mt5.rollover_tick_scraper

  # Specific date
  python -m data_platform.providers.mt5.rollover_tick_scraper --date 2026-06-01

  # Backfill a range
  python -m data_platform.providers.mt5.rollover_tick_scraper --from 2026-01-01 --to 2026-06-01

  # Parallel backfill (recommended)
  python -m data_platform.providers.mt5.rollover_tick_scraper --from 2026-01-01 --workers 4

  # Specific symbols only
  python -m data_platform.providers.mt5.rollover_tick_scraper --symbols EURUSD USDJPY XAUUSD NDX
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

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
from data_platform.storage.contracts import MT5_TICKS_SCHEMA
from data_platform.storage import write_mt5_ticks
from data_platform.providers.mt5.scraper import _assert_darwinex_terminal

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MT5_DATA_DIR = _REPO_ROOT / "data" / "mt5_data"

NY_TZ  = ZoneInfo("America/New_York")
UTC_TZ = timezone.utc

# Default Darwinex FX/CFD session, in BROKER time (the timestamps stored/returned
# by MT5 are broker EET wall-clock, mislabelled UTC; see docs/library/Data/mt5_timezones.md).
#   Active session: 01:00–23:59 broker.  Rollover dead zone: 00:00–00:59 broker.
DEFAULT_SESSION_CLOSE_HOUR_BROKER = 0     # session ends / rollover at 00:00 broker (= 17:00 NY)
DEFAULT_SESSION_OPEN_HOUR_BROKER  = 1     # session reopens at 01:00 broker (= 18:00 NY)
WINDOW_MINUTES                    = 60    # capture 60 min before close and 60 min after open

# Per-symbol session schedule JSON (written by build_symbol_sessions.py after M1 bars).
# When present, each entry overrides the default hours for that symbol. Hours are in
# BROKER time (matching the stored timestamps).
# Format: { "SYMBOL": { "close_hour_broker": H, "close_minute_broker": M,
#                       "open_hour_broker":  H, "open_minute_broker":  M } }
_SESSIONS_CACHE: dict[str, dict] | None = None
_SESSIONS_PATH = MT5_DATA_DIR / "_symbol_sessions.json"


def _load_sessions() -> dict[str, dict]:
    global _SESSIONS_CACHE
    if _SESSIONS_CACHE is not None:
        return _SESSIONS_CACHE
    if _SESSIONS_PATH.exists():
        import json
        _SESSIONS_CACHE = json.loads(_SESSIONS_PATH.read_text(encoding="utf-8"))
    else:
        _SESSIONS_CACHE = {}
    return _SESSIONS_CACHE


def _session_hours(symbol: str) -> tuple[int, int, int, int]:
    """Return (close_hour, close_min, open_hour, open_min) in BROKER time."""
    sessions = _load_sessions()
    if symbol in sessions:
        s = sessions[symbol]
        return (s["close_hour_broker"], s.get("close_minute_broker", 0),
                s["open_hour_broker"],  s.get("open_minute_broker", 0))
    return DEFAULT_SESSION_CLOSE_HOUR_BROKER, 0, DEFAULT_SESSION_OPEN_HOUR_BROKER, 0

TICK_CAP = 200_000   # MT5 hard cap per copy_ticks_range call

_ROLLOVER_SCHEMA = MT5_TICKS_SCHEMA  # alias for the canonical contract

# Canonical window labels used in storage paths and log messages
EXIT_LABEL  = "ticks_rollover_exit"   # 16:00–16:59 NY
ENTRY_LABEL = "ticks_rollover_entry"  # 18:00–19:00 NY

# Per-file write lock: serialises writes to the same parquet file across threads.
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_WRITE_LOCKS_MUTEX = threading.Lock()


def _file_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _WRITE_LOCKS_MUTEX:
        if key not in _WRITE_LOCKS:
            _WRITE_LOCKS[key] = threading.Lock()
        return _WRITE_LOCKS[key]


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def exit_window(for_date: date, symbol: str = "") -> tuple[datetime, datetime]:
    """
    Last WINDOW_MINUTES before the rollover on broker-date *for_date* for *symbol*.

    Windows are in BROKER time (the wall-clock copy_ticks_range interprets and
    returns). Default: rollover/close at 00:00 broker → exit window 23:00–00:00
    broker (= 16:00–17:00 NY). The datetimes are tagged UTC but carry broker
    wall-clock values (no astimezone conversion — that would re-introduce the
    EET offset bug). See docs/library/Data/mt5_timezones.md.
    """
    ch, cm, _, _ = _session_hours(symbol)
    close_broker = datetime(for_date.year, for_date.month, for_date.day,
                            ch, cm, 0, tzinfo=UTC_TZ)
    return close_broker - timedelta(minutes=WINDOW_MINUTES), close_broker


def entry_window(for_date: date, symbol: str = "") -> tuple[datetime, datetime]:
    """
    First WINDOW_MINUTES after the session reopens on broker-date *for_date*.

    Broker time (see exit_window). Default: reopen at 01:00 broker → entry window
    01:00–02:00 broker (= 18:00–19:00 NY).
    """
    _, _, oh, om = _session_hours(symbol)
    open_broker = datetime(for_date.year, for_date.month, for_date.day,
                           oh, om, 0, tzinfo=UTC_TZ)
    return open_broker, open_broker + timedelta(minutes=WINDOW_MINUTES)


def dates_to_backfill(from_date: date, to_date: date) -> list[date]:
    """Weekdays in [from_date, to_date] inclusive."""
    out, current = [], from_date
    while current <= to_date:
        if current.weekday() < 5:
            out.append(current)
        current += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _window_path(symbol: str, label: str, year: int) -> Path:
    return MT5_DATA_DIR / symbol / label / f"year={year}" / "part.parquet"


def _already_stored(symbol: str, label: str,
                    w_start: datetime, w_end: datetime) -> bool:
    """True if any tick for this symbol/label inside the window is on disk."""
    path = _window_path(symbol, label, w_start.year)
    if not path.exists():
        return False
    try:
        tbl = pq.read_table(path, columns=["time"])
        for t in tbl.column("time").to_pylist():
            ts = t.as_py() if hasattr(t, "as_py") else t
            if isinstance(ts, datetime):
                ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC_TZ)
                if w_start <= ts <= w_end:
                    return True
    except Exception:
        pass
    return False


def _write_window_ticks(symbol: str, label: str, raw) -> int:
    """Append ticks to the correct year-partition file. Returns rows written."""
    if raw is None or len(raw) == 0:
        return 0

    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["year"] = df["time"].dt.year
    total = 0

    for year, grp in df.groupby("year"):
        path = _window_path(symbol, label, int(year))
        path.parent.mkdir(parents=True, exist_ok=True)
        grp = grp.drop(columns=["year"]).drop_duplicates("time_msc").sort_values("time_msc")
        tbl_new = pa.Table.from_pandas(
            grp[["time_msc", "bid", "ask", "last", "volume", "time", "flags"]],
            schema=_ROLLOVER_SCHEMA, preserve_index=False,
        )
        with _file_lock(path):
            if path.exists():
                tbl_old = pq.read_table(path, schema=_ROLLOVER_SCHEMA)
                old_max = pc.max(tbl_old.column("time_msc")).as_py()
                new_min = pc.min(tbl_new.column("time_msc")).as_py()
                if old_max is not None and new_min is not None and new_min > old_max:
                    combined = pa.concat_tables([tbl_old, tbl_new])
                else:
                    combined = pa.Table.from_pandas(
                        pa.concat_tables([tbl_old, tbl_new])
                        .to_pandas()
                        .drop_duplicates("time_msc")
                        .sort_values("time_msc"),
                        schema=_ROLLOVER_SCHEMA, preserve_index=False,
                    )
                write_mt5_ticks(combined, path)
            else:
                write_mt5_ticks(tbl_new, path)
        total += len(grp)

    return total


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_window(symbol: str, label: str,
                 w_start: datetime, w_end: datetime,
                 skip_existing: bool) -> dict:
    """Fetch ticks for one symbol/window. Never raises."""
    result = {"symbol": symbol, "label": label,
              "date": w_start.date().isoformat(), "ticks": 0, "error": None}
    try:
        if skip_existing and _already_stored(symbol, label, w_start, w_end):
            result["ticks"] = -1   # sentinel: already on disk
            return result

        # Retry symbol_select once — index CFDs sometimes need a second attempt
        # when the terminal hasn't seen the symbol recently.
        if not mt5.symbol_select(symbol, True):
            time.sleep(0.2)
            if not mt5.symbol_select(symbol, True):
                result["error"] = "symbol_select failed"
                return result

        ticks = mt5.copy_ticks_range(symbol, w_start, w_end, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return result   # market closed / holiday — no data for this window

        if len(ticks) >= TICK_CAP:
            logger.warning("%s %s %s: hit 200k cap — window may be incomplete",
                           symbol, label, w_start.date())

        result["ticks"] = _write_window_ticks(symbol, label, ticks)

    except Exception as exc:
        result["error"] = str(exc)[:100]

    return result


def fetch_both_windows(
    symbol: str,
    for_date: date,
    skip_existing: bool,
) -> tuple[dict, dict]:
    """Fetch exit + entry windows for one symbol on one date.

    Window times are per-symbol when _symbol_sessions.json exists,
    otherwise fall back to the Darwinex FX/CFD default broker-time windows
    (exit 23:00-00:00 / entry 01:00-02:00 broker = 16:00-17:00 / 18:00-19:00 NY).
    """
    ex_start, ex_end = exit_window(for_date, symbol)
    en_start, en_end = entry_window(for_date, symbol)
    r_exit  = fetch_window(symbol, EXIT_LABEL,  ex_start, ex_end,  skip_existing)
    r_entry = fetch_window(symbol, ENTRY_LABEL, en_start, en_end,  skip_existing)
    return r_exit, r_entry


# ---------------------------------------------------------------------------
# MT5 connection
# ---------------------------------------------------------------------------

def connect() -> bool:
    """Bind to the Darwinex MT5 terminal at $MT5_PATH.

    MT5_PATH must be set; no-arg initialize() fallback is removed to prevent
    silent contamination of data/mt5_data with the wrong broker's symbols (ADR-7).
    Raises RuntimeError if MT5_PATH is unset or the terminal is not Darwinex.
    """
    path = os.environ.get("MT5_PATH")
    if not path:
        raise RuntimeError(
            "MT5_PATH is not set. Set it in .env or the environment to the "
            "Darwinex terminal executable path. data/mt5_data is the "
            "Darwinex-only store (ADR-7)."
        )
    if not mt5.initialize(path):
        logger.error("mt5.initialize(%s) failed: %s", path, mt5.last_error())
        return False
    _assert_darwinex_terminal()
    acct = mt5.account_info()
    info = mt5.terminal_info()
    logger.info("MT5 attached: build=%s login=%s server=%s",
                info.build, acct.login if acct else "N/A", acct.server if acct else "N/A")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MT5 rollover-window tick scraper. Captures exit "
                    "(23:00-23:59 broker = 16:00-16:59 NY) and entry "
                    "(01:00-02:00 broker = 18:00-19:00 NY) windows."
    )
    p.add_argument("--symbols", nargs="*", metavar="SYM",
                   help="Symbols to fetch (default: all in terminal)")
    p.add_argument("--date", metavar="YYYY-MM-DD", default=None,
                   help="Single date (default: most recently completed rollover)")
    p.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD", default=None,
                   help="Start of backfill range")
    p.add_argument("--to", dest="to_date", metavar="YYYY-MM-DD", default=None,
                   help="End of backfill range (default: today)")
    p.add_argument("--workers", type=int, default=4, metavar="N",
                   help="Parallel threads. Default: 4.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-fetch even if data already exists for the window")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not connect():
        logger.error("Could not connect to MT5 — aborting")
        sys.exit(1)

    try:
        symbols = args.symbols or [s.name for s in (mt5.symbols_get() or [])]
        if not symbols:
            logger.error("No symbols found"); sys.exit(1)

        # Work in BROKER time: rollover-dates are broker calendar dates X (rollover
        # at X 00:00 broker). "Now" in broker time = current EET wall-clock.
        broker_now = datetime.now(ZoneInfo("Europe/Bucharest"))
        broker_today = broker_now.date()
        if args.date:
            target_dates = [date.fromisoformat(args.date)]
        elif args.from_date:
            from_d = date.fromisoformat(args.from_date)
            to_d   = date.fromisoformat(args.to_date) if args.to_date else broker_today
            target_dates = dates_to_backfill(from_d, to_d)
        else:
            # Default: most recently completed rollover. The entry window ends at
            # 02:00 broker — if we are past that today (broker date), today's
            # rollover has completed; else use yesterday's.
            if broker_now.hour >= DEFAULT_SESSION_OPEN_HOUR_BROKER + WINDOW_MINUTES // 60:
                target_dates = [broker_today]
            else:
                target_dates = [broker_today - timedelta(days=1)]

        skip = not args.overwrite
        MT5_DATA_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Rollover tick scrape | %d symbols | %d date(s) | workers=%d | "
            "exit=23:00-23:59 broker (16:00-16:59 NY)  entry=01:00-02:00 broker (18:00-19:00 NY)",
            len(symbols), len(target_dates), args.workers,
        )

        total_exit_ticks = total_entry_ticks = total_errors = total_skipped = 0

        for d in target_dates:
            ex_start, ex_end = exit_window(d)
            en_start, en_end = entry_window(d)
            logger.info(
                "Date %s  exit=%s-%s UTC  entry=%s-%s UTC",
                d,
                ex_start.strftime("%H:%M"), ex_end.strftime("%H:%M"),
                en_start.strftime("%H:%M"), en_end.strftime("%H:%M"),
            )

            results_exit  = []
            results_entry = []

            if args.workers <= 1:
                for sym in symbols:
                    re, rn = fetch_both_windows(sym, d, skip)
                    results_exit.append(re)
                    results_entry.append(rn)
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(fetch_both_windows, sym, d, skip): sym
                        for sym in symbols
                    }
                    for fut in as_completed(futures):
                        re, rn = fut.result()
                        results_exit.append(re)
                        results_entry.append(rn)

            day_exit   = sum(r["ticks"] for r in results_exit  if r["ticks"] > 0)
            day_entry  = sum(r["ticks"] for r in results_entry if r["ticks"] > 0)
            day_skip   = sum(1 for r in results_exit + results_entry if r["ticks"] == -1)
            day_errors = sum(1 for r in results_exit + results_entry if r["error"])

            total_exit_ticks  += day_exit
            total_entry_ticks += day_entry
            total_skipped     += day_skip
            total_errors      += day_errors

            logger.info("  %s: exit=%d ticks  entry=%d ticks  skipped=%d  errors=%d",
                        d, day_exit, day_entry, day_skip, day_errors)
            for r in results_exit + results_entry:
                if r["error"]:
                    logger.warning("    ERR %s %s: %s", r["symbol"], r["label"], r["error"])

    finally:
        mt5.shutdown()

    print(
        f"\nDone:\n"
        f"  Exit  window ticks: {total_exit_ticks:,}\n"
        f"  Entry window ticks: {total_entry_ticks:,}\n"
        f"  Already stored (skipped): {total_skipped}\n"
        f"  Errors: {total_errors}"
    )


if __name__ == "__main__":
    main()
