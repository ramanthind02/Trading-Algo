"""
Build per-symbol session-hours map from M1 bars → _symbol_sessions.json.

The rollover tick scraper places its exit/entry windows relative to each
symbol's *own* daily close/open, not a single FX rollover time.  A US-stock
CFD closes 16:00 ET and reopens 09:30 ET; an FX pair "closes" 17:00 NY and
reopens 18:00 NY.  This module determines those hours empirically from M1
bars and writes them in the format the rollover scraper reads:

    data/mt5_data/_symbol_sessions.json
    { "EURUSD": {"close_hour_broker":0,"close_minute_broker":0,
                 "open_hour_broker":1,"open_minute_broker":0}, ... }

Hours are in BROKER time (EET/EEST) — the same frame as the stored timestamps
and the rollover scraper. MT5 returns broker wall-clock mislabelled UTC, so we
read the wall-clock directly and DO NOT convert to NY (that would re-introduce
the EET offset bug). See docs/library/Data/mt5_timezones.md.

Method (per symbol)
-------------------
1. Fetch recent M1 bars in-memory (default last 150 days) via copy_rates_range.
   Bars are fetched but NOT stored — this avoids seeding the M1 store from a
   recent date, which would block a later full-history M1 backfill (the
   scraper resumes forward from the last stored bar).
2. Read bar open-times as broker wall-clock (no tz conversion).
3. Gap-detect sessions: an inter-bar gap > gap_minutes ends a session.
4. The modal session close-time and open-time (HH:MM, broker) across all detected
   sessions is the symbol's daily schedule — robust to holidays/early closes.

Symbols with too few bars to infer a stable schedule are omitted; the rollover
scraper falls back to its FX/CFD default (exit 23:00-00:00 / entry 01:00-02:00
broker = 16:00-17:00 / 18:00-19:00 NY) for them.

Run:
    python -m data_platform.providers.mt5.build_symbol_sessions
    python -m data_platform.providers.mt5.build_symbol_sessions --days 200 --workers 4
    python -m data_platform.providers.mt5.build_symbol_sessions --symbols EURUSD AAPL NDX
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

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

from data_platform.providers.mt5 import brokers
from data_platform.providers.mt5.scraper import _EXPECTED_BROKER, _assert_terminal_broker

logger = get_logger(__name__)

MT5_DATA_DIR = _REPO_ROOT / "data" / "mt5_data"


def sessions_path_for(broker: str | None) -> Path:
    """Per-broker sessions file (reopen times differ per broker); flat = legacy."""
    if broker:
        return MT5_DATA_DIR / broker / "_symbol_sessions.json"
    return MT5_DATA_DIR / "_symbol_sessions.json"

NY_TZ = ZoneInfo("America/New_York")
UTC   = timezone.utc

# A symbol needs at least this many detected sessions for a trustworthy mode.
MIN_SESSIONS = 8
# Gap (minutes) between consecutive M1 bars that marks a session boundary.
GAP_MINUTES = 10

_print_lock = threading.Lock()


def connect(broker: str | None = None) -> bool:
    """Attach to ``broker``'s terminal (bind via path) or via $MT5_PATH.

    The no-arg mt5.initialize() fallback is removed: with several terminals
    installed it attaches nondeterministically and would contaminate data/mt5_data
    with the wrong broker's symbols (ADR-7).  When no broker is given, MT5_PATH
    must be set.  Raises RuntimeError if the path is unavailable or the connected
    terminal does not belong to the requested broker (Darwinex by default).
    """
    if broker:
        path = brokers.terminal_path(broker)
    else:
        path_str = os.environ.get("MT5_PATH")
        if not path_str:
            raise RuntimeError(
                "MT5_PATH is not set. Pass --broker or set MT5_PATH in .env to the "
                "Darwinex terminal executable path. data/mt5_data is the "
                "Darwinex-only store (ADR-7)."
            )
        path = Path(path_str)
    ok = mt5.initialize(path=str(path))
    if not ok:
        logger.error("mt5.initialize(path=%s) failed: %s", path, mt5.last_error())
        return False
    # Assert the terminal we bound is the broker we asked for — an explicit
    # broker request must NOT be forced through the Darwinex-only check.
    _assert_terminal_broker(broker or _EXPECTED_BROKER)
    acct = mt5.account_info()
    logger.info("MT5 attached: login=%s server=%s (broker=%s)",
                acct.login if acct else "N/A", acct.server if acct else "N/A", broker or "default")
    return True


def infer_one(symbol: str, days: int) -> dict | None:
    """
    Return {close_hour_broker, close_minute_broker, open_hour_broker, open_minute_broker}
    for *symbol*, or None if there isn't enough data. Hours are BROKER wall-clock.
    """
    if not mt5.symbol_select(symbol, True):
        return None

    now = datetime.now(UTC)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, now - timedelta(days=days), now)
    if rates is None or len(rates) == 0:
        return None

    # Bar open-times are broker wall-clock (mislabelled UTC); strip the false tz to
    # read the broker HH:MM directly — do NOT tz_convert (that re-adds the offset).
    times = pd.to_datetime(pd.Series([int(r["time"]) for r in rates]), unit="s", utc=True)
    if len(times) < 2:
        return None
    local = times.dt.tz_localize(None).sort_values().reset_index(drop=True)

    # Gap-detect sessions
    deltas = local.diff()
    boundary = deltas > pd.Timedelta(minutes=GAP_MINUTES)
    session_id = boundary.cumsum()

    df = pd.DataFrame({"ts": local, "sid": session_id})
    grp = df.groupby("sid")["ts"]
    opens  = grp.min()
    closes = grp.max()

    if len(opens) < MIN_SESSIONS:
        return None

    open_hhmm  = Counter(opens.dt.strftime("%H:%M")).most_common(1)[0][0]
    close_hhmm = Counter(closes.dt.strftime("%H:%M")).most_common(1)[0][0]

    oh, om = (int(x) for x in open_hhmm.split(":"))
    ch, cm = (int(x) for x in close_hhmm.split(":"))

    return {
        "close_hour_broker": ch, "close_minute_broker": cm,
        "open_hour_broker":  oh, "open_minute_broker":  om,
        "_sessions_observed": int(len(opens)),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-symbol session-hours JSON from M1 bars.")
    p.add_argument("--broker", default=None,
                   help="Bind this broker's terminal (e.g. ftmo, darwinex) and write a "
                        "per-broker sessions file. Omit to use the default terminal + flat file.")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="Subset (default: all terminal symbols).")
    p.add_argument("--days", type=int, default=150,
                   help="Days of recent M1 to sample for inference (default: 150).")
    p.add_argument("--workers", type=int, default=4, help="Parallel threads (default: 4).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not connect(args.broker):
        sys.exit(1)
    sessions_path = sessions_path_for(args.broker)

    try:
        symbols = args.symbols or [s.name for s in (mt5.symbols_get() or [])]
        if not symbols:
            logger.error("No symbols found"); sys.exit(1)
        symbols = sorted(symbols)
        logger.info("Inferring sessions for %d symbols (sample=%dd, workers=%d)",
                    len(symbols), args.days, args.workers)

        sessions: dict[str, dict] = {}
        inferred = skipped = 0
        done = 0

        def _work(sym: str) -> tuple[str, dict | None]:
            try:
                return sym, infer_one(sym, args.days)
            except Exception as exc:
                logger.warning("infer %s failed: %s", sym, exc)
                return sym, None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_work, s): s for s in symbols}
            for fut in as_completed(futures):
                sym, res = fut.result()
                done += 1
                if res is not None:
                    sessions[sym] = res
                    inferred += 1
                else:
                    skipped += 1
                if done % 50 == 0:
                    logger.info("  [%d/%d] inferred=%d skipped=%d", done, len(symbols), inferred, skipped)
    finally:
        mt5.shutdown()

    sessions_path.parent.mkdir(parents=True, exist_ok=True)
    sessions_path.write_text(json.dumps(sessions, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("Wrote %d symbol sessions to %s (%d had insufficient data → use default)",
                inferred, sessions_path, skipped)

    # Quick sanity summary: group by (open,close) to show the schedule clusters
    clusters = Counter(
        (v["open_hour_broker"], v["open_minute_broker"], v["close_hour_broker"], v["close_minute_broker"])
        for v in sessions.values()
    )
    print(f"\nInferred {inferred} symbols, {skipped} fell back to default.")
    print("Top session-hour clusters (open broker → close broker : count):")
    for (oh, om, ch, cm), n in clusters.most_common(12):
        print(f"  {oh:02d}:{om:02d} → {ch:02d}:{cm:02d}   {n:>4} symbols")


if __name__ == "__main__":
    main()
