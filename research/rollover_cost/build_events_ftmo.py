r"""Build rollover-event features from REAL FTMO ticks (binds to the FTMO terminal).

Mirror of build_events.py but for FTMO: connects to the FTMO Global Markets terminal,
fetches the exit/entry rollover windows for US500.cash / US100.cash / XAUUSD with FTMO's
own bid/ask, and computes the same per-event features using FTMO's specs (point, tick,
swap). FTMO is also an EET-server broker (rollover at 00:00 server), so the broker-time
windows match the Darwinex study; only the spreads/swaps/fills differ.

Output: research/rollover_cost/outputs/events_ftmo_{CANON}.parquet  (CANON = ES/NQ/GC)

Run:  .\.venv\Scripts\python.exe -m research.rollover_cost.build_events_ftmo --days 365
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.core.runtime_bootstrap import bootstrap_runtime

bootstrap_runtime(_REPO)  # UTF-8 console + .env load (was: import scripts._bootstrap)
import MetaTrader5 as mt5

from research.rollover_cost.config import SymbolSpec
from research.rollover_cost.build_events import (
    rollover_dates, event_features, EXIT_WINDOW_MIN, ENTRY_WINDOW_MIN, DEADZONE_MIN, FETCH_PAD_MIN,
)
from lib.core.logger import get_logger

logger = get_logger(__name__)
UTC = timezone.utc
OUT = _REPO / "research" / "rollover_cost" / "outputs"
FTMO_PATH = os.environ.get("FTMO_DEMO_TERMINAL_PATH",
                           r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe")
# canonical -> FTMO native symbol
FTMO_SYMS = {"ES": "US500.cash", "NQ": "US100.cash", "GC": "XAUUSD"}


def connect() -> bool:
    ok = mt5.initialize(path=FTMO_PATH, login=int(os.environ["FTMO_DEMO_LOGIN"]),
                        password=os.environ["FTMO_DEMO_PASSWORD"], server=os.environ["FTMO_DEMO_SERVER"])
    if not ok:
        logger.error("FTMO initialize failed: %s", mt5.last_error()); return False
    a = mt5.account_info()
    logger.info("Connected FTMO: login=%s server=%s", a.login, a.server)
    return True


def ftmo_spec(canon: str, sym: str) -> SymbolSpec:
    mt5.symbol_select(sym, True)
    i = mt5.symbol_info(sym)
    return SymbolSpec(canon, i.point, i.trade_tick_size, i.trade_contract_size,
                      i.swap_long, i.swap_short, i.swap_rollover3days)


def fetch_window(sym: str, start: datetime, end: datetime) -> pd.DataFrame:
    ticks = mt5.copy_ticks_range(sym, start, end, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def build_symbol(canon: str, sym: str, days: int) -> pd.DataFrame:
    spec = ftmo_spec(canon, sym)
    logger.info("FTMO %s->%s spec: point=%s tick=%s swap_long=%s triple=%s",
                canon, sym, spec.point, spec.tick_size, spec.swap_long_pts, spec.triple_weekday)
    rows = []
    rolls = rollover_dates(days)
    for i, roll in enumerate(rolls):
        fs = roll - timedelta(minutes=EXIT_WINDOW_MIN + FETCH_PAD_MIN)
        fe = roll + timedelta(minutes=DEADZONE_MIN + ENTRY_WINDOW_MIN + FETCH_PAD_MIN)
        df = fetch_window(sym, fs, fe)
        if df.empty:
            continue
        roll_ts = pd.Timestamp(roll)
        exit_df = df[df["time"] < roll_ts].reset_index(drop=True)
        entry_df = df[df["time"] >= roll_ts + pd.Timedelta(minutes=DEADZONE_MIN)].reset_index(drop=True)
        feat = event_features(canon, roll, exit_df, entry_df, spec=spec)
        if feat is not None:
            rows.append(feat)
        if (i + 1) % 25 == 0:
            logger.info("  %s: %d/%d events (%d kept)", canon, i + 1, len(rolls), len(rows))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", nargs="*", default=list(FTMO_SYMS.keys()))
    args = p.parse_args()
    if not connect():
        sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for canon in args.symbols:
            sym = FTMO_SYMS[canon]
            logger.info("Building FTMO events for %s (%s), lookback %dd", canon, sym, args.days)
            dfo = build_symbol(canon, sym, args.days)
            if dfo.empty:
                logger.warning("  %s: no events", canon); continue
            out = OUT / f"events_ftmo_{canon}.parquet"
            dfo.to_parquet(out, index=False)
            logger.info("  %s: wrote %d events -> %s", canon, len(dfo), out)
    finally:
        mt5.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
