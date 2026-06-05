r"""Scrape daily (D1) bars for the full MT5 (Darwinex) symbol universe.

Daily bars have no tick-cap / chunking complexity — one ``copy_rates_range`` per
symbol over full history. Stored to:

    data/mt5_data/{SYMBOL}/bars_D1/part.parquet

Schema (zstd): time (UTC date), open/high/low/close (float32), tick_volume (int32),
spread (int16), real_volume (int64). Incremental: appends only sessions newer than
the last stored bar; dedupes on ``time``.

Per [[mt5_data_scraper]], ``copy_rates_range`` is the correct call (copy_rates_from
returns 0 bars cold). First call per symbol triggers a broker download (~tens of
seconds for deep-history symbols); subsequent calls read the terminal cache.

Run (TWS/Darwinex MT5 terminal must be running + logged in):
    python -m data_platform.providers.mt5.daily_scraper                 # all symbols
    python -m data_platform.providers.mt5.daily_scraper --symbols EURUSD XAUUSD US500.cash
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _HERE.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
    _HERE.parents[3],
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import MetaTrader5 as mt5  # noqa: E402
from lib.core.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

MT5_DATA_DIR = _REPO_ROOT / "data" / "mt5_data"
# Darwinex daily history reaches back to the 2000s for FX; 1970 lower bound is safe.
HISTORY_START = datetime(1970, 1, 1, tzinfo=timezone.utc)

_D1_SCHEMA = pa.schema([
    ("time",        pa.timestamp("s", tz="UTC")),
    ("open",        pa.float32()),
    ("high",        pa.float32()),
    ("low",         pa.float32()),
    ("close",       pa.float32()),
    ("tick_volume", pa.int64()),
    ("spread",      pa.int32()),
    ("real_volume", pa.int64()),
])
_COLS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


def connect() -> bool:
    if not mt5.initialize():
        logger.error("mt5.initialize() failed: %s", mt5.last_error())
        return False
    info = mt5.terminal_info()
    acct = mt5.account_info()
    logger.info("MT5 attached: build=%s login=%s server=%s",
                getattr(info, "build", "?"),
                acct.login if acct else "N/A", acct.server if acct else "N/A")
    return True


def _daily_path(symbol: str) -> Path:
    return MT5_DATA_DIR / symbol / "bars_D1" / "part.parquet"


def _last_stored_date(symbol: str) -> datetime | None:
    path = _daily_path(symbol)
    if not path.exists():
        return None
    try:
        vals = pq.read_table(path, columns=["time"]).column("time").to_pylist()
        if not vals:
            return None
        ts = vals[-1]
        ts = ts.as_py() if hasattr(ts, "as_py") else ts
        return ts if getattr(ts, "tzinfo", None) else pd.Timestamp(ts).to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception as exc:
        logger.warning("could not read last date for %s: %s", symbol, exc)
        return None


def _write_daily(symbol: str, df: pd.DataFrame) -> int:
    """Append-only write; dedupe on time. Returns rows written."""
    if df.empty:
        return 0
    path = _daily_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df[_COLS].copy()
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
    pq.write_table(
        pa.Table.from_pandas(df, schema=_D1_SCHEMA, preserve_index=False),
        path, compression="zstd", compression_level=3,
    )
    return len(df)


def scrape_symbol(symbol: str, to_dt: datetime) -> dict:
    """Fetch + store one symbol's new daily bars. Never raises — errors are returned."""
    try:
        if not mt5.symbol_select(symbol, True):
            return {"symbol": symbol, "bars": 0, "error": "symbol_select failed"}
        last = _last_stored_date(symbol)
        if last is None:
            start = HISTORY_START
        else:
            # last may be a datetime or pd.Timestamp; normalise to a tz-aware datetime.
            start = (pd.Timestamp(last) + pd.Timedelta(days=1)).to_pydatetime()
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start, to_dt)
        if rates is None or len(rates) == 0:
            return {"symbol": symbol, "bars": 0, "error": None}  # up to date / no data
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if "real_volume" not in df.columns:
            df["real_volume"] = 0
        written = _write_daily(symbol, df)
        return {"symbol": symbol, "bars": len(df), "total": written, "error": None}
    except Exception as exc:
        return {"symbol": symbol, "bars": 0, "error": str(exc)[:80]}


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape MT5 daily (D1) bars for all symbols.")
    p.add_argument("--symbols", nargs="*", default=None, help="Subset (default: all terminal symbols).")
    p.add_argument("--pace", type=float, default=0.0, help="Seconds to sleep between symbols.")
    args = p.parse_args()

    if not connect():
        sys.exit(1)
    to_dt = datetime.now(timezone.utc)
    try:
        symbols = args.symbols or [s.name for s in (mt5.symbols_get() or [])]
        if not symbols:
            logger.error("No symbols found"); sys.exit(1)
        logger.info("MT5 daily scrape | %d symbols | to=%s", len(symbols), to_dt.date())
        MT5_DATA_DIR.mkdir(parents=True, exist_ok=True)

        results = []
        for i, sym in enumerate(symbols, 1):
            r = scrape_symbol(sym, to_dt)
            results.append(r)
            status = "OK" if r["error"] is None else f"ERR({r['error']})"
            logger.info("  [%d/%d] %-22s new=%-7d %s", i, len(symbols), sym, r["bars"], status)
            if args.pace:
                time.sleep(args.pace)
    finally:
        mt5.shutdown()

    ok = sum(1 for r in results if r["error"] is None)
    new_bars = sum(r["bars"] for r in results)
    print(f"\nDone: {ok}/{len(results)} symbols OK, {new_bars:,} new daily bars written.")
    failures = [r for r in results if r["error"]]
    if failures:
        print(f"  {len(failures)} symbols with errors (first 10):")
        for r in failures[:10]:
            print(f"    {r['symbol']:<22} {r['error']}")


if __name__ == "__main__":
    main()
