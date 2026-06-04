"""
MT5 broker capability discovery.

Outputs:
  1. All 844 symbols with path / digits / trade_mode
  2. Tick history depth for a representative set (oldest date reachable)
  3. Bar history note: bars require chart to be pre-loaded in the UI;
     use copy_ticks_from + resample for reliable programmatic access
  4. Writes data/mt5_data/_discovery.json

Run once per session (MT5 IPC allows only one Python connection at a time):
  .\.venv\Scripts\python.exe scripts\_mt5_discovery.py
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import MetaTrader5 as mt5

OUT_PATH = _REPO_ROOT / "data" / "mt5_data" / "_discovery.json"

TICK_PROBE_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "EURGBP",
    "XAUUSD", "XTIUSD",
    "SP500", "NDX", "GDAXI", "WS30", "UK100",
    "AAPL", "MSFT", "AMZN", "NVDA",
]

FAR_DATE  = datetime(2000, 1, 1, tzinfo=timezone.utc)
MAX_TICKS = 200_000


def connect() -> bool:
    """Attach to the already-running MT5 terminal (no args = no IPC conflict)."""
    if not mt5.initialize():
        print(f"initialize() failed: {mt5.last_error()}", flush=True)
        return False
    return True


def oldest_tick_date(symbol: str) -> str:
    """
    Find the oldest available tick by binary-searching from FAR_DATE.
    Fetches 200k ticks starting at FAR_DATE; reports the first tick date.
    """
    mt5.symbol_select(symbol, True)
    ticks = mt5.copy_ticks_from(symbol, FAR_DATE, MAX_TICKS, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return "N/A"
    return datetime.fromtimestamp(int(ticks[0]["time"]), tz=timezone.utc).date().isoformat()


def newest_tick_date(symbol: str) -> str:
    """Get the most recent tick date by fetching from near-now backwards."""
    mt5.symbol_select(symbol, True)
    now = datetime.now(timezone.utc)
    # Go back 7 days to ensure we catch last close
    ticks = mt5.copy_ticks_from(symbol, now - timedelta(days=7), MAX_TICKS, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return "N/A"
    return datetime.fromtimestamp(int(ticks[-1]["time"]), tz=timezone.utc).date().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tick-probe", action="store_true",
                        help="Skip tick history probing (faster)")
    args = parser.parse_args()

    if not connect():
        sys.exit(1)

    term = mt5.terminal_info()
    acct = mt5.account_info()
    print(f"Terminal : build={term.build}  connected={term.connected}")
    print(f"Account  : login={acct.login if acct else 'N/A'}  server={acct.server if acct else 'N/A'}")
    print()

    all_syms = mt5.symbols_get() or []
    print(f"Total symbols : {len(all_syms)}")
    print()

    # ── 1. Full symbol list ───────────────────────────────────────────────────
    print(f"{'Symbol':<30} {'Digits':>6}  {'TradeMode':>10}  Path")
    print("-" * 80)
    for s in sorted(all_syms, key=lambda x: x.name):
        print(f"  {s.name:<28} {s.digits:>6}  {s.trade_mode:>10}  {s.path}")
    print()

    # ── 2. Tick history depth ─────────────────────────────────────────────────
    tick_results: dict[str, dict] = {}
    all_sym_names = {s.name for s in all_syms}

    if not args.no_tick_probe:
        print("Probing tick history depth (200k ticks from year 2000) …")
        print()
        print(f"{'Symbol':<20}  {'Oldest tick':>12}  {'Newest tick':>12}")
        print("-" * 50)
        for sym in TICK_PROBE_SYMBOLS:
            if sym not in all_sym_names:
                print(f"  {sym:<18}  NOT IN TERMINAL")
                tick_results[sym] = {"oldest": "N/A", "newest": "N/A", "in_terminal": False}
                continue
            oldest = oldest_tick_date(sym)
            newest = newest_tick_date(sym)
            print(f"  {sym:<18}  {oldest:>12}  {newest:>12}", flush=True)
            tick_results[sym] = {"oldest": oldest, "newest": newest, "in_terminal": True}
        print()

    mt5.shutdown()

    # ── 3. Write JSON ─────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "terminal_build": term.build,
        "broker": acct.server if acct else "N/A",
        "total_symbols": len(all_syms),
        "notes": {
            "bars": (
                "copy_rates_from returns 0 bars unless the chart was pre-loaded in the "
                "MT5 UI. For reliable programmatic access, fetch ticks and resample."
            ),
            "ticks": "200k cap per copy_ticks_from call. Oldest dates shown are where broker history starts.",
            "ipc":   "MT5 Python IPC allows only one Python process at a time. "
                     "Do not call mt5.shutdown() then re-initialize in the same session.",
        },
        "all_symbols": [
            {"name": s.name, "path": s.path, "digits": s.digits, "trade_mode": s.trade_mode}
            for s in sorted(all_syms, key=lambda x: x.name)
        ],
        "tick_history": tick_results,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {OUT_PATH}")


if __name__ == "__main__":
    main()
