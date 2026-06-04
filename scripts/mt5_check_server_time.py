#!/usr/bin/env python3
"""
MT5 broker server-time diagnostic (broker-agnostic).

Quick check on whichever MT5 terminal is currently logged in (FTMO,
IC Markets, Pepperstone, BlackBull, …): infers the broker's UTC
offset from a symbol tick and reports broker midnight in ET. For a
FULL session/halt-window diagnostic that drives the actual scheduler,
use scripts/mt5_diagnose_trading_session.py instead.

Symbol candidates default to a mix likely available on most US-index/
metals brokers; pass --symbols to override per broker.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py<3.9 fallback
    from backports.zoneinfo import ZoneInfo  # type: ignore

ET = ZoneInfo("America/New_York")
DEFAULT_CANDIDATE_SYMBOLS = ["US500.cash", "XAUUSD", "EURUSD", "US100.cash"]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help=(
            "Comma-separated list of MT5 symbol names to try for the tick. "
            "First one that resolves is used. "
            f"Default: {','.join(DEFAULT_CANDIDATE_SYMBOLS)} (FTMO-style)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    candidates = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else DEFAULT_CANDIDATE_SYMBOLS
    )

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 package not installed.", file=sys.stderr)
        return 2

    if not mt5.initialize():
        print(f"ERROR: mt5.initialize() failed: {mt5.last_error()}", file=sys.stderr)
        return 2

    try:
        terminal_info = mt5.terminal_info()
        account_info = mt5.account_info()
        broker = getattr(account_info, "company", "?") if account_info else "?"
        login = getattr(account_info, "login", "?") if account_info else "?"
        server = getattr(account_info, "server", "?") if account_info else "?"
        terminal_path = getattr(terminal_info, "path", "?") if terminal_info else "?"

        tick = None
        symbol_used = None
        for sym in candidates:
            if not mt5.symbol_select(sym, True):
                continue
            t = mt5.symbol_info_tick(sym)
            if t is not None and getattr(t, "time", 0):
                tick = t
                symbol_used = sym
                break

        if tick is None:
            print(
                f"ERROR: could not get tick for any of: {candidates}. "
                "Pass --symbols with the names available on your broker.",
                file=sys.stderr,
            )
            return 3

        broker_naive = datetime.utcfromtimestamp(int(tick.time)).replace(microsecond=0)
        real_utc = datetime.now(tz=timezone.utc).replace(microsecond=0)
        offset_hours = round(
            (broker_naive.replace(tzinfo=timezone.utc) - real_utc).total_seconds()
            / 3600.0,
            2,
        )

        broker_midnight_utc = real_utc.replace(hour=0, minute=0, second=0) - timedelta(
            hours=offset_hours
        )
        broker_midnight_et = broker_midnight_utc.astimezone(ET)

        print()
        print(f"Terminal path:               {terminal_path}")
        print(f"Logged-in account:           {login} @ {server}  (broker: {broker})")
        print(f"Symbol used for tick:        {symbol_used}")
        print(f"MT5 server time (tick):      {broker_naive}  (naive, broker-local)")
        print(f"Real UTC right now:          {real_utc.replace(tzinfo=None)}  (naive)")
        print(f"Inferred broker UTC offset:  {offset_hours:+.2f} hours")
        print(
            f"Broker midnight today in ET: "
            f"{broker_midnight_et.strftime('%H:%M ET (%Z)')}"
        )
        print()
        print("Common offset hints:")
        print("    +2 = EET / GMT+2 (many EU prop firms in winter)")
        print("    +3 = EEST / GMT+3 (many EU prop firms in summer)")
        print("     0 = UTC server tz")
        print("    -5/-4 = US Eastern broker (EST/EDT)")
        print()
        print("This script only confirms server time + broker midnight. The")
        print("actual schedule depends on the per-symbol halt/reset window")
        print("(can extend before AND after broker midnight). For the full")
        print("picture run:")
        print("    python scripts/mt5_diagnose_trading_session.py")
        print(
            "    # add --symbols A,B,C if your broker doesn't use FTMO names"
        )

        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
