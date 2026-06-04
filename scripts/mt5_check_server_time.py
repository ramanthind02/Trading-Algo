#!/usr/bin/env python3
"""
MT5 broker server-time diagnostic.

One-shot script to verify on the FTMO box that the broker's MT5 D1
candle aligns with the assumption baked into the scheduler (D1 closes
at ~17:00 ET, so daily rebalance at 17:05 ET sees the just-closed
bar). Run after pulling the branch and once whenever DST flips, just
to confirm the assumption still holds.

If the inferred broker UTC offset differs from +2 (winter) or +3
(summer), or the recommended ET times don't match what's in
install_cfd_prop_tasks.ps1, fix the scheduler params before the next
scheduled run.
"""

from __future__ import annotations

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
CANDIDATE_SYMBOLS = ["US500.cash", "XAUUSD", "EURUSD", "US100.cash"]


def main() -> int:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 package not installed.", file=sys.stderr)
        return 2

    if not mt5.initialize():
        print(f"ERROR: mt5.initialize() failed: {mt5.last_error()}", file=sys.stderr)
        return 2

    try:
        tick = None
        symbol_used = None
        for sym in CANDIDATE_SYMBOLS:
            if not mt5.symbol_select(sym, True):
                continue
            t = mt5.symbol_info_tick(sym)
            if t is not None and getattr(t, "time", 0):
                tick = t
                symbol_used = sym
                break

        if tick is None:
            print(
                f"ERROR: could not get tick for any of: {CANDIDATE_SYMBOLS}",
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
        print(f"Symbol used for tick:        {symbol_used}")
        print(f"MT5 server time (tick):      {broker_naive}  (naive, broker-local)")
        print(f"Real UTC right now:          {real_utc.replace(tzinfo=None)}  (naive)")
        print(f"Inferred broker UTC offset:  {offset_hours:+.2f} hours")
        print(
            f"Broker midnight today in ET: "
            f"{broker_midnight_et.strftime('%H:%M ET (%Z)')}"
        )
        print()

        if offset_hours in (2.0, 3.0):
            print("OK: broker offset matches FTMO EET (+2 winter) / EEST (+3 summer).")
            print("    Recommended daily rebalance:  17:05 ET  (5 min after bar close)")
            print("    Recommended weekend close:    16:45 ET  (15 min before bar close)")
        else:
            print("WARNING: broker offset does NOT match the assumed FTMO +2/+3.")
            shift = offset_hours - 3.0
            print("         Adjust scripts/scheduler/install_cfd_prop_tasks.ps1 by")
            print(f"         shifting -DailyRunTime / -WeekendCloseTime by {shift:+.2f}h")
            print("         relative to the 17:05 / 16:45 ET defaults.")

        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
