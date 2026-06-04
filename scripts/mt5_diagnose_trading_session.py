#!/usr/bin/env python3
"""
Comprehensive FTMO MT5 trading-session diagnostic.

Run on the actual FTMO VPS to verify (empirically, from real broker
data) every scheduling assumption baked into the CFD-prop automation:

  1. Broker server clock + UTC offset
  2. WHEN the D1 candle actually flips — examine the OPEN timestamp
     of the last 7 D1 bars (the bar's `time` field is its open in
     broker-local time)
  3. WHEN the daily no-trade window is — empirical M1-gap analysis.
     Each tradable symbol has a brief daily reset / swap-rollover
     where no ticks flow. We find every gap > 2 min in the last
     ~3000 M1 bars and report start/end in ET. The most common
     gap-start minute IS the daily reset window for that symbol.
  4. Broker-reported trade & quote sessions for each day of week,
     converted to ET — authoritative source for "is X tradable now"
  5. Per-symbol swap rates (long / short / mode) so we know what we
     pay holding through the rollover

Output is verbose by design — paste the entire thing back and we'll
adjust scripts/scheduler/install_cfd_prop_tasks.ps1 if reality
disagrees with the 17:05 / 16:45 ET defaults.

Always exits 0 so Task Scheduler won't flag it as a failure.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Optional

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

SYMBOLS = ["US500.cash", "US100.cash", "XAUUSD", "XAGUSD"]
DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# How many M1 bars to fetch for gap analysis. ~3000 covers ~2 active
# trading days even with overnight gaps.
M1_FETCH_COUNT = 3000

# Minimum gap (minutes) to flag. 2 min filters routine sub-minute jitter.
GAP_MIN_MINUTES = 2.0

# Cap gaps at 6 hours so weekend gaps (Fri-close → Sun-open ~ 48h) don't
# pollute the daily-reset signal.
GAP_MAX_MINUTES = 6 * 60

# Map MT5 trade_mode enum -> human label.
TRADE_MODE_LABELS = {
    0: "DISABLED",
    1: "LONGONLY",
    2: "SHORTONLY",
    3: "CLOSEONLY",
    4: "FULL",
}

# Map MT5 swap_mode enum -> human label.
SWAP_MODE_LABELS = {
    0: "DISABLED",
    1: "POINTS",
    2: "MONEY_SYMBOL",
    3: "POINTS_RATE",
    4: "MARGIN_SYMBOL_CURRENCY",
    5: "MARGIN_DEPOSIT_CURRENCY",
    6: "INTEREST_CURRENT",
    7: "INTEREST_OPEN",
    8: "REOPEN_CURRENT",
    9: "REOPEN_BID",
}


def main() -> int:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 package not installed.", file=sys.stderr)
        return 0

    if not mt5.initialize():
        print(f"ERROR: mt5.initialize() failed: {mt5.last_error()}", file=sys.stderr)
        return 0

    try:
        print("=" * 72)
        print("FTMO MT5 trading-session diagnostic")
        print(f"Real local time:  {datetime.now()}")
        print(f"Real UTC:         {datetime.now(tz=timezone.utc).replace(microsecond=0)}")
        print("=" * 72)
        print()

        offset_hours = _report_server_time(mt5)
        print()

        _report_recommendation_preview(offset_hours)
        print()

        for sym in SYMBOLS:
            print("=" * 72)
            print(f"SYMBOL: {sym}")
            print("=" * 72)
            if not mt5.symbol_select(sym, True):
                print(f"  NOT AVAILABLE on this MT5 terminal ({mt5.last_error()})")
                print()
                continue

            try:
                _report_symbol_info(mt5, sym)
            except Exception as e:
                print(f"  symbol_info section failed: {type(e).__name__}: {e}")
            print()
            try:
                _report_sessions(mt5, sym, offset_hours)
            except Exception as e:
                print(f"  sessions section failed: {type(e).__name__}: {e}")
            print()
            try:
                _report_d1_bars(mt5, sym, offset_hours)
            except Exception as e:
                print(f"  D1 bars section failed: {type(e).__name__}: {e}")
            print()
            try:
                _report_m1_gaps(mt5, sym, offset_hours)
            except Exception as e:
                print(f"  M1 gap section failed: {type(e).__name__}: {e}")
            print()

        print("=" * 72)
        print("Paste this entire output back so we can fix the schedule if needed.")
        print("=" * 72)
        return 0
    finally:
        mt5.shutdown()


def _report_server_time(mt5) -> float:
    """Print broker clock & UTC offset. Returns offset_hours (broker - UTC)."""

    tick = None
    sym_used = None
    for sym in SYMBOLS + ["EURUSD"]:
        if not mt5.symbol_select(sym, True):
            continue
        t = mt5.symbol_info_tick(sym)
        if t is not None and getattr(t, "time", 0):
            tick = t
            sym_used = sym
            break

    if tick is None:
        print("ERROR: could not get any symbol tick to derive server time.")
        return 0.0

    broker_naive = datetime.utcfromtimestamp(int(tick.time)).replace(microsecond=0)
    real_utc = datetime.now(tz=timezone.utc).replace(microsecond=0)
    offset_hours = round(
        (broker_naive.replace(tzinfo=timezone.utc) - real_utc).total_seconds() / 3600.0,
        2,
    )

    print(f"1. Broker server time (from {sym_used} tick):")
    print(f"     server-local (naive): {broker_naive}")
    print(f"     real UTC:             {real_utc.replace(tzinfo=None)}")
    print(f"     -> broker UTC offset: {offset_hours:+.2f} hours")
    if offset_hours == 2.0:
        print("     -> matches FTMO EET (winter / standard time)")
    elif offset_hours == 3.0:
        print("     -> matches FTMO EEST (summer / DST)")
    else:
        print("     -> DOES NOT match FTMO's expected +2/+3. Investigate.")

    return offset_hours


def _broker_midnight_in_et(offset_hours: float) -> datetime:
    """When does broker-midnight fall in ET right now?"""
    now_utc = datetime.now(tz=timezone.utc)
    broker_midnight_utc = (
        now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        - timedelta(hours=offset_hours)
    )
    return broker_midnight_utc.astimezone(ET)


def _report_recommendation_preview(offset_hours: float) -> None:
    bm_et = _broker_midnight_in_et(offset_hours)
    bm_total_min = bm_et.hour * 60 + bm_et.minute

    daily_min = (bm_total_min + 5) % (24 * 60)
    weekend_min = (bm_total_min - 15) % (24 * 60)

    daily_h, daily_m = divmod(daily_min, 60)
    weekend_h, weekend_m = divmod(weekend_min, 60)

    print("2. Schedule preview (BEFORE confirming with M1 gap analysis below):")
    print(f"     Broker midnight in ET:  {bm_et.strftime('%H:%M %Z')}")
    print(f"     Implied D1 bar close:   ~{bm_et.strftime('%H:%M %Z')}")
    print(f"     Proposed daily rebal:    {daily_h:02d}:{daily_m:02d} ET (broker midnight + 5 min)")
    print(f"     Proposed weekend close:  {weekend_h:02d}:{weekend_m:02d} ET (broker midnight - 15 min)")
    print()
    print("   ⚠ Confirm against §5 (M1 gap analysis) for each symbol — if the")
    print("     daily no-trade window is longer than 5 min, the rebalance must")
    print("     land AFTER the gap ends, not just after broker midnight.")


def _report_symbol_info(mt5, sym: str) -> None:
    info = mt5.symbol_info(sym)
    if info is None:
        print(f"  symbol_info({sym}) returned None")
        return
    trade_mode = TRADE_MODE_LABELS.get(int(getattr(info, "trade_mode", -1)), str(getattr(info, "trade_mode", "?")))
    swap_mode = SWAP_MODE_LABELS.get(int(getattr(info, "swap_mode", -1)), str(getattr(info, "swap_mode", "?")))
    print("3. Symbol info:")
    print(f"     trade_mode:   {trade_mode}  (FULL = can buy & sell now)")
    print(f"     spread (pts): {getattr(info, 'spread', '?')}")
    print(f"     digits:       {getattr(info, 'digits', '?')}")
    print(f"     point:        {getattr(info, 'point', '?')}")
    print(f"     swap_mode:    {swap_mode}")
    print(f"     swap_long:    {getattr(info, 'swap_long', '?')}")
    print(f"     swap_short:   {getattr(info, 'swap_short', '?')}")
    rollover = getattr(info, "swap_rollover3days", None)
    if rollover is not None:
        try:
            idx = int(rollover)
            print(f"     swap_rollover3days: {DAY_NAMES[idx % 7]} (broker-tz, triple-swap day)")
        except Exception:
            print(f"     swap_rollover3days: {rollover}")
    else:
        print("     swap_rollover3days: <not exposed by this MT5 build>")
    print(f"     volume_min:   {getattr(info, 'volume_min', '?')}")
    print(f"     volume_step:  {getattr(info, 'volume_step', '?')}")


def _report_sessions(mt5, sym: str, offset_hours: float) -> None:
    quotes_fn = getattr(mt5, "symbol_info_sessions_quotes", None)
    trade_fn = getattr(mt5, "symbol_info_sessions_trade", None)
    if quotes_fn is None and trade_fn is None:
        print("4. Broker-reported sessions: API not available in this MT5 Python build")
        print("   (symbol_info_sessions_quotes/_trade missing — skipped; rely on M1")
        print("   gap analysis below for empirical session windows).")
        return
    print("4. Broker-reported QUOTE & TRADE sessions per day of week:")
    print(f"   (server-tz times converted to ET assuming current broker offset "
          f"{offset_hours:+.2f}h; DST transitions may shift ET by 1h)")
    print()
    print(f"   {'Day':<5} {'Quote sessions (broker tz | ET)':<58} {'Trade sessions (broker tz | ET)'}")
    for dow in range(7):
        quote_sessions = _collect_sessions(quotes_fn, sym, dow) if quotes_fn else []
        trade_sessions = _collect_sessions(trade_fn, sym, dow) if trade_fn else []
        q_str = _format_sessions(quote_sessions, offset_hours) if quote_sessions else "closed"
        t_str = _format_sessions(trade_sessions, offset_hours) if trade_sessions else "closed"
        print(f"   {DAY_NAMES[dow]:<5} {q_str:<58} {t_str}")


def _collect_sessions(fn, sym: str, dow: int) -> list[tuple[int, int]]:
    if fn is None:
        return []
    sessions: list[tuple[int, int]] = []
    for idx in range(6):
        try:
            result = fn(sym, dow, idx)
        except Exception:
            break
        if result is None:
            break
        try:
            from_s = int(result[0])
            to_s = int(result[1])
        except (TypeError, IndexError):
            try:
                from_s = int(getattr(result, "from_"))
                to_s = int(getattr(result, "to"))
            except Exception:
                break
        sessions.append((from_s, to_s))
    return sessions


def _format_sessions(sessions: list[tuple[int, int]], offset_hours: float) -> str:
    parts = []
    for from_s, to_s in sessions:
        from_str = _seconds_to_hhmm(from_s)
        to_str = _seconds_to_hhmm(to_s)
        from_et = _broker_hhmm_to_et_str(from_s, offset_hours)
        to_et = _broker_hhmm_to_et_str(to_s, offset_hours)
        parts.append(f"{from_str}-{to_str} | {from_et}-{to_et} ET")
    return ", ".join(parts)


def _seconds_to_hhmm(seconds: int) -> str:
    total_min = (seconds // 60) % (24 * 60)
    h, m = divmod(total_min, 60)
    return f"{h:02d}:{m:02d}"


def _broker_hhmm_to_et_str(seconds_from_midnight: int, offset_hours: float) -> str:
    """Treat 'seconds_from_midnight' as broker-tz, convert to ET hh:mm."""
    today = datetime.now().date()
    h, m = divmod((int(seconds_from_midnight) // 60) % (24 * 60), 60)
    broker_dt = datetime.combine(today, dt_time(h, m)).replace(
        tzinfo=timezone(timedelta(hours=offset_hours))
    )
    return broker_dt.astimezone(ET).strftime("%H:%M")


def _report_d1_bars(mt5, sym: str, offset_hours: float) -> None:
    import MetaTrader5 as mt5_const

    rates = mt5.copy_rates_from_pos(sym, mt5_const.TIMEFRAME_D1, 0, 7)
    if rates is None or len(rates) == 0:
        print(f"  D1 fetch returned no bars ({mt5.last_error()})")
        return
    print("5. Last 7 D1 bars — `time` is bar OPEN, converted to all 3 tz:")
    print(f"   {'Broker-local naive':<22} {'UTC':<22} {'ET':<22} {'OHLC'}")
    for r in rates:
        broker_naive = datetime.utcfromtimestamp(int(r["time"]))
        utc_dt = broker_naive - timedelta(hours=offset_hours)
        et_dt = utc_dt.replace(tzinfo=timezone.utc).astimezone(ET)
        ohlc = f"O={r['open']:.2f} H={r['high']:.2f} L={r['low']:.2f} C={r['close']:.2f}"
        print(
            f"   {str(broker_naive):<22} "
            f"{str(utc_dt):<22} "
            f"{et_dt.strftime('%a %Y-%m-%d %H:%M %Z'):<22} "
            f"{ohlc}"
        )
    print()
    print("   ⮕ If the broker-local column shows '00:00' for every bar, the bar")
    print("     opens at BROKER MIDNIGHT (= the ET time in column 3). That ET")
    print("     time is when each D1 bar OPENS, and the bar of the prior date")
    print("     closed at that same moment.")


def _report_m1_gaps(mt5, sym: str, offset_hours: float) -> None:
    import MetaTrader5 as mt5_const

    rates = mt5.copy_rates_from_pos(sym, mt5_const.TIMEFRAME_M1, 0, M1_FETCH_COUNT)
    if rates is None or len(rates) < 100:
        n = 0 if rates is None else len(rates)
        print(f"6. M1 gap analysis: only {n} bars fetched, skipping ({mt5.last_error()})")
        return

    gaps: list[tuple[int, int, float]] = []
    for i in range(1, len(rates)):
        prev = int(rates[i - 1]["time"])
        curr = int(rates[i]["time"])
        delta_min = (curr - prev) / 60.0
        if GAP_MIN_MINUTES < delta_min < GAP_MAX_MINUTES:
            gaps.append((prev, curr, delta_min))

    print(
        f"6. M1 gap analysis — {len(rates)} bars fetched "
        f"(~{(int(rates[-1]['time']) - int(rates[0]['time'])) / 3600:.1f} hours of broker activity)"
    )
    if not gaps:
        print("   No gaps > 2 min < 6h found — symbol may be 24/x with no daily reset.")
        return

    # Most common gap-start minute = daily reset window for this symbol.
    start_minute_counter: Counter[int] = Counter()
    for prev, _curr, _dm in gaps:
        prev_dt = datetime.utcfromtimestamp(prev)
        start_minute_counter[prev_dt.hour * 60 + prev_dt.minute] += 1

    print(f"   {len(gaps)} gap(s) > 2 min < 6h. Last 5 (most recent):")
    for prev, curr, dm in gaps[-5:]:
        prev_naive = datetime.utcfromtimestamp(prev)
        curr_naive = datetime.utcfromtimestamp(curr)
        prev_et = (prev_naive - timedelta(hours=offset_hours)).replace(
            tzinfo=timezone.utc
        ).astimezone(ET)
        curr_et = (curr_naive - timedelta(hours=offset_hours)).replace(
            tzinfo=timezone.utc
        ).astimezone(ET)
        print(
            f"     gap of {dm:5.1f}min : "
            f"{prev_et.strftime('%a %m-%d %H:%M %Z')} -> "
            f"{curr_et.strftime('%a %m-%d %H:%M %Z')}"
        )

    top = start_minute_counter.most_common(3)
    print("   Most common gap-start times (broker tz, then ET):")
    for minute_of_day, count in top:
        h, m = divmod(minute_of_day, 60)
        broker_str = f"{h:02d}:{m:02d}"
        et_str = _broker_hhmm_to_et_str(minute_of_day * 60, offset_hours)
        print(f"     broker {broker_str}  =  {et_str} ET  ({count} occurrence(s))")

    # Compute median gap length at the most common reset minute (gives a
    # concrete "no-trade window length" estimate).
    if top:
        primary_min = top[0][0]
        same_minute_gaps = [
            dm
            for prev, _curr, dm in gaps
            if (datetime.utcfromtimestamp(prev).hour * 60
                + datetime.utcfromtimestamp(prev).minute) == primary_min
        ]
        if same_minute_gaps:
            same_minute_gaps.sort()
            median = same_minute_gaps[len(same_minute_gaps) // 2]
            print(
                f"   Typical no-trade window length at {_broker_hhmm_to_et_str(primary_min * 60, offset_hours)} ET: "
                f"median {median:.1f} min "
                f"(min {min(same_minute_gaps):.1f}, max {max(same_minute_gaps):.1f})"
            )
            print(
                f"   ⮕ To safely rebalance, schedule must be AT LEAST "
                f"{int(max(same_minute_gaps)) + 2} min after the gap starts "
                f"(or just use 17:05 ET if gap starts at 17:00 ET and is < 5 min)."
            )


if __name__ == "__main__":
    sys.exit(main())
