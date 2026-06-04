"""
Find true oldest tick date via binary search, then report total tick count.
Uses copy_ticks_range in a bisect to locate the first date with data.
"""
import sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import MetaTrader5 as mt5

ok = mt5.initialize()
if not ok:
    print("init failed:", mt5.last_error()); sys.exit(1)

print(f"Connected: {mt5.account_info().server}")
print()

# Known lower bounds from prior probing (saves binary search iterations)
KNOWN_LOWER = {
    "SP500":  datetime(2023, 1,  1, tzinfo=timezone.utc),
    "SPY":    datetime(2020, 1,  1, tzinfo=timezone.utc),
    "NDX":    datetime(2017, 1,  1, tzinfo=timezone.utc),
    "QQQ":    datetime(2020, 1,  1, tzinfo=timezone.utc),
    "XTIUSD": datetime(2019, 1,  1, tzinfo=timezone.utc),
    "XAUUSD": datetime(2017, 1,  1, tzinfo=timezone.utc),
    "GLD":    datetime(2020, 1,  1, tzinfo=timezone.utc),
    "XAGUSD": datetime(2017, 1,  1, tzinfo=timezone.utc),
    "SLV":    datetime(2020, 1,  1, tzinfo=timezone.utc),
}
ABSOLUTE_UPPER = datetime(2026, 1, 1, tzinfo=timezone.utc)


def has_ticks(sym: str, date: datetime) -> bool:
    """True if there are any ticks in the 7-day window starting at *date*."""
    ticks = mt5.copy_ticks_range(sym, date, date + timedelta(days=7), mt5.COPY_TICKS_ALL)
    return ticks is not None and len(ticks) > 0


def find_oldest_tick(sym: str) -> datetime | None:
    """Binary search to find the earliest month with tick data."""
    lo = KNOWN_LOWER.get(sym, datetime(2010, 1, 1, tzinfo=timezone.utc))
    hi = ABSOLUTE_UPPER

    # Quick check: does lo already have data?
    if not has_ticks(sym, lo):
        # Scan forward monthly until we find data
        cursor = lo
        while cursor < hi:
            if has_ticks(sym, cursor):
                hi = cursor
                break
            cursor += timedelta(days=30)
        else:
            return None  # no data found at all
        lo = cursor - timedelta(days=30)

    # Binary search at month granularity
    while (hi - lo).days > 31:
        mid = lo + (hi - lo) / 2
        if has_ticks(sym, mid):
            hi = mid
        else:
            lo = mid

    # Refine at week granularity
    cursor = lo
    while cursor < hi:
        if has_ticks(sym, cursor):
            return cursor
        cursor += timedelta(days=7)

    return hi


def count_ticks_from(sym: str, from_dt: datetime) -> int:
    """Estimate total tick count from from_dt to now by chunking."""
    now = datetime.now(timezone.utc)
    total = 0
    cursor = from_dt
    chunk = timedelta(days=30)
    while cursor < now:
        end = min(cursor + chunk, now)
        ticks = mt5.copy_ticks_range(sym, cursor, end, mt5.COPY_TICKS_ALL)
        total += len(ticks) if ticks is not None else 0
        cursor = end
    return total


all_syms = {s.name for s in (mt5.symbols_get() or [])}

TARGETS = [
    ("S&P 500",  "SP500",  "SPY"),
    ("Nasdaq",   "NDX",    "QQQ"),
    ("Oil/WTI",  "XTIUSD", None),
    ("Gold",     "XAUUSD", "GLD"),
    ("Silver",   "XAGUSD", "SLV"),
]

print(f"{'Instrument':<12} {'Symbol':<10} {'Type':<6} {'Oldest tick':>12}  {'>=2018?'}")
print("-" * 58)

results = {}
for label, cfd, etf in TARGETS:
    for sym, sym_type in [(cfd, "CFD"), (etf, "ETF")]:
        if sym is None:
            print(f"  {label:<10} {'N/A':<10} {sym_type:<6} {'N/A':>12}")
            continue
        if sym not in all_syms:
            print(f"  {label:<10} {sym:<10} {sym_type:<6} {'NOT IN TERMINAL':>12}")
            continue

        mt5.symbol_select(sym, True)
        print(f"  {label:<10} {sym:<10} {sym_type:<6}  searching...", end="", flush=True)
        t0 = time.perf_counter()
        oldest = find_oldest_tick(sym)
        elapsed = time.perf_counter() - t0

        if oldest is None:
            print(f"\r  {label:<10} {sym:<10} {sym_type:<6} {'no data':>12}")
            results[sym] = None
            continue

        oldest_str = oldest.date().isoformat()
        has_2018   = "YES" if oldest.year <= 2018 else f"NO  (starts {oldest.year})"
        print(f"\r  {label:<10} {sym:<10} {sym_type:<6} {oldest_str:>12}  {has_2018}  ({elapsed:.0f}s)")
        results[sym] = oldest

mt5.shutdown()
print()
print("Recommendation:")
for label, cfd, etf in TARGETS:
    cfd_oldest = results.get(cfd)
    etf_oldest = results.get(etf)
    options = [(cfd, cfd_oldest), (etf, etf_oldest)]
    options = [(s, d) for s, d in options if d is not None and s is not None]
    if not options:
        print(f"  {label}: no data found")
        continue
    best_sym, best_date = min(options, key=lambda x: x[1])
    print(f"  {label}: use {best_sym} (oldest: {best_date.date()})")
