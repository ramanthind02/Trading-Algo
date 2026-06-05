"""
Measure the broker round-trip time for data that hasn't been fetched before.
Uses AUDNZD 2012 data — less likely to be pre-cached than EURUSD.
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

# Test symbols and time ranges that are unlikely to be cached
COLD_TESTS = [
    ("AUDNZD", datetime(2012, 3, 5,  tzinfo=timezone.utc), datetime(2012, 3, 12, tzinfo=timezone.utc)),
    ("AUDNZD", datetime(2012, 3, 12, tzinfo=timezone.utc), datetime(2012, 3, 19, tzinfo=timezone.utc)),
    ("AUDNZD", datetime(2012, 3, 19, tzinfo=timezone.utc), datetime(2012, 3, 26, tzinfo=timezone.utc)),
    ("EURCHF",  datetime(2013, 6, 3,  tzinfo=timezone.utc), datetime(2013, 6, 10, tzinfo=timezone.utc)),
    ("EURCHF",  datetime(2013, 6, 10, tzinfo=timezone.utc), datetime(2013, 6, 17, tzinfo=timezone.utc)),
    ("XAUUSD", datetime(2014, 9, 1,  tzinfo=timezone.utc), datetime(2014, 9, 8,  tzinfo=timezone.utc)),
    ("XAUUSD", datetime(2014, 9, 8,  tzinfo=timezone.utc), datetime(2014, 9, 15, tzinfo=timezone.utc)),
]

print(f"{'Symbol':<10} {'Period':<25} {'Ticks':>10}  {'Sec':>8}  {'M ticks/min':>13}  Notes")
print("-" * 80)

for sym, start, end in COLD_TESTS:
    mt5.symbol_select(sym, True)
    t0 = time.perf_counter()
    ticks = mt5.copy_ticks_range(sym, start, end, mt5.COPY_TICKS_ALL)
    elapsed = time.perf_counter() - t0
    n = len(ticks) if ticks is not None else 0
    rate = n / elapsed / 1e6 * 60 if elapsed > 0.001 else 0
    period = f"{start.date()} - {end.date()}"
    note = "broker download" if elapsed > 1 else "cached"
    print(f"  {sym:<8} {period:<25} {n:>10,}  {elapsed:>8.3f}  {rate:>13.1f}  {note}")

print()
# Now re-fetch the same ranges (should be cached now)
print("Re-fetching same ranges (now cached):")
print(f"{'Symbol':<10} {'Period':<25} {'Ticks':>10}  {'Sec':>8}  {'M ticks/min':>13}")
print("-" * 75)
for sym, start, end in COLD_TESTS:
    mt5.symbol_select(sym, True)
    t0 = time.perf_counter()
    ticks = mt5.copy_ticks_range(sym, start, end, mt5.COPY_TICKS_ALL)
    elapsed = time.perf_counter() - t0
    n = len(ticks) if ticks is not None else 0
    rate = n / elapsed / 1e6 * 60 if elapsed > 0.001 else 0
    period = f"{start.date()} - {end.date()}"
    print(f"  {sym:<8} {period:<25} {n:>10,}  {elapsed:>8.3f}  {rate:>13.1f}")

mt5.shutdown()
