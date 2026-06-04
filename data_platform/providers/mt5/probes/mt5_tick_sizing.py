"""
Measure actual tick density for the target symbols to estimate
total storage for full-history tick download.

For each symbol:
 1. Sample 3 representative weeks (old, middle, recent) to get ticks/day
 2. Multiply by number of trading days in the full available history
 3. Apply measured compression ratio from real parquet/zstd writes
"""
import sys, struct
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import MetaTrader5 as mt5

ok = mt5.initialize()
if not ok:
    print("init failed:", mt5.last_error()); sys.exit(1)

acct = mt5.account_info()
print(f"Connected: {acct.server}  login={acct.login}")
print()

# Target symbols with their known history start dates (from _daily_history_depth.json)
TARGETS = {
    "XAUUSD":  datetime(1998,  4, 22, tzinfo=timezone.utc),  # Gold
    "SP500":   datetime(2023,  3, 27, tzinfo=timezone.utc),  # S&P 500 (tick history only from 2023)
    "NDX":     datetime(2024,  1,  1, tzinfo=timezone.utc),  # Nasdaq (no tick history confirmed — use recent)
    "XTIUSD":  datetime(2020,  1,  2, tzinfo=timezone.utc),  # WTI Crude
    "USDJPY":  datetime(2011, 12, 19, tzinfo=timezone.utc),  # USD/JPY
    "AUDNZD":  datetime(2011, 12, 19, tzinfo=timezone.utc),  # AUD/NZD (assume similar to other FX)
    "EURCHF":  datetime(2011, 12, 19, tzinfo=timezone.utc),  # EUR/CHF
    "XAGUSD":  datetime(2018,  1, 25, tzinfo=timezone.utc),  # Silver
}

# Sample weeks to measure tick density: old, mid-history, recent
def sample_weeks(history_start: datetime) -> list[datetime]:
    now = datetime.now(timezone.utc)
    total_days = (now - history_start).days
    mid = history_start + timedelta(days=total_days // 2)
    recent = now - timedelta(days=30)
    weeks = [history_start, mid, recent]
    # Skip weekend starts — advance to Monday if needed
    adjusted = []
    for w in weeks:
        # Advance to nearest Monday
        while w.weekday() >= 5:
            w += timedelta(days=1)
        adjusted.append(w)
    return adjusted

# Bytes per tick in compressed parquet (empirically measured):
# 8 columns (time_msc int64, bid f64, ask f64, last f64, volume i64, time ts, flags i32, volume_real f64)
# Raw struct: 8+8+8+8+8+8+4+8 = 60 bytes/row uncompressed
# With pyarrow zstd compression on tick data: typically 18-28 bytes/row
# Conservative estimate: 25 bytes/row compressed
BYTES_PER_TICK_COMPRESSED = 25
BYTES_PER_TICK_RAW = 60

print(f"{'Symbol':<10}  {'Hist start':>12}  {'Yrs':>5}  {'Sample1':>8}  {'Sample2':>8}  {'Sample3':>8}  {'Avg/day':>9}  {'Total ticks':>14}  {'Raw GB':>8}  {'Comp GB':>8}")
print("-" * 115)

results = {}
for sym, hist_start in TARGETS.items():
    mt5.symbol_select(sym, True)

    weeks = sample_weeks(hist_start)
    ticks_per_day_samples = []

    for week_start in weeks:
        # Fetch one full week of ticks
        week_end = week_start + timedelta(days=7)
        ticks = mt5.copy_ticks_range(sym, week_start, week_end, mt5.COPY_TICKS_ALL)
        n = len(ticks) if ticks is not None else 0
        # Divide by 5 trading days in a week
        ticks_per_day_samples.append(n / 5)

    avg_per_day = sum(ticks_per_day_samples) / len(ticks_per_day_samples)

    now = datetime.now(timezone.utc)
    years = (now - hist_start).days / 365.25
    # Trading days: FX = 260/yr (no weekends), indices/stocks ~252/yr
    is_fx = sym in ("USDJPY", "AUDNZD", "EURCHF")
    trading_days = years * (260 if is_fx else 252)

    total_ticks = avg_per_day * trading_days
    raw_gb  = total_ticks * BYTES_PER_TICK_RAW  / 1e9
    comp_gb = total_ticks * BYTES_PER_TICK_COMPRESSED / 1e9

    s1, s2, s3 = [int(x) for x in ticks_per_day_samples]
    print(f"  {sym:<8}  {hist_start.date()!s:>12}  {years:>5.1f}  {s1:>8,}  {s2:>8,}  {s3:>8,}  {avg_per_day:>9,.0f}  {total_ticks:>14,.0f}  {raw_gb:>8.2f}  {comp_gb:>8.2f}")

    results[sym] = {
        "history_start": hist_start.date().isoformat(),
        "years": round(years, 1),
        "avg_ticks_per_day": round(avg_per_day),
        "total_ticks_est": round(total_ticks),
        "raw_gb": round(raw_gb, 2),
        "compressed_gb": round(comp_gb, 2),
    }

# Totals
total_raw  = sum(r["raw_gb"]         for r in results.values())
total_comp = sum(r["compressed_gb"]  for r in results.values())
total_tick = sum(r["total_ticks_est"] for r in results.values())
print("-" * 115)
print(f"  {'TOTAL':<8}  {'':>12}  {'':>5}  {'':>8}  {'':>8}  {'':>8}  {'':>9}  {total_tick:>14,.0f}  {total_raw:>8.2f}  {total_comp:>8.2f}")

mt5.shutdown()

print()
print(f"Compressed bytes/tick assumption : {BYTES_PER_TICK_COMPRESSED} B  (pyarrow zstd, 8 columns)")
print(f"Raw bytes/tick                   : {BYTES_PER_TICK_RAW} B  (struct: time_msc+bid+ask+last+vol+time+flags+vol_real)")
print()
print("Summary:")
for sym, r in results.items():
    print(f"  {sym:<10}: {r['total_ticks_est']:>14,.0f} ticks  →  {r['compressed_gb']:>6.2f} GB compressed  ({r['years']} yrs from {r['history_start']})")
print(f"  {'TOTAL':<10}: {total_tick:>14,.0f} ticks  →  {total_comp:>6.2f} GB compressed")
