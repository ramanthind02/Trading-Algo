"""Tick download time estimate for the 9-symbol target universe."""
from datetime import date

TODAY  = date(2026, 6, 4)
COLD_S = 12   # measured: seconds per cold broker chunk (10-18s range, 12s median)
TDY    = 260  # trading days per year (FX/metals/energy all 24/5)

SYMBOLS = [
    # (symbol, start_date, ticks_per_day_measured)
    ("NDX",    date(2018,  1, 24), 327_000),
    ("XAUUSD", date(2018,  1, 24), 144_000),
    ("XAGUSD", date(2018,  1, 24), 158_000),
    ("EURUSD", date(2011, 12, 19), 120_000),
    ("USDJPY", date(2011, 12, 19), 120_000),
    ("AUDNZD", date(2011, 12, 19), 130_000),
    ("EURCHF", date(2011, 12, 19),  93_000),
    ("XTIUSD", date(2019, 12, 27),  81_000),
    ("SP500",  date(2021, 12,  1),  84_000),
]

print("Tick bootstrap time estimates")
print("Measured latency: 12s per cold broker chunk  |  chunk = max ticks under 200k cap")
print()
print(f"  {'Symbol':<10} {'From':<12} {'Yrs':>5}  {'Chunk':>6}  {'Chunks':>8}  {'Hours':>7}  {'Nights@8h':>10}")
print("  " + "-" * 65)

rows = []
total_h = 0.0
for sym, start, tpd in SYMBOLS:
    years       = (TODAY - start).days / 365.25
    trading_days = years * TDY
    chunk_d     = max(1, 150_000 // tpd)   # adaptive sizing: target 150k ticks/call
    n_chunks    = trading_days / chunk_d
    hours       = n_chunks * COLD_S / 3600
    total_h    += hours
    rows.append((sym, start, years, chunk_d, n_chunks, hours))
    print(f"  {sym:<10} {str(start):<12} {years:>5.1f}  {chunk_d:>4}d  {n_chunks:>8.0f}  {hours:>7.1f}h  {hours/8:>9.1f}")

print("  " + "-" * 65)
print(f"  {'TOTAL':<10} {'':<12} {'':>5}  {'':>6}  {'':>8}  {total_h:>7.1f}h  {total_h/8:>9.1f}")

print()
print(f"  Wall time at 12s/chunk:  {total_h:.0f}h  ({total_h/8:.0f} overnight sessions of 8h)")
print(f"  Optimistic at 10s/chunk: {total_h*(10/12):.0f}h")
print(f"  Conservative at 18s/chunk: {total_h*(18/12):.0f}h")

print()
print("Suggested nightly schedule (shortest histories first):")
print()

NIGHTS = [
    ("Night 1",   ["SP500", "XTIUSD"],           "shortest histories — done in one night"),
    ("Night 2",   ["XAGUSD"],                     "silver"),
    ("Night 3",   ["XAUUSD"],                     "gold"),
    ("Night 4",   ["NDX"],                        "NDX (high tick density, needs own night)"),
    ("Night 5",   ["EURCHF", "USDJPY"],           "two FX pairs together"),
    ("Nights 6-7",["EURUSD"],                     "EURUSD alone — 14.5 years"),
    ("Nights 7-8",["AUDNZD"],                     "AUDNZD alone — 14.5 years"),
]

sym_h = {sym: h for sym, _, _, _, _, h in rows}

for night, syms, note in NIGHTS:
    h = sum(sym_h[s] for s in syms)
    cmd_syms = " ".join(syms)
    print(f"  {night:<12}  ~{h:.1f}h  {note}")
    print(f"             python -m scripts.fetch_mt5_data --ticks --symbols {cmd_syms}")
    print()

print("After bootstrap:")
print("  Daily incremental = 1-3 new chunks per symbol per day")
daily_chunks = len(SYMBOLS) * 2   # ~2 chunks/symbol/day average
daily_s = daily_chunks * COLD_S
print(f"  ~{daily_chunks} chunks/day across all 9 symbols = {daily_s}s = {daily_s/60:.1f} min/day")
