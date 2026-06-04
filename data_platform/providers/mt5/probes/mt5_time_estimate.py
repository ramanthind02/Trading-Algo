"""Revised download time estimates based on measured 12s cold latency per chunk."""
data = [
    # (symbol, years, ticks_per_day, is_fx)
    ("EURUSD",  14.5,  120_000, True),
    ("USDJPY",  14.5,  120_000, True),
    ("AUDNZD",  14.5,  130_000, True),
    ("EURCHF",  14.5,   93_000, True),
    ("XAUUSD",  28.1,  144_000, False),
    ("XAGUSD",   8.4,  158_000, False),
    ("XTIUSD",   6.4,   81_000, False),
    ("SP500",    3.2,   84_000, False),
    ("NDX",      2.4,  327_000, False),
]

COLD_S   = 12    # measured: seconds per broker chunk download
TDY_FX   = 260
TDY_EQ   = 252
BYTES_C  = 25    # zstd compressed bytes per tick

print("Revised time estimates — measured 12s cold latency per broker chunk")
print()
hdr = f"  {'Symbol':<10}  {'Yrs':>5}  {'Chunk':>7}  {'Chunks':>8}  {'Hours':>7}  {'GB':>7}"
print(hdr)
print("  " + "-" * 55)

total_h = 0.0
total_gb = 0.0
total_ticks = 0

for sym, years, tpd, is_fx in data:
    tdy = TDY_FX if is_fx else TDY_EQ
    days = years * tdy
    chunk_d = max(1, 150_000 // tpd)   # adaptive sizing: target 150k ticks/call
    n_chunks = days / chunk_d
    hours = n_chunks * COLD_S / 3600
    comp_gb = tpd * days * BYTES_C / 1e9
    total_h  += hours
    total_gb += comp_gb
    total_ticks += tpd * days
    print(f"  {sym:<10}  {years:>5.1f}  {chunk_d:>5}d  {n_chunks:>8,.0f}  {hours:>7.1f}  {comp_gb:>7.1f}")

print("  " + "-" * 55)
print(f"  {'TOTAL':<10}  {'':>5}  {'':>7}  {'':>8}  {total_h:>7.1f}  {total_gb:>7.1f}")
print()
print(f"Cold download: {total_h:.0f}h total  ({total_h/8:.0f} overnight sessions of 8h)")
print()
print("After pre-warming MT5 terminal cache (one-time MQL5 EA scroll):")
print("  Re-reads run at ~500M ticks/min (measured)")
warm_min = total_ticks / 500e6
print(f"  {total_ticks/1e9:.2f}B ticks -> {warm_min:.1f} min to read everything")
print()
print("Practical recommendation:")
print("  1. Let data_platform.providers.mt5.scraper run overnight per symbol (cold broker download)")
print("  2. Each subsequent run is near-instant (local cache hit)")
print("  3. Daily incremental adds ~1-3 chunks/symbol/day -> < 1 min/symbol/day")
