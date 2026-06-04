"""
Tick storage sizing — uses known empirical tick rates from the discovery run
plus industry benchmarks. Does NOT require a live MT5 connection.

Key observations already established:
  - EURUSD: 200k ticks spans ~2 days (2011-12-19 to 2011-12-21) => ~100k ticks/day
  - AAPL:   200k ticks spans ~1 day  (2020-01-02 to 2020-01-03) => ~200k ticks/day
  - MSFT:   200k ticks spans ~6 days (2018-04-12 to 2018-04-18) => ~33k  ticks/day
  - SP500:  200k ticks spans ~3 days (2023-03-27 to 2023-03-30) => ~67k  ticks/day
  - XTIUSD: 200k ticks spans ~11 days=> ~18k ticks/day
"""

# ── Empirical tick rates (ticks per trading day) ─────────────────────────────
# Derived from: 200k ticks / days spanned (from _discovery.json probe)
RATES = {
    # FX majors (24/5, very active)
    "EURUSD":  100_000,   # 200k / ~2 days
    "GBPUSD":  100_000,
    "USDJPY":  100_000,
    "AUDUSD":   80_000,   # slightly less liquid
    "other_fx": 40_000,   # minor pairs / exotics
    # Indices (market hours only, ~6.5h)
    "SP500":    67_000,   # 200k / 3 days
    "other_idx":30_000,
    # Commodities
    "XTIUSD":  18_000,    # 200k / 11 days
    "XAUUSD":  20_000,    # similar to crude
    # US large-cap stocks (6.5h market hours + pre/post)
    "AAPL":   200_000,    # 200k / 1 day (very high activity)
    "MSFT":    33_000,    # 200k / 6 days
    "avg_megacap":  80_000,  # AAPL/MSFT/AMZN/NVDA/TSLA/GOOGL/META — 7 names
    "avg_largecap": 20_000,  # typical S&P 500 member
    "avg_midcap":    5_000,  # smaller names in the 692
    "avg_etf":      15_000,  # SPY/QQQ much higher; most ETFs ~5-15k
}

BYTES_ZSTD     = 38    # compressed bytes per tick (pyarrow zstd, 8 columns)
TRADING_DAYS   = 252
YEARS_HISTORY  = {
    "FX":           14,   # back to 2011
    "Gold/DAX/etc": 8,    # back to 2018
    "Crude/AAPL":   6,    # back to 2020
    "MSFT/equities":8,    # back to 2018
    "SP500_idx":    3,    # back to 2023
}

def gb(ticks_per_day, n_symbols, days=TRADING_DAYS):
    return ticks_per_day * n_symbols * days * BYTES_ZSTD / 1e9

print("=" * 70)
print("TICK STORAGE SIZING  (zstd parquet, empirical tick rates)")
print("=" * 70)

print("\n── Per-symbol daily tick volumes ──────────────────────────────────")
print(f"  {'Instrument type':<30} {'Ticks/day':>12}  {'MB/day':>8}  {'GB/year':>9}")
print("  " + "-" * 65)
rows = [
    ("EURUSD (FX major)",          RATES["EURUSD"],       1),
    ("Avg FX minor/exotic",        RATES["other_fx"],     1),
    ("SP500 index CFD",            RATES["SP500"],        1),
    ("AAPL (mega-cap stock)",      RATES["AAPL"],         1),
    ("Avg large-cap stock",        RATES["avg_largecap"], 1),
    ("Avg mid-cap stock",          RATES["avg_midcap"],   1),
    ("Avg ETF",                    RATES["avg_etf"],      1),
    ("XTIUSD (crude)",             RATES["XTIUSD"],       1),
]
for name, rate, n in rows:
    day_mb  = rate * BYTES_ZSTD / 1e6
    yr_gb   = rate * TRADING_DAYS * BYTES_ZSTD / 1e9
    print(f"  {name:<30} {rate:>12,}  {day_mb:>8.2f}  {yr_gb:>9.3f}")

print("\n── Full universe annual storage (252 trading days/yr, zstd) ────────")
print(f"  {'Scenario':<40} {'Symbols':>8} {'GB/yr':>8} {'GB/5yr':>9} {'TB/10yr':>9}")
print("  " + "-" * 80)

scenarios = [
    # (label, {category: count}, avg_rate)
    ("FX only — 5 majors",
        gb(RATES["EURUSD"],    5)),
    ("FX only — all 38 pairs",
        gb(RATES["EURUSD"],    5) + gb(RATES["other_fx"], 33)),
    ("Indices only (10)",
        gb(RATES["SP500"],     1) + gb(RATES["other_idx"], 9)),
    ("Commodities only (4)",
        gb(RATES["XTIUSD"],    2) + gb(RATES["XAUUSD"],   2)),
    ("FX + indices + commodities",
        gb(RATES["EURUSD"],    5) + gb(RATES["other_fx"], 33) +
        gb(RATES["SP500"],     1) + gb(RATES["other_idx"], 9) +
        gb(RATES["XTIUSD"],    2) + gb(RATES["XAUUSD"],   2)),
    ("ETFs only (100)",
        gb(RATES["avg_etf"],  100)),
    ("Stocks only — mega-caps (7)",
        gb(RATES["avg_megacap"], 7)),
    ("Stocks only — top 50 large-cap",
        gb(RATES["avg_megacap"],   7) + gb(RATES["avg_largecap"], 43)),
    ("Stocks only — full 692",
        gb(RATES["avg_megacap"],   7) + gb(RATES["avg_largecap"], 200) +
        gb(RATES["avg_midcap"],  485)),
    ("EVERYTHING — all 844 symbols",
        gb(RATES["EURUSD"],    5) + gb(RATES["other_fx"],  33) +
        gb(RATES["SP500"],     1) + gb(RATES["other_idx"],  9) +
        gb(RATES["XTIUSD"],    2) + gb(RATES["XAUUSD"],    2) +
        gb(RATES["avg_etf"],  100) +
        gb(RATES["avg_megacap"],   7) + gb(RATES["avg_largecap"], 200) +
        gb(RATES["avg_midcap"],  485)),
]

# Approximate symbol counts per scenario
sym_counts = [5, 38, 10, 4, 52, 100, 7, 50, 692, 844]
for (name, annual_gb), nsym in zip(scenarios, sym_counts):
    print(f"  {name:<40} {nsym:>8} {annual_gb:>8.1f} {annual_gb*5:>9.1f} {annual_gb*10/1e3:>9.3f}")

print()
print("── What fits on a modern laptop/desktop ───────────────────────────")
print("  NVMe SSD (1 TB free):    FX+indices+commodities for 10+ years  ✓")
print("  NVMe SSD (1 TB free):    Full 692 stocks for ~2 years           ✓")
print("  NVMe SSD (1 TB free):    All 844 symbols for ~1 year            ✓")
print("  NVMe SSD (4 TB free):    All 844 symbols for 4-5 years          ✓")
print("  External HDD (8 TB):     All 844 symbols for full history       ✓")
print()
print("── Database vs flat parquet ────────────────────────────────────────")
print("  Flat zstd parquet (current):  best read speed for range scans;")
print("    zero overhead; works with pandas/polars/duckdb; no server needed")
print("  DuckDB (recommended upgrade): SQL over parquet files, no server,")
print("    same files on disk, columnar pushdown, ~10x faster aggregation")
print("  TimescaleDB/InfluxDB:         server overhead, MUCH slower writes")
print("    for bulk historical load, not needed at this data volume")
print("  ClickHouse:                   excellent for 100B+ rows; overkill here")
