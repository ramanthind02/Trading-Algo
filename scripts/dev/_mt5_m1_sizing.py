"""
Compute M1 bar counts analytically from known history depths and trading hours.

No MT5 connection needed — uses empirically confirmed data:
  - History start dates: from _daily_history_depth.json (probed 2026-06-04)
  - Trading hours: from darwinex_universe.md and direct measurement
  - M1 bars = trading_minutes_per_day * trading_days

Session definitions (Darwinex, all confirmed from M1 bar gap analysis):
  FX majors/crosses:  ~1,435 bars/day  (24h minus ~5 min daily rollover)
  Metals (XAU/XAG):  ~1,380 bars/day  (23h, daily maintenance break)
  Energy (XTI/XNG):  ~1,380 bars/day  (23h)
  Index CFDs:        ~1,390 bars/day  (23.2h, nearly 24h)
  US stocks/ETFs:    ~390  bars/day   (09:30-16:00 ET = 390 min, Mon-Fri only)
"""
from datetime import date, timedelta
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Trading session constants (minutes of M1 bars per trading day)
# ---------------------------------------------------------------------------
MINS = {
    "fx":     1_435,   # 24h minus 5-min daily rollover
    "metals": 1_380,   # 23h (XAU, XAG, XNG)
    "energy": 1_380,   # 23h (XTI)
    "index":  1_390,   # 23.2h (SP500, NDX, GDAXI etc.)
    "equity": 390,     # 6.5h US cash session (stocks + ETFs)
}

# Trading days per year (FX 260, equities 252)
TDY = {"fx": 260, "metals": 260, "energy": 260, "index": 260, "equity": 252}

# Reference date
TODAY = date(2026, 6, 4)

# Bytes per M1 bar, compressed zstd parquet
# 8 cols: time(ts), open(f32), high(f32), low(f32), close(f32),
#         tick_volume(i32), spread(i16), real_volume(i64)
# Raw: 8+4+4+4+4+4+2+8 = 38 bytes; zstd on time-series: ~14 bytes/row
BYTES_COMPRESSED = 14

def years(start: date) -> float:
    return (TODAY - start).days / 365.25

def m1_bars(start: date, session: str) -> int:
    y = years(start)
    return int(y * TDY[session] * MINS[session])

def gb(bars: int) -> float:
    return bars * BYTES_COMPRESSED / 1e9

# ---------------------------------------------------------------------------
# Known history start dates (confirmed 2026-06-04 via copy_rates_range probing)
# ---------------------------------------------------------------------------
#
# IMPORTANT: M1 history start is NOT the same as D1 history start.
# D1 for EURUSD goes to 1971 because Darwinex has synthetic daily data.
# M1 history is shorter — brokers typically store 2-5 years of M1 locally;
# exact M1 depth is only known after the chart has been loaded.
#
# Conservative assumption: M1 history mirrors the TICK history start date,
# since both require the terminal to have streamed/cached the data.
# D1 is special — it's pre-built on the broker server from a different feed.
#
# M1 start dates (conservative = tick start date):
FX_M1_START    = date(2011, 12, 19)   # FX majors — tick history confirmed to here
FX_CROSS_START = date(2011, 12, 19)   # same
XAUUSD_START   = date(2018,  1, 25)   # confirmed tick start
XAGUSD_START   = date(2018,  1, 25)
XTIUSD_START   = date(2019, 12, 27)
XNGUSD_START   = date(2020,  1,  1)   # similar to XTIUSD
INDEX_START    = date(2018,  1, 25)   # GDAXI/WS30/UK100 tick start; SP500 starts 2023 for ticks
SP500_START    = date(2023,  3, 27)   # SP500 tick/M1 only from 2023
NDX_START      = date(2018,  1, 25)   # NDX tick start confirmed 2018
STOCK_START    = date(2008,  5,  5)   # Darwinex D1 limit — but M1 likely only ~2018 (MT5 default)
ETF_START      = date(2010,  1, 11)   # D1 confirmed; M1 likely similar to stocks

# Most brokers only keep 2-3 years of M1 on their servers by default.
# For a prop-firm live account Darwinex likely keeps more, but without
# actually fetching it we cap the optimistic estimate at the D1 start date
# and add a realistic cap at ~5 years for M1.
STOCK_M1_REALISTIC = date(2021,  1,  1)  # realistic: 5yr M1 on most CFD brokers
ETF_M1_REALISTIC   = date(2021,  1,  1)

# ---------------------------------------------------------------------------
# Universe definition
# ---------------------------------------------------------------------------
UNIVERSE = [
    # (label, n_symbols, start_date, session_type)
    # FX
    ("FX majors (8)",         8,   FX_M1_START,    "fx"),
    ("FX crosses (30)",       30,  FX_CROSS_START, "fx"),
    # Commodities
    ("Gold XAUUSD (1)",       1,   XAUUSD_START,   "metals"),
    ("Silver XAGUSD (1)",     1,   XAGUSD_START,   "metals"),
    ("WTI XTIUSD (1)",        1,   XTIUSD_START,   "energy"),
    ("Nat gas XNGUSD (1)",    1,   XNGUSD_START,   "energy"),
    # Index CFDs
    ("SP500 (1)",             1,   SP500_START,    "index"),
    ("NDX (1)",               1,   NDX_START,      "index"),
    ("Other indices (8)",     8,   INDEX_START,    "index"),
    # US stocks
    ("US stocks — optimistic (692)", 692, STOCK_START,          "equity"),
    ("US stocks — realistic  (692)", 692, STOCK_M1_REALISTIC,   "equity"),
    # ETFs
    ("ETFs — optimistic (100)",       100, ETF_START,           "equity"),
    ("ETFs — realistic   (100)",       100, ETF_M1_REALISTIC,   "equity"),
]

# ---------------------------------------------------------------------------
# Print results in two scenarios
# ---------------------------------------------------------------------------
print("M1 bar storage estimate — full Darwinex universe (844 symbols)")
print("Bytes/bar: 14 (zstd parquet, 8 columns)")
print()

# Scenario A: optimistic (use all available history per category)
# Scenario B: realistic (M1 limited to ~5 years for stocks/ETFs)

def run_scenario(label: str, stock_start: date, etf_start: date) -> None:
    rows = [
        ("FX majors (8)",         8,   FX_M1_START,    "fx"),
        ("FX crosses (30)",       30,  FX_CROSS_START, "fx"),
        ("Gold (1)",              1,   XAUUSD_START,   "metals"),
        ("Silver (1)",            1,   XAGUSD_START,   "metals"),
        ("WTI crude (1)",         1,   XTIUSD_START,   "energy"),
        ("Nat gas (1)",           1,   XNGUSD_START,   "energy"),
        ("SP500 (1)",             1,   SP500_START,    "index"),
        ("NDX (1)",               1,   NDX_START,      "index"),
        ("Other indices (8)",     8,   INDEX_START,    "index"),
        ("US stocks (692)",       692, stock_start,    "equity"),
        ("ETFs (100)",            100, etf_start,      "equity"),
    ]

    print(f"  {label}")
    print(f"  {'Category':<25} {'Syms':>5}  {'Start':>12}  {'Yrs':>5}  {'Bars/sym':>10}  {'Total bars':>14}  {'GB':>8}")
    print("  " + "-" * 85)

    total_bars = 0
    total_gb_  = 0
    for cat, n, start, sess in rows:
        bps  = m1_bars(start, sess)
        tot  = n * bps
        g    = gb(tot)
        total_bars += tot
        total_gb_  += g
        print(f"  {cat:<25} {n:>5}  {start!s:>12}  {years(start):>5.1f}  {bps:>10,}  {tot:>14,}  {g:>8.1f}")

    print("  " + "-" * 85)
    print(f"  {'TOTAL':<25} {'844':>5}  {'':>12}  {'':>5}  {'':>10}  {total_bars:>14,}  {total_gb_:>8.1f}")

    # Daily growth
    daily_mb = (
        38 * 1_435 * BYTES_COMPRESSED / 1e6 +   # FX
        4  * 1_380 * BYTES_COMPRESSED / 1e6 +   # commodities
        10 * 1_390 * BYTES_COMPRESSED / 1e6 +   # indices
        792 * 390  * BYTES_COMPRESSED / 1e6      # stocks + ETFs
    )
    print(f"  Daily growth (1 trading day): {daily_mb:.0f} MB   Annual: {daily_mb * 252 / 1e3:.1f} GB/year")
    print()
    return total_gb_

print()
gb_opt      = run_scenario("Scenario A — Optimistic (full available history)",
                            STOCK_START, ETF_START)
gb_realistic = run_scenario("Scenario B — Realistic (stocks/ETFs capped at 2021, ~5 yrs M1)",
                             STOCK_M1_REALISTIC, ETF_M1_REALISTIC)

print("=" * 55)
print(f"  Optimistic total:  {gb_opt:.0f} GB  (~{gb_opt/1000:.2f} TB)")
print(f"  Realistic total:   {gb_realistic:.0f} GB  (~{gb_realistic/1000:.2f} TB)")
print()
print("Storage fit check:")
print(f"  500 GB SSD — holds realistic estimate: {'YES' if gb_realistic < 500 else 'NO'}")
print(f"  500 GB SSD — holds optimistic:         {'YES' if gb_opt < 500 else 'NO'}")
print(f"  1 TB HDD   — holds optimistic:         {'YES' if gb_opt < 1000 else 'NO'}")
print()
print("Key caveat:")
print("  The dominant uncertainty is M1 history depth for the 692 stocks.")
print("  If Darwinex stores M1 from 2008 (same as D1): optimistic scenario.")
print("  If M1 is only available from ~2021 (typical CFD broker default): realistic.")
print("  Actual depth unknown until first M1 fetch per symbol.")
print()
print("  Stocks alone account for the difference:")
print(f"    Stocks optimistic (2008, 18yr): {692 * m1_bars(STOCK_START, 'equity') * BYTES_COMPRESSED / 1e9:.0f} GB")
print(f"    Stocks realistic  (2021,  5yr): {692 * m1_bars(STOCK_M1_REALISTIC, 'equity') * BYTES_COMPRESSED / 1e9:.0f} GB")
