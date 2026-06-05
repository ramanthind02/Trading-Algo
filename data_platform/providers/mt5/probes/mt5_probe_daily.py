"""
Probe D1/W1/MN1 history depth using copy_rates_range (the working API call).

copy_rates_range(sym, FAR_DATE, now) triggers a broker download on first call,
then reads from local cache on subsequent calls. This is the correct approach.

Run this ONCE — it will download and cache history for all listed symbols.
Subsequent runs (and data_platform.providers.mt5.scraper) will be instant for cached symbols.

Results written to data/mt5_data/_daily_history_depth.json
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5

ok = mt5.initialize()
if not ok:
    print("init failed:", mt5.last_error()); sys.exit(1)

acct = mt5.account_info()
print(f"Connected: login={acct.login}  server={acct.server}")
print("NOTE: First run downloads history from broker — may take 30-90s per symbol.")
print()

FAR  = datetime(1970, 1, 1, tzinfo=timezone.utc)
NOW  = datetime.now(timezone.utc)
MAX  = 200_000

SYMBOLS = [
    # FX — expected deep history
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "NZDUSD", "EURAUD", "EURCAD",
    # Indices
    "SP500", "NDX", "GDAXI", "WS30", "UK100", "FCHI40", "STOXX50E",
    "NI225", "AUS200",
    # Commodities
    "XAUUSD", "XTIUSD", "XAGUSD", "XNGUSD",
    # US mega-cap stocks
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA",
    "JPM", "V", "MA", "UNH", "XOM", "CVX", "JNJ", "WMT", "KO",
    "IBM", "GS", "HD", "PG",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB",
]

all_syms = {s.name for s in (mt5.symbols_get() or [])}

print(f"{'Symbol':<12} {'D1 bars':>8}  {'Oldest D1':>12}  {'Newest D1':>12}  {'W1':>6}  {'MN1':>6}  {'Yrs':>5}")
print("-" * 76)

results = {}
for sym in SYMBOLS:
    if sym not in all_syms:
        print(f"  {sym:<10}  NOT IN TERMINAL")
        continue

    mt5.symbol_select(sym, True)

    d1  = mt5.copy_rates_range(sym, mt5.TIMEFRAME_D1,  FAR, NOW)
    w1  = mt5.copy_rates_range(sym, mt5.TIMEFRAME_W1,  FAR, NOW)
    mn1 = mt5.copy_rates_range(sym, mt5.TIMEFRAME_MN1, FAR, NOW)

    d1_n  = len(d1)  if d1  is not None else 0
    w1_n  = len(w1)  if w1  is not None else 0
    mn1_n = len(mn1) if mn1 is not None else 0

    oldest = newest = yrs = "N/A"
    if d1_n > 0:
        oldest = datetime.fromtimestamp(int(d1[0]["time"]),  tz=timezone.utc).date().isoformat()
        newest = datetime.fromtimestamp(int(d1[-1]["time"]), tz=timezone.utc).date().isoformat()
        yrs = f"{d1_n / 252:.1f}"
    else:
        oldest = newest = "no data"
        yrs = "0"

    print(f"  {sym:<10} {d1_n:>8}  {oldest:>12}  {newest:>12}  {w1_n:>6}  {mn1_n:>6}  {yrs:>5}", flush=True)

    results[sym] = {
        "d1_bars": d1_n, "w1_bars": w1_n, "mn1_bars": mn1_n,
        "oldest_d1": oldest, "newest_d1": newest,
        "approx_years": float(yrs) if yrs != "0" and yrs != "N/A" else 0,
    }

mt5.shutdown()

out = Path(__file__).resolve().parents[1] / "data" / "mt5_data" / "_daily_history_depth.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "generated_utc": NOW.isoformat(),
    "note": "Uses copy_rates_range(FAR, NOW) — triggers broker download on first call",
    "results": results,
}, indent=2), encoding="utf-8")
print(f"\nResults written to {out}")
print("done")
