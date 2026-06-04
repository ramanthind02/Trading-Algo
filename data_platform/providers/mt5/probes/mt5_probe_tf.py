"""
Probe representative symbols across all MT5 timeframes + ticks.

Reports bar counts at each TF, oldest/newest D1 date, and tick history depth.
Writes results to data/mt5_data/_probe_results.json.

Run ONCE per session (IPC channel contention means back-to-back runs can fail):
  .\.venv\Scripts\python.exe scripts\_mt5_probe_tf.py
"""
from __future__ import annotations

import json, sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import MetaTrader5 as mt5

ok = mt5.initialize()
if not ok:
    print("initialize failed:", mt5.last_error()); sys.exit(1)

info = mt5.terminal_info()
acct = mt5.account_info()
print(f"build={info.build}  connected={info.connected}  login={acct.login if acct else 'N/A'}  server={acct.server if acct else 'N/A'}")
print()

all_syms = {s.name for s in (mt5.symbols_get() or [])}
print(f"Total symbols: {len(all_syms)}")

FAR_DATE  = datetime(2000, 1, 1, tzinfo=timezone.utc)
MAX_BARS  = 200_000
MAX_TICKS = 200_000

# Representative instruments with candidate symbol names
CANDIDATES: dict[str, list[str]] = {
    "EURUSD":    ["EURUSD"],
    "GBPUSD":    ["GBPUSD"],
    "USDJPY":    ["USDJPY"],
    "AUDUSD":    ["AUDUSD"],
    "USDCHF":    ["USDCHF"],
    "XAUUSD":    ["XAUUSD"],
    "XTIUSD":    ["XTIUSD", "USOIL", "WTI"],
    "SP500":     ["SP500", "SPX500", "US500"],
    "NAS100":    ["NAS100", "US100", "NDX100", "USTEC"],
    "GER40":     ["GER40", "GER30", "DAX40", "DE40"],
    "AAPL":      ["AAPL"],
    "MSFT":      ["MSFT"],
    "BTCUSD":    ["BTCUSD", "BTCUSDT", "BTC/USD"],
}

resolved: dict[str, str] = {}
for label, cands in CANDIDATES.items():
    for c in cands:
        if c in all_syms:
            resolved[label] = c
            break

print("\nSymbol resolution:")
for label, cands in CANDIDATES.items():
    sym = resolved.get(label, f"NOT FOUND (tried {cands})")
    print(f"  {label:<12} -> {sym}")
print()

TIMEFRAMES = [
    (mt5.TIMEFRAME_M1,  "M1"),
    (mt5.TIMEFRAME_M5,  "M5"),
    (mt5.TIMEFRAME_M15, "M15"),
    (mt5.TIMEFRAME_M30, "M30"),
    (mt5.TIMEFRAME_H1,  "H1"),
    (mt5.TIMEFRAME_H4,  "H4"),
    (mt5.TIMEFRAME_D1,  "D1"),
    (mt5.TIMEFRAME_W1,  "W1"),
    (mt5.TIMEFRAME_MN1, "MN1"),
]

# ── Bar depth table ──────────────────────────────────────────────────────────
hdr = f"\n{'Instrument':<12}"
for _, n in TIMEFRAMES:
    hdr += f" {n:>8}"
hdr += f"  {'Oldest D1':>12}  {'Newest D1':>12}  MT5 sym"
print(hdr)
print("-" * 140)

bar_results: dict[str, dict] = {}
for label, sym in resolved.items():
    mt5.symbol_select(sym, True)
    row_data: dict[str, int] = {}
    oldest_d1 = newest_d1 = "N/A"
    row = f"  {label:<10}"
    for tf_const, tf_name in TIMEFRAMES:
        rates = mt5.copy_rates_from(sym, tf_const, FAR_DATE, MAX_BARS)
        count = len(rates) if rates is not None else 0
        row_data[tf_name] = count
        row += f" {count:>8}"
        if tf_const == mt5.TIMEFRAME_D1 and rates is not None and len(rates) > 0:
            oldest_d1 = datetime.fromtimestamp(int(rates[0]["time"]),  tz=timezone.utc).date().isoformat()
            newest_d1 = datetime.fromtimestamp(int(rates[-1]["time"]), tz=timezone.utc).date().isoformat()
    row += f"  {oldest_d1:>12}  {newest_d1:>12}  {sym}"
    print(row, flush=True)
    bar_results[label] = {"symbol": sym, "bars": row_data, "oldest_d1": oldest_d1, "newest_d1": newest_d1}

# ── Tick depth ───────────────────────────────────────────────────────────────
print(f"\n\n{'Instrument':<12}  {'Ticks (from 2000)':>18}  {'Oldest tick':>12}  {'Newest tick':>12}  MT5 sym")
print("-" * 85)

tick_results: dict[str, dict] = {}
for label, sym in resolved.items():
    mt5.symbol_select(sym, True)
    ticks = mt5.copy_ticks_from(sym, FAR_DATE, MAX_TICKS, mt5.COPY_TICKS_ALL)
    count = len(ticks) if ticks is not None else 0
    oldest_t = newest_t = "N/A"
    if ticks is not None and count > 0:
        oldest_t = datetime.fromtimestamp(int(ticks[0]["time"]),  tz=timezone.utc).date().isoformat()
        newest_t = datetime.fromtimestamp(int(ticks[-1]["time"]), tz=timezone.utc).date().isoformat()
    print(f"  {label:<10}  {count:>18}  {oldest_t:>12}  {newest_t:>12}  {sym}", flush=True)
    tick_results[label] = {"symbol": sym, "tick_count": count, "oldest": oldest_t, "newest": newest_t}

mt5.shutdown()

# ── Write JSON ───────────────────────────────────────────────────────────────
out = _REPO_ROOT / "data" / "mt5_data" / "_probe_results.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "terminal_build": info.build,
    "total_symbols": len(all_syms),
    "bar_results": bar_results,
    "tick_results": tick_results,
}, indent=2), encoding="utf-8")
print(f"\nResults written to {out}")
print("done")
