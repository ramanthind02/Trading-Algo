r"""Trading-calendar mismatch + holding-return drift across feeds (2018-2026).

- Calendar: which dates does the CFD trade that the future doesn't (and vice
  versa)? Holiday/session-calendar gaps mis-align a shared daily grid.
- Drift: annualized geometric return of each tradable series over the SAME
  window exposes dividend (ETF price vs TR), futures roll yield, and CFD basis.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import importlib.util
from research.feed_comparison import feed_compare as fc

WIN = ("2018-01-01", "2026-06-02")


def clip(s):
    s = s[(s.index >= pd.Timestamp(WIN[0])) & (s.index <= pd.Timestamp(WIN[1]))]
    return s.dropna()


def ann_ret(s):
    s = clip(s).astype(float)
    if len(s) < 50:
        return np.nan, np.nan, np.nan
    lr = np.log(s).diff().dropna()
    lr = lr[np.isfinite(lr)]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    return cagr, lr.mean() * 252, lr.std() * np.sqrt(252)


print("=" * 96)
print("TRADING-CALENDAR MISMATCH (2018+): dates present in one feed but not the other")
print("=" * 96)
pairs = [
    ("ES_fut", fc.s_ohlc("ES", "D"), "SP500_CFD", fc.s_mt5_snapshot("SP500")),
    ("ES_fut", fc.s_ohlc("ES", "D"), "SPY_CFD", fc.s_mt5_snapshot("SPY")),
    ("GC_fut", fc.s_ohlc("GC", "D"), "XAUUSD_CFD", fc.s_mt5_snapshot("XAUUSD")),
    ("GC_fut", fc.s_ohlc("GC", "D"), "GLD_CFD", fc.s_mt5_snapshot("GLD")),
    ("CL_fut", fc.s_ohlc("CL", "D"), "XTIUSD_CFD", fc.s_mt5_snapshot("XTIUSD")),
]
for an, a, bn, b in pairs:
    da = set(clip(a).index.normalize())
    db = set(clip(b).index.normalize())
    only_a = sorted(da - db)
    only_b = sorted(db - da)
    print(f"\n{an} vs {bn}:  common={len(da & db)}  {an}-only={len(only_a)}  {bn}-only={len(only_b)}")
    if only_a:
        print(f"   {an}-only (future trades, CFD flat) e.g.: "
              + ", ".join(str(d.date()) for d in only_a[:8]))
    if only_b:
        print(f"   {bn}-only (CFD trades, future flat) e.g.: "
              + ", ".join(str(d.date()) for d in only_b[:8]))

print("\n" + "=" * 96)
print("HOLDING-RETURN DRIFT (2018+, same window): CAGR / ann-log-ret / ann-vol")
print("=" * 96)
print(f"  {'series':<26}{'CAGR':>9}{'ann_logret':>12}{'ann_vol':>10}")
rows = [
    ("ES_fut_adj", fc.s_ohlc("ES", "D")),
    ("ES_fut_unadj", fc.s_ohlc_unadj("ES")),
    ("SPX_cash", fc.s_cash("us_indices", "SPX")),
    ("SPY_price(CAP)", fc.s_etf("S", "SPY", "CAP")),
    ("SPY_total_ret(TR)", fc.s_etf("S", "SPY", "TR")),
    ("SP500_CFD", fc.s_mt5_snapshot("SP500")),
    ("SPY_CFD", fc.s_mt5_snapshot("SPY")),
    ("--NQ--", pd.Series(dtype=float)),
    ("NQ_fut_adj", fc.s_ohlc("NQ", "D")),
    ("NDX_cash", fc.s_cash("us_indices", "NDX")),
    ("QQQ_price(CAP)", fc.s_etf("Q", "QQQ", "CAP")),
    ("QQQ_total_ret(TR)", fc.s_etf("Q", "QQQ", "TR")),
    ("QQQ_CFD", fc.s_mt5_snapshot("QQQ")),
    ("--GC--", pd.Series(dtype=float)),
    ("GC_fut_adj", fc.s_ohlc("GC", "D")),
    ("GLD_price(CAP)", fc.s_etf("G", "GLD", "CAP")),
    ("XAUUSD_CFD", fc.s_mt5_snapshot("XAUUSD")),
    ("GLD_CFD", fc.s_mt5_snapshot("GLD")),
]
for name, s in rows:
    if s.empty:
        print(f"  {name}")
        continue
    cagr, alr, av = ann_ret(s)
    print(f"  {name:<26}{cagr*100:>8.2f}%{alr*100:>11.2f}%{av*100:>9.2f}%")
