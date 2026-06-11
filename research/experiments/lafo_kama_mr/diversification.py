"""Does LAFO actually diversify a trend/breakout book? (the user's stated goal)

LAFO = long-only deep-dip KAMA MR (this study). Trend proxy = an opening-range
BREAKOUT on the same NDX RTH sessions (reusing the orb_ibs engine, IBS filter off) —
a stand-in for the user's Donchian/Keltner breakout EAs. We compare daily return
series: correlation, and whether a 50/50 blend lifts the combined Sharpe.

Run:  .\.venv\Scripts\python.exe -m research.experiments.lafo_kama_mr.diversification
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.lafo_kama_mr import engine as E
from research.experiments.orb_ibs_nas100 import poc as ORB

Y0, Y1 = 2018, 2026
OUT = "research/experiments/lafo_kama_mr/outputs"
TD = E.TRADING_DAYS


def _sharpe(a: np.ndarray) -> float:
    return float(a.mean() / a.std() * np.sqrt(TD)) if a.std() > 0 else 0.0


def _maxdd(a: np.ndarray) -> float:
    eq = np.cumsum(a)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0


def lafo_daily() -> pd.Series:
    p = E.Params(tf="M15", session="rth", entry_mode="rel", threshold=0.025,
                 exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False,
                 use_recorded_spread=True, slip_pts=1.0)
    bars = E.prep_bars("NDX", Y0, Y1, p)
    _, tr = E.run("NDX", Y0, Y1, p)
    sd = np.sort(bars["day"].unique())
    s = pd.Series(0.0, index=pd.Index(sd, name="day"))
    if not tr.empty:
        s.loc[tr.groupby("day")["ret"].sum().index] = tr.groupby("day")["ret"].sum().values
    return s


def breakout_daily() -> pd.Series:
    """ORB-only breakout (no IBS), recorded spread — trend/breakout proxy on NDX RTH."""
    win = ORB.load_sessions(2018)
    p = ORB.Params(use_ibs=False, longs=True, shorts=True, rr=3.0, use_recorded_spread=True)
    tr = ORB.simulate(win, p)
    sd = np.sort(win["day"].unique())
    s = pd.Series(0.0, index=pd.Index(sd, name="day"))
    if not tr.empty:
        s.loc[tr.groupby("day")["ret"].sum().index] = tr.groupby("day")["ret"].sum().values
    return s


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    lafo = lafo_daily()
    brk = breakout_daily()
    df = pd.concat([lafo.rename("lafo"), brk.rename("breakout")], axis=1).fillna(0.0)
    # align to common days
    df = df.loc[df.index.isin(lafo.index) & df.index.isin(brk.index)]

    a, b = df["lafo"].to_numpy(), df["breakout"].to_numpy()
    corr = float(np.corrcoef(a, b)[0, 1])

    # per-year correlation to expose the regime complementarity
    df["year"] = pd.DatetimeIndex(df.index).year
    print("=" * 80)
    print("LAFO (long deep-dip MR) vs ORB BREAKOUT (trend proxy) — NDX RTH, net daily")
    print("=" * 80)
    print(f"full-sample daily-return correlation = {corr:.3f}\n")

    rows = []
    for y, g in df.groupby("year"):
        x, y2 = g["lafo"].to_numpy(), g["breakout"].to_numpy()
        c = float(np.corrcoef(x, y2)[0, 1]) if x.std() > 0 and y2.std() > 0 else np.nan
        rows.append({"year": y, "corr": round(c, 2),
                     "lafo_ret%": round(100 * x.sum(), 1), "brk_ret%": round(100 * y2.sum(), 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    # standalone vs 50/50 blend
    blend = 0.5 * a + 0.5 * b
    print("\n" + "=" * 80)
    print("Standalone vs 50/50 blend (net, daily-aggregated)")
    print("=" * 80)
    for name, r in (("LAFO only", a), ("Breakout only", b), ("50/50 blend", blend)):
        print(f"  {name:14s}  Sharpe={_sharpe(r):.2f}  ann%={100*r.mean()*TD:5.2f}  "
              f"maxDD%={100*_maxdd(r):5.1f}")
    df.drop(columns="year").to_csv(f"{OUT}/diversification_daily.csv")
