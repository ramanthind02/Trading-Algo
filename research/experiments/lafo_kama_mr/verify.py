"""The deciding red-team checks on the LAFO survivor (M15 2.5% long revert, NDX RTH):

  1. Entry-timing + overnight-gap diagnostics  — is it really an intraday MR or a
     gap-down fade? (red-team: ~72% of entries in the first 90 min ET).
  2. Gap-fade decomposition — does a raw "buy big overnight gap-down, exit EOD, NO KAMA"
     control replicate it? If so the KAMA fair-value machinery is cosmetic.
  3. Honest bootstrap CI — stationary block bootstrap of the daily series + drop-top-N
     concentration, to replace the flattering 1.01 point estimate.
  4. Open-auction cost stress — re-cost entries in the first 30 min with a punitive spread.
  5. Leave-one-year-out jackknife — how much rides on 2020/2022/2025.

Run:  .\.venv\Scripts\python.exe -m research.experiments.lafo_kama_mr.verify
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.lafo_kama_mr import engine as E

Y0, Y1 = 2018, 2026
OUT = "research/experiments/lafo_kama_mr/outputs"
TD = E.TRADING_DAYS
RTH_OPEN_S = E.RTH_OPEN_S

WIN = E.Params(tf="M15", session="rth", entry_mode="rel", threshold=0.025,
               exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False,
               use_recorded_spread=True, slip_pts=1.0)


def _sharpe(a): a = np.asarray(a); return float(a.mean() / a.std() * np.sqrt(TD)) if a.std() > 0 else 0.0
def _maxdd(a): eq = np.cumsum(a); return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0


def _daily(tr, sd, col="ret"):
    g = pd.Series(0.0, index=pd.Index(sd, name="day"))
    if not tr.empty:
        s = tr.groupby("day")[col].sum()
        g.loc[s.index] = s.values
    return g


def day_table(bars):
    """Per-day: first RTH open, last close, prev close, overnight gap."""
    t = bars.groupby("day").agg(o=("open", "first"), c=("close", "last"))
    t["prev_c"] = t["c"].shift(1)
    t["gap"] = t["o"] / t["prev_c"] - 1.0
    t["intraday"] = t["c"] / t["o"] - 1.0
    return t


# --------------------------------------------------------------------------- #
def diagnostics(bars, tr):
    sd = np.sort(bars["day"].unique())
    dt = day_table(bars)
    # entry time of day (ET) distribution
    et_min = (tr["entry_sec"] - RTH_OPEN_S) / 60.0  # minutes after 09:30 ET
    first30 = float((et_min < 30).mean())
    first90 = float((et_min < 90).mean())
    # overnight gap on trade days vs all days
    trade_days = set(pd.to_datetime(tr["day"]))
    gap_all = dt["gap"]
    gap_trade = dt.loc[dt.index.isin(trade_days), "gap"]
    gap_notrade = dt.loc[~dt.index.isin(trade_days), "gap"]
    print("--- 1. ENTRY TIMING & OVERNIGHT GAP ---")
    print(f"  entries in first 30 min ET: {first30:.0%} | first 90 min: {first90:.0%}")
    print(f"  mean overnight gap — trade days: {gap_trade.mean()*100:+.2f}%  "
          f"non-trade days: {gap_notrade.mean()*100:+.2f}%  all: {gap_all.mean()*100:+.2f}%")
    print(f"  median entry time = {np.median(et_min):.0f} min after open ET\n")


def gapfade_control(bars):
    """Buy first RTH bar on big overnight gap-down, exit EOD (NO KAMA). Sweep gap thr."""
    dt = day_table(bars).dropna(subset=["gap"])
    sd = dt.index.to_numpy()
    print("--- 2. GAP-FADE CONTROL (no KAMA): buy gap-down open, exit EOD ---")
    rows = []
    for g in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03):
        mask = dt["gap"] < -g
        r = np.where(mask, dt["intraday"], 0.0)
        rows.append({"gap_thr%": g * 100, "sharpe": round(_sharpe(r), 2),
                     "ann%": round(100 * r.mean() * TD, 2), "maxDD%": round(100 * _maxdd(r), 1),
                     "days_traded": int(mask.sum()), "win%": round(100 * (dt.loc[mask, "intraday"] > 0).mean(), 1)})
    gf = pd.DataFrame(rows)
    print(gf.to_string(index=False))
    # correlation of best gap-fade to LAFO daily
    _, tr = E.run("NDX", Y0, Y1, WIN)
    lafo = _daily(tr, np.sort(bars["day"].unique()))
    best_g = 0.02
    gfr = pd.Series(np.where(dt["gap"] < -best_g, dt["intraday"], 0.0), index=dt.index)
    common = lafo.index.intersection(gfr.index)
    corr = float(np.corrcoef(lafo.loc[common], gfr.loc[common])[0, 1])
    print(f"  corr(LAFO daily, gap-fade@2% daily) = {corr:.2f}   "
          f"(LAFO net Sharpe {_sharpe(lafo.to_numpy()):.2f} vs gap-fade@2% {gf.loc[gf['gap_thr%']==2.0,'sharpe'].iloc[0]})\n")
    return gf


def bootstrap_ci(bars, tr, n=3000, block=20, seed=1):
    """Stationary block bootstrap of the net daily series + drop-top-N concentration."""
    sd = np.sort(bars["day"].unique())
    daily = _daily(tr, sd).to_numpy()
    nday = len(daily)
    rng = np.random.default_rng(seed)
    sh = np.empty(n)
    for k in range(n):
        idx = []
        while len(idx) < nday:
            start = rng.integers(0, nday)
            L = rng.geometric(1.0 / block)
            idx.extend(range(start, min(start + L, nday)))
        s = daily[np.array(idx[:nday])]
        sh[k] = _sharpe(s)
    print("--- 3. HONEST BOOTSTRAP CI (stationary block, net daily) ---")
    print(f"  point Sharpe = {_sharpe(daily):.2f}")
    print(f"  bootstrap mean = {sh.mean():.2f} | 5th = {np.percentile(sh,5):.2f} | "
          f"50th = {np.percentile(sh,50):.2f} | 95th = {np.percentile(sh,95):.2f} | "
          f"P(SR<0.5) = {(sh<0.5).mean():.0%}")
    # drop-top-N by net_pts concentration
    pts = tr.sort_values("net_pts", ascending=False)
    tot = tr["net_pts"].sum()
    for nN in (5, 10, 20):
        keep = pts.iloc[nN:]
        d2 = _daily(keep, sd).to_numpy()
        share = 100 * pts.iloc[:nN]["net_pts"].sum() / tot
        print(f"  drop top {nN:2d} trades (={share:.0f}% of net pts): Sharpe -> {_sharpe(d2):.2f}")
    print()


def open_auction_stress(bars, tr):
    """Re-cost: entries in the first 30 min pay a punitive round-trip spread."""
    sd = np.sort(bars["day"].unique())
    base = _daily(tr, sd)
    print("--- 4. OPEN-AUCTION COST STRESS (punitive spread on first-30-min entries) ---")
    print(f"  base (recorded spread + 1pt slip): Sharpe {_sharpe(base.to_numpy()):.2f}")
    early = (tr["entry_sec"] - RTH_OPEN_S) < 30 * 60
    print(f"  share of entries in first 30 min: {early.mean():.0%}")
    for extra_pts in (2.0, 4.0, 6.0):
        t2 = tr.copy()
        # add `extra_pts` idx-pt round-trip cost (as return) to early entries
        add = np.where(early, extra_pts / t2["entry"], 0.0)
        t2["ret"] = t2["ret"] - add
        d = _daily(t2, sd)
        print(f"  + {extra_pts:.0f}pt extra RT spread on open entries: Sharpe {_sharpe(d.to_numpy()):.2f} "
              f"(ann% {100*d.mean()*TD:.2f})")
    print()


def leave_one_year_out(bars, tr):
    sd = np.sort(bars["day"].unique())
    daily = _daily(tr, sd)
    years = pd.DatetimeIndex(daily.index).year
    print("--- 5. LEAVE-ONE-YEAR-OUT jackknife ---")
    rows = []
    for y in sorted(set(years)):
        keep = daily[years != y]
        rows.append({"excl_year": y, "sharpe_rest": round(_sharpe(keep.to_numpy()), 2)})
    loo = pd.DataFrame(rows)
    print(loo.to_string(index=False))
    print(f"  LOO Sharpe range: {loo['sharpe_rest'].min():.2f} .. {loo['sharpe_rest'].max():.2f} "
          f"(full-sample {_sharpe(daily.to_numpy()):.2f})\n")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    bars = E.prep_bars("NDX", Y0, Y1, WIN)
    _, tr = E.run("NDX", Y0, Y1, WIN)
    print("=" * 80)
    print(f"VERIFY — LAFO M15 2.5% long revert, NDX RTH net  ({len(tr)} trades)")
    print("=" * 80 + "\n")
    diagnostics(bars, tr)
    gapfade_control(bars)
    bootstrap_ci(bars, tr)
    open_auction_stress(bars, tr)
    leave_one_year_out(bars, tr)
