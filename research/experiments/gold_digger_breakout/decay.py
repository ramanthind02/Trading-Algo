"""
Is the USDJPY model dead? Empirical decay diagnostics.

Five tests:
  1. Rolling 126-day annualized Sharpe of the daily R series (decay onset, envelope).
  2. Long vs short by year  -> WHICH side broke in 2026 (directional tilt vs mechanism).
  3. Whipsaw/false-breakout by year (SL-rate, avg winner R, range width regime).
  4. Block-bootstrap: is the 2026-YTD result within the 2018-2025 distribution of
     same-length windows, or unprecedented? (regime break vs deep-but-seen drawdown)
  5. Cross-market: same model on all JPY crosses + USD majors + metals -> localize the
     2026 break (yen-specific regime vs broad intraday-breakout decay vs idiosyncratic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.gold_digger_breakout.poc import (
    load_m1, prep_days, build_trades,
)

GAP = True


def daily_R(tr: pd.DataFrame, all_days: np.ndarray) -> pd.Series:
    s = pd.Series(0.0, index=pd.Index(np.unique(all_days)))
    g = tr.groupby("day")["R"].sum()
    s.loc[g.index] = g.values
    return s


def long_short_by_year(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr.copy()
    t["year"] = pd.to_datetime(t["day"]).dt.year
    rows = []
    for y, g in t.groupby("year"):
        row = {"year": y, "n": len(g)}
        for side, nm in [(1, "L"), (-1, "S")]:
            gs = g[g.side == side]
            R = gs["R"].values
            row[f"{nm}_n"] = len(R)
            row[f"{nm}_totR"] = round(R.sum(), 1)
            row[f"{nm}_avgR"] = round(R.mean(), 3) if len(R) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def whipsaw_by_year(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr.copy()
    t["year"] = pd.to_datetime(t["day"]).dt.year
    rows = []
    for y, g in t.groupby("year"):
        R = g["R"].values
        wins = g[g.R > 0]["R"].values
        rows.append(dict(
            year=y, n=len(g),
            sl_rate=round((g.reason == "SL").mean(), 3),
            time_rate=round((g.reason == "time").mean(), 3),
            avg_win_R=round(wins.mean(), 3) if len(wins) else np.nan,
            med_width=round(g["width"].median(), 4),
            avg_R=round(R.mean(), 3),
        ))
    return pd.DataFrame(rows)


def bootstrap_window(tr: pd.DataFrame, train_end="2025-12-31",
                     block=5, n_boot=20000, seed=12345) -> dict:
    """Is the 2026-YTD total R within the 2018-2025 distribution of same-length
    (same trade count) windows? Block bootstrap on the per-trade R series."""
    t = tr.sort_values("day")
    is_2026 = pd.to_datetime(t["day"]).dt.year == 2026
    R_train = t.loc[~is_2026, "R"].values
    actual_2026 = t.loc[is_2026, "R"].sum()
    k = int(is_2026.sum())  # number of 2026 trades
    rng = np.random.default_rng(seed)
    nb = len(R_train)
    sims = np.empty(n_boot)
    for b in range(n_boot):
        acc, filled = [], 0
        while filled < k:
            start = rng.integers(0, nb - block)
            chunk = R_train[start:start + block]
            acc.append(chunk)
            filled += block
        sims[b] = np.concatenate(acc)[:k].sum()
    pct = (sims < actual_2026).mean()
    # also: worst historical rolling k-trade window (no resampling)
    csum = np.concatenate([[0], np.cumsum(t["R"].values)])
    roll = csum[k:] - csum[:-k]
    return dict(k_trades=k, actual_2026_totR=round(actual_2026, 1),
                boot_mean=round(sims.mean(), 1), boot_p05=round(np.percentile(sims, 5), 1),
                boot_p01=round(np.percentile(sims, 1), 1),
                pct_below=round(pct, 4),
                worst_hist_kwin=round(roll.min(), 1))


def jpy_deep():
    print(f"\n{'='*70}\nUSDJPY decay deep-dive\n{'='*70}")
    df = load_m1("USDJPY")
    days = prep_days(df)
    all_days = df["day"].values
    tr = build_trades(days, 3, 6, 18, gap_fill=GAP)

    print("\n--- long vs short by year (which side broke?) ---")
    print(long_short_by_year(tr).to_string(index=False))
    print("\n--- whipsaw / regime by year ---")
    print(whipsaw_by_year(tr).to_string(index=False))
    print("\n--- bootstrap: is 2026 YTD within the 2018-2025 envelope? ---")
    bs = bootstrap_window(tr)
    for k, v in bs.items():
        print(f"   {k:18s}: {v}")
    # rolling sharpe plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dR = daily_R(tr, all_days)
    roll = dR.rolling(126).mean() / dR.rolling(126).std() * np.sqrt(252)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(dR.index, np.cumsum(dR.values), lw=1.2)
    ax[0].axhline(0, color="k", lw=.5); ax[0].set_title("USDJPY cum R (gross)"); ax[0].grid(alpha=.3)
    ax[1].plot(roll.index, roll.values, lw=1.1, color="tab:red")
    ax[1].axhline(0, color="k", lw=.5); ax[1].axhline(1, color="g", lw=.5, ls=":")
    ax[1].set_title("rolling 126d annualized Sharpe"); ax[1].grid(alpha=.3)
    for a in ax:
        a.axvspan(pd.Timestamp("2026-01-01"), dR.index.max(), color="orange", alpha=.15)
    plt.tight_layout()
    plt.savefig("research/experiments/gold_digger_breakout/outputs/usdjpy_decay.png", dpi=110)
    print("   wrote outputs/usdjpy_decay.png")


def cross_market():
    print(f"\n{'='*70}\nCross-market: same model, full-history vs 2026 YTD\n{'='*70}")
    jpy = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]
    usd = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD"]
    ctrl = ["XAUUSD", "XAGUSD"]
    rows = []
    for grp, syms in [("JPY-cross", jpy), ("USD-major", usd), ("metal", ctrl)]:
        for sym in syms:
            try:
                df = load_m1(sym)
                days = prep_days(df)
                tr = build_trades(days, 3, 6, 18, gap_fill=GAP)
            except Exception as e:
                rows.append(dict(grp=grp, sym=sym, note=f"skip:{e}"))
                continue
            yr = pd.to_datetime(tr["day"]).dt.year
            full = tr["R"].values
            t26 = tr.loc[yr == 2026, "R"].values
            t25 = tr.loc[yr == 2025, "R"].values
            rows.append(dict(
                grp=grp, sym=sym, n=len(full),
                full_avgR=round(full.mean(), 3),
                full_totR=round(full.sum(), 0),
                y2025_totR=round(t25.sum(), 1),
                y2026_n=len(t26),
                y2026_totR=round(t26.sum(), 1),
                y2026_avgR=round(t26.mean(), 3) if len(t26) else np.nan,
                y2026_win=round((t26 > 0).mean(), 2) if len(t26) else np.nan,
            ))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    jpy_deep()
    cross_market()
