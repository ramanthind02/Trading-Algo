"""Equity curves + diagnostic charts for the LAFO/KAMA deep-dip MR study.

Writes PNGs to outputs/. Matplotlib (repo visualization policy: CSV-first, mpl charts).

Run:  .\.venv\Scripts\python.exe -m research.experiments.lafo_kama_mr.plots
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.experiments.lafo_kama_mr import engine as E
from research.experiments.orb_ibs_nas100 import poc as ORB

Y0, Y1 = 2018, 2026
OUT = "research/experiments/lafo_kama_mr/outputs"
TD = E.TRADING_DAYS


def _daily(p: E.Params, sym="NDX") -> tuple[pd.Series, pd.Series]:
    """Return (net_daily, gross_daily) return series on the full session-day grid."""
    bars = E.prep_bars(sym, Y0, Y1, p)
    _, tr = E.run(sym, Y0, Y1, p)
    sd = np.sort(bars["day"].unique())
    net = pd.Series(0.0, index=pd.Index(sd, name="day"))
    gross = pd.Series(0.0, index=pd.Index(sd, name="day"))
    if not tr.empty:
        net.loc[tr.groupby("day")["ret"].sum().index] = tr.groupby("day")["ret"].sum().values
        gross.loc[tr.groupby("day")["gross_ret"].sum().index] = tr.groupby("day")["gross_ret"].sum().values
    return net, gross


def _sharpe(a): return float(a.mean() / a.std() * np.sqrt(TD)) if a.std() > 0 else 0.0
def _dd(eq): return eq - np.maximum.accumulate(eq)


WIN = E.Params(tf="M15", session="rth", entry_mode="rel", threshold=0.025,
               exit_mode="revert", stop_atr_mult=3.5, longs=True, shorts=False,
               use_recorded_spread=True, slip_pts=1.0)


def fig_headline():
    net, gross = _daily(WIN)
    x = pd.to_datetime(net.index)
    eq_n, eq_g = 100 * net.cumsum(), 100 * gross.cumsum()
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    ax.plot(x, eq_g, lw=1.3, color="#9ecae1", label=f"gross (Sharpe {_sharpe(gross.to_numpy()):.2f})")
    ax.plot(x, eq_n, lw=1.6, color="#08519c",
            label=f"net @recorded spread+1pt slip (Sharpe {_sharpe(net.to_numpy()):.2f})")
    ax.set_title("LAFO / KAMA deep-dip MR — NDX M15 RTH, long-only, thr 2.5%, revert exit\n"
                 "cumulative return (sum of daily), 2018-2026")
    ax.set_ylabel("cumulative return (%)"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    axd.fill_between(x, _dd(eq_n.to_numpy()), 0, color="#de2d26", alpha=0.5)  # eq_n already %
    axd.set_ylabel("drawdown (%)"); axd.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_headline_equity.png", dpi=120); plt.close(fig)
    print("wrote fig_headline_equity.png")


def fig_diversification():
    net, _ = _daily(WIN)
    win = ORB.load_sessions(2018)
    tr = ORB.simulate(win, ORB.Params(use_ibs=False, rr=3.0, use_recorded_spread=True))
    sd = np.sort(win["day"].unique())
    brk = pd.Series(0.0, index=pd.Index(sd, name="day"))
    brk.loc[tr.groupby("day")["ret"].sum().index] = tr.groupby("day")["ret"].sum().values
    df = pd.concat([net.rename("lafo"), brk.rename("brk")], axis=1).dropna()
    blend = 0.5 * df["lafo"] + 0.5 * df["brk"]
    x = pd.to_datetime(df.index)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, 100 * df["lafo"].cumsum(), lw=1.6, color="#08519c",
            label=f"LAFO deep-dip MR (Sharpe {_sharpe(df['lafo'].to_numpy()):.2f}, maxDD {abs(_dd(df['lafo'].cumsum().to_numpy()).min())*100:.0f}%)")
    ax.plot(x, 100 * df["brk"].cumsum(), lw=1.3, color="#d95f0e",
            label=f"ORB breakout proxy (Sharpe {_sharpe(df['brk'].to_numpy()):.2f}, maxDD {abs(_dd(df['brk'].cumsum().to_numpy()).min())*100:.0f}%)")
    ax.plot(x, 100 * blend.cumsum(), lw=1.8, color="#238b45", ls="--",
            label=f"50/50 blend (Sharpe {_sharpe(blend.to_numpy()):.2f})")
    corr = float(np.corrcoef(df["lafo"], df["brk"])[0, 1])
    ax.set_title(f"Diversification: LAFO (MR) vs trend/breakout proxy — NDX RTH net\n"
                 f"daily-return correlation = {corr:+.2f}  (MR carries 2018 & 2020 where breakout bleeds)")
    ax.set_ylabel("cumulative return (%)"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_diversification_equity.png", dpi=120); plt.close(fig)
    print("wrote fig_diversification_equity.png")


def fig_threshold_plateau():
    g = pd.read_csv(f"{OUT}/rob_sp500_generalization.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {("NDX", "M15"): ("#08519c", "-o"), ("NDX", "M5"): ("#3182bd", "--o"),
              ("SP500", "M15"): ("#d95f0e", "-s"), ("SP500", "M5"): ("#fe9929", "--s")}
    for (sym, tf), grp in g.groupby(["sym", "tf"]):
        c, st = styles[(sym, tf)]
        ax.plot(100 * grp["thr"], grp["sharpe"], st, color=c, label=f"{sym} {tf}")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("entry threshold — dip below KAMA (%)"); ax.set_ylabel("frictionless Sharpe")
    ax.set_title("Long-only deep-dip plateau: edge rises with dislocation depth\n"
                 "NDX peaks ~2.5%, SP500 (lower vol) peaks ~1.5% — same mechanism, scaled by vol")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_threshold_plateau.png", dpi=120); plt.close(fig)
    print("wrote fig_threshold_plateau.png")


def fig_cost_decay():
    s = pd.read_csv(f"{OUT}/cost_slippage_sweep.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, grp in s.groupby("config"):
        ax.plot(grp["slip_pts"], grp["net_sharpe"], "-o", label=name)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("slippage (idx pts per side, on top of recorded ~0.9pt spread)")
    ax.set_ylabel("net Sharpe")
    ax.set_title("Cost robustness: deep-dip configs barely decay (large edge / low turnover);\n"
                 "vol-normalised z-entry is the cost-fragile one")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_cost_decay.png", dpi=120); plt.close(fig)
    print("wrote fig_cost_decay.png")


def fig_annual():
    net, _ = _daily(WIN)
    yr = net.groupby(pd.to_datetime(net.index).year).sum() * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#238b45" if v >= 0 else "#de2d26" for v in yr.values]
    ax.bar(yr.index.astype(str), yr.values, color=colors)
    for i, v in enumerate(yr.values):
        ax.text(i, v + (0.3 if v >= 0 else -0.6), f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("net return (%)")
    ax.set_title("LAFO annual net return — active/profitable in chop (2020,2022), dormant in trends (2023,2026)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_annual_returns.png", dpi=120); plt.close(fig)
    print("wrote fig_annual_returns.png")


if __name__ == "__main__":
    fig_headline()
    fig_diversification()
    fig_threshold_plateau()
    fig_cost_decay()
    fig_annual()
    print("all figures written to", OUT)
