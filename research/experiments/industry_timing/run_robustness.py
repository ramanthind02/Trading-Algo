"""Robustness battery for the Industry-Timing method on OUR instruments.

Tests the four things that could overturn the Phase-0 negative finding
("timing subtracts Sharpe vs always-long vol-target on our book"):

  A. BREADTH      - is the weakness just too few markets? widen the universe.
  B. CRISIS       - the paper's real pitch is downside protection, not Sharpe.
  C. SUBPERIOD    - was the (weak) edge ever there / has it decayed?
  D. PARAM GRID   - is there ANY (entry,exit,k) where timing beats always-long?
  E. DIVERSIFY    - correlation of per-instrument timed streams (breadth ceiling).

Run: .\.venv\Scripts\python.exe research\experiments\industry_timing\run_robustness.py
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Config, backtest, benchmark_buyhold, metrics

OUT = Path(__file__).resolve().parent / "outputs"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

EQUITY4 = ["ES", "NQ", "YM", "RTY"]
CORE8 = ["ES", "NQ", "YM", "RTY", "GC", "SI", "CL", "TLT"]
# Maximum cross-asset breadth we have daily futures data for (diagnostic only —
# not all of these are in the live vault). Equity / rates / metals / energy.
WIDE = ["ES", "NQ", "YM", "RTY", "TLT", "US", "TY", "FV", "TU",
        "GC", "SI", "PL", "HG", "CL", "HO"]


def _ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def section(title: str) -> None:
    print("\n" + "=" * 92 + f"\n{title}\n" + "=" * 92)


def breadth(cfg: Config) -> None:
    section("A. BREADTH — does the method need many markets? (timed vs always-long VT)")
    print(f"  {'universe':12s} {'N':>3s} {'timed Sh':>9s} {'buyhold Sh':>11s} {'timed-bh':>9s} "
          f"{'timed MDD':>10s} {'bh MDD':>8s}")
    for name, uni in [("equity4", EQUITY4), ("core8", CORE8), ("wide15", WIDE)]:
        t = backtest(uni, cfg)
        b = benchmark_buyhold(uni, cfg)
        mt, mb = t["metrics_gross"], b["metrics"]
        print(f"  {name:12s} {len(uni):3d} {mt['sharpe']:9.2f} {mb['sharpe']:11.2f} "
              f"{mt['sharpe']-mb['sharpe']:9.2f} {mt['mdd']*100:9.1f}% {mb['mdd']*100:7.1f}%")


def crisis(cfg: Config) -> None:
    section("B. CRISIS PARTICIPATION — calendar-year returns, timed vs always-long (core8)")
    t = backtest(CORE8, cfg)["gross_ret"]
    b = benchmark_buyhold(CORE8, cfg)["ret"]
    df = pd.DataFrame({"timed": t, "buyhold": b}).dropna()
    yr = df.groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    yr["timed-bh"] = yr["timed"] - yr["buyhold"]
    # highlight crisis years
    print(yr.round(1).to_string())
    crises = [2008, 2018, 2020, 2022]
    sub = yr.loc[yr.index.isin(crises)]
    print(f"\n  crisis-year avg:  timed {sub['timed'].mean():+.1f}%   "
          f"buyhold {sub['buyhold'].mean():+.1f}%   edge {sub['timed-bh'].mean():+.1f}%")


def subperiod(cfg: Config) -> None:
    section("C. SUBPERIOD — Sharpe by era (core8, timed vs always-long)")
    t = backtest(CORE8, cfg)["gross_ret"]
    b = benchmark_buyhold(CORE8, cfg)["ret"]
    eras = [("1997-2007", "1997-01-01", "2007-12-31"),
            ("2008-2014", "2008-01-01", "2014-12-31"),
            ("2015-2026", "2015-01-01", "2026-12-31")]
    print(f"  {'era':12s} {'timed Sh':>9s} {'buyhold Sh':>11s}")
    for name, s, e in eras:
        ts = _ann_sharpe(t.loc[s:e]); bs = _ann_sharpe(b.loc[s:e])
        print(f"  {name:12s} {ts:9.2f} {bs:11.2f}")
    # rolling 3y sharpe to CSV
    roll = pd.DataFrame({
        "timed": t.rolling(756).apply(lambda x: x.mean() / x.std() * np.sqrt(252), raw=True),
        "buyhold": b.rolling(756).apply(lambda x: x.mean() / x.std() * np.sqrt(252), raw=True),
    })
    roll.to_csv(OUT / "rolling3y_sharpe.csv")
    print(f"  (rolling 3y Sharpe -> {OUT/'rolling3y_sharpe.csv'})")


def param_grid(cfg: Config) -> None:
    section("D. PARAMETER GRID — any (entry,exit,k) where timed beats always-long? (core8)")
    bh = benchmark_buyhold(CORE8, cfg)["metrics"]["sharpe"]
    print(f"  always-long VT Sharpe to beat = {bh:.2f}\n")
    print(f"  {'entry':>6s} {'exit':>5s} {'k':>4s} {'timed Sh':>9s} {'vs bh':>7s} {'MDD':>7s}")
    best = (-9, None)
    for entry in (10, 20, 40):
        for ex in (20, 40, 80):
            for k in (1.5, 2.0, 3.0):
                c = replace(cfg, entry_don=entry, entry_ema=entry, entry_atr=entry,
                            exit_don=ex, exit_ema=ex, exit_atr=ex, entry_k=k, exit_k=k)
                m = backtest(CORE8, c)["metrics_gross"]
                flag = "  <-- beats" if m["sharpe"] > bh else ""
                print(f"  {entry:6d} {ex:5d} {k:4.1f} {m['sharpe']:9.2f} "
                      f"{m['sharpe']-bh:7.2f} {m['mdd']*100:6.1f}%{flag}")
                if m["sharpe"] > best[0]:
                    best = (m["sharpe"], (entry, ex, k))
    print(f"\n  best timed config = {best[1]} at Sharpe {best[0]:.2f} "
          f"({'BEATS' if best[0] > bh else 'still below'} always-long {bh:.2f})")


def diversification(cfg: Config) -> None:
    section("E. DIVERSIFICATION — corr of per-instrument timed return streams (core8)")
    streams = {}
    for t in CORE8:
        streams[t] = backtest([t], replace(cfg, n_assets=8))["gross_ret"]
    df = pd.DataFrame(streams).dropna(how="all")
    corr = df.corr()
    print(corr.round(2).to_string())
    # avg pairwise corr (the diversification ceiling)
    n = len(CORE8)
    off = corr.values[np.triu_indices(n, 1)]
    print(f"\n  avg pairwise corr = {np.nanmean(off):.2f}  "
          f"(paper's 48 industries are far more numerous -> far higher FDM/diversification)")


def wide15_deepdive(cfg: Config) -> None:
    section("F. WIDE15 DEEP-DIVE — is the timing-beats-always-long flip robust?")
    # FAIR cost sensitivity: cost BOTH legs (always-long also rebalances daily ~23x/yr)
    print("  FAIR cost sensitivity (BOTH legs net of one-way bps on |Δw|):")
    print(f"    {'bps':>4s} {'timed Sh':>9s} {'alwl Sh':>8s} {'timed-alwl':>11s}")
    for bps in (0, 1, 2, 5, 10):
        mt = backtest(WIDE, replace(cfg, cost_bps=bps))["metrics_net"]
        mb = benchmark_buyhold(WIDE, replace(cfg, cost_bps=bps))["metrics_net"]
        print(f"    {bps:4d} {mt['sharpe']:9.2f} {mb['sharpe']:8.2f} {mt['sharpe']-mb['sharpe']:11.2f}")
    t_turn = backtest(WIDE, cfg)["metrics_gross"]["ann_turnover"]
    b_turn = benchmark_buyhold(WIDE, cfg)["metrics"]["ann_turnover"]
    print(f"  turnover: timed {t_turn:.1f}x/yr  vs  always-long {b_turn:.1f}x/yr "
          f"(=> costs hit BOTH; not a differentiator)")

    # subperiod timed vs alwl + EXPLICIT post-1997 restriction (the era-decay kill)
    print("\n  subperiod (timed vs always-long, wide15) — the load-bearing era argument:")
    t = backtest(WIDE, cfg)["gross_ret"]
    b = benchmark_buyhold(WIDE, cfg)["ret"]
    for name, s, e in [("pre-1997*", "1900-01-01", "1996-12-31"),
                       ("1997-2007", "1997-01-01", "2007-12-31"),
                       ("2008-2014", "2008-01-01", "2014-12-31"),
                       ("2015-2026", "2015-01-01", "2026-12-31"),
                       ("1997+ (all)", "1997-01-01", "2026-12-31")]:
        print(f"    {name:12s} timed {_ann_sharpe(t.loc[s:e]):.2f}   alwl {_ann_sharpe(b.loc[s:e]):.2f}")
    print("    *pre-1997 = a NO-EQUITY bonds+commodities book (all equities start >=1997, TLT 2002)")

    # post-1997 both-net at 2/5bp (does the edge survive in the era we can actually trade?)
    print("\n  post-1997 only, BOTH net of cost:")
    for bps in (0, 2, 5):
        mt = backtest(WIDE, replace(cfg, cost_bps=bps, start="1997-01-01"))["metrics_net"]
        mb = benchmark_buyhold(WIDE, replace(cfg, cost_bps=bps, start="1997-01-01"))["metrics_net"]
        print(f"    {bps}bps : timed {mt['sharpe']:.2f}  alwl {mb['sharpe']:.2f}")


def main() -> None:
    cfg = Config()
    breadth(cfg)
    crisis(cfg)
    subperiod(cfg)
    param_grid(cfg)
    diversification(cfg)
    wide15_deepdive(cfg)


if __name__ == "__main__":
    main()
