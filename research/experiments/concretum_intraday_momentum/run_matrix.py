"""
Robustness / stability matrix for the Concretum intraday-momentum strategy.

Usage:  python -m research.experiments.concretum_intraday_momentum.run_matrix SYM [Y0 Y1]

Writes tidy CSVs to outputs/<dim>__<SYM>.csv across these robustness dimensions:
  escalation   the 3-model ladder x {full, paper-era <=2024-04, post-pub >=2024-05}
  yearly       per-calendar-year stability + profit concentration (headline model)
  vm           Volatility-Multiplier plateau (paper sec 4.4)
  lookback     time-of-day sigma lookback plateau
  sizing       vol_target x lev_cap sensitivity
  freq         decision grid: last30 / hourly / semi / quarter-hourly
  stop         opp vs curr vs curr+VWAP ablation (binary + vol-target)
  longshort    long-only / short-only / both (beta vs symmetric microstructure edge)
  cost         round-trip cost waterfall (the intraday gate) + recorded-spread
  dow          day-of-week effect (paper sec 4.3)

The HEADLINE model is curr+VWAP + vol-target (paper Table 3). For every cell we
report BOTH the vol-target Sharpe (paper-comparable) and the binary Sharpe
(leverage=1, the implementation-invariant anchor).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E

OUT = Path("research/experiments/concretum_intraday_momentum/outputs")
OUT.mkdir(parents=True, exist_ok=True)

PAPER_END = pd.Timestamp("2024-04-30")    # paper's data ends April 2024
POST_START = pd.Timestamp("2024-05-01")

DATA_START = {"SPY": 2010, "QQQ": 2010, "ES": 2015, "SP500": 2018, "NDX": 2018}


def _col(p: E.Params) -> str:
    return "lev_ret" if p.sizing == "vol_target" else "unlev_ret"


def _net_col(p: E.Params) -> str:
    return "net_lev" if p.sizing == "vol_target" else "net_unlev"


def m_window(res: pd.DataFrame, col: str, lo=None, hi=None) -> dict:
    sub = res.copy()
    if lo is not None:
        sub = sub[sub.index >= lo]
    if hi is not None:
        sub = sub[sub.index <= hi]
    return E.metrics(sub, col)


# --------------------------------------------------------------------------- #
def dim_escalation(dt, sym, nmk):
    ladder = [
        ("opp_binary", E.Params(stop_mode="opp", sizing="binary")),
        ("curr_binary", E.Params(stop_mode="curr", sizing="binary")),
        ("currvwap_binary", E.Params(stop_mode="curr_vwap", sizing="binary")),
        ("currvwap_voltarget", E.Params(stop_mode="curr_vwap", sizing="vol_target")),
    ]
    rows = []
    for label, p in ladder:
        res = E.simulate(dt, p, nmk)
        col = _col(p)
        for wlabel, lo, hi in [("full", None, None),
                               ("paper<=2024-04", None, PAPER_END),
                               ("post>=2024-05", POST_START, None)]:
            m = m_window(res, col, lo, hi)
            m.update(model=label, window=wlabel, sym=sym)
            rows.append(m)
    return pd.DataFrame(rows)


def dim_yearly(dt, sym, nmk):
    p = E.Params(stop_mode="curr_vwap", sizing="vol_target")
    res = E.simulate(dt, p, nmk)
    pb = E.Params(stop_mode="curr_vwap", sizing="binary")
    resb = E.simulate(dt, pb, nmk)
    rows = []
    for yr, g in res[res.valid].groupby(res[res.valid].index.year):
        gb = resb[resb.valid]
        gb = gb[gb.index.year == yr]
        rows.append({
            "sym": sym, "year": int(yr), "days": len(g),
            "sharpe_vt": E.metrics(g, "lev_ret")["sharpe"],
            "ann%_vt": E.metrics(g, "lev_ret")["ann_ret%"],
            "tot%_vt": E.metrics(g, "lev_ret")["tot_ret%"],
            "maxDD%_vt": E.metrics(g, "lev_ret")["maxDD%"],
            "sharpe_bin": E.metrics(gb, "unlev_ret")["sharpe"] if len(gb) else 0.0,
            "tot%_bin": E.metrics(gb, "unlev_ret")["tot_ret%"] if len(gb) else 0.0,
        })
    return pd.DataFrame(rows)


def dim_vm(dt, sym, nmk):
    rows = []
    for vm in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        for sizing in ["binary", "vol_target"]:
            p = E.Params(vm=vm, stop_mode="curr_vwap", sizing=sizing)
            res = E.simulate(dt, p, nmk)
            m = E.metrics(res, _col(p))
            m.update(sym=sym, vm=vm, sizing=sizing)
            rows.append(m)
    return pd.DataFrame(rows)


def dim_lookback(dt_by_lb, sym, nmk):
    # lookback only changes sigma rolling window -> same dt; just vary param
    dt = dt_by_lb
    rows = []
    for lb in [7, 10, 14, 21, 28]:
        for sizing in ["binary", "vol_target"]:
            p = E.Params(vol_lookback_tod=lb, stop_mode="curr_vwap", sizing=sizing)
            res = E.simulate(dt, p, nmk)
            m = E.metrics(res, _col(p))
            m.update(sym=sym, lookback_tod=lb, sizing=sizing)
            rows.append(m)
    return pd.DataFrame(rows)


def dim_sizing(dt, sym, nmk):
    rows = []
    for vt in [0.01, 0.015, 0.02, 0.03]:
        for cap in [2.0, 4.0, 6.0]:
            p = E.Params(stop_mode="curr_vwap", sizing="vol_target", vol_target=vt, lev_cap=cap)
            res = E.simulate(dt, p, nmk)
            m = E.metrics(res, "lev_ret")
            m.update(sym=sym, vol_target=vt, lev_cap=cap)
            rows.append(m)
    return pd.DataFrame(rows)


def dim_freq(dt_full_loader, sym):
    rows = []
    grids = {"last30": E.MARKS_ET_LAST30, "hourly": E.MARKS_ET_HOUR,
             "semi": E.MARKS_ET_SEMI, "quarter": E.MARKS_ET_Q}
    for gl, marks in grids.items():
        dt = dt_full_loader(marks)
        nmk = len(marks)
        for sizing in ["binary", "vol_target"]:
            p = E.Params(stop_mode="curr_vwap", sizing=sizing)
            res = E.simulate(dt, p, nmk)
            m = E.metrics(res, _col(p))
            m.update(sym=sym, grid=gl, n_marks=nmk, sizing=sizing)
            rows.append(m)
    return pd.DataFrame(rows)


def dim_stop(dt, sym, nmk):
    rows = []
    for mode in ["opp", "curr", "curr_vwap"]:
        for sizing in ["binary", "vol_target"]:
            p = E.Params(stop_mode=mode, sizing=sizing)
            res = E.simulate(dt, p, nmk)
            m = E.metrics(res, _col(p))
            m.update(sym=sym, stop=mode, sizing=sizing)
            rows.append(m)
    return pd.DataFrame(rows)


def dim_longshort(dt, sym, nmk):
    rows = []
    for label, lo, sh in [("both", True, True), ("long_only", True, False),
                          ("short_only", False, True)]:
        for sizing in ["binary", "vol_target"]:
            p = E.Params(stop_mode="curr_vwap", sizing=sizing, longs=lo, shorts=sh)
            res = E.simulate(dt, p, nmk)
            m = E.metrics(res, _col(p))
            m.update(sym=sym, side=label, sizing=sizing)
            rows.append(m)
    return pd.DataFrame(rows)


def dim_cost(dt, sym, nmk):
    rows = []
    # one-way bps per unit-notional traded; round trip ~= 2x. Paper SPY ~1.1 bps one-way.
    for bps in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        p = E.Params(stop_mode="curr_vwap", sizing="vol_target", cost_bps_per_turn=bps)
        res = E.simulate(dt, p, nmk)
        m = E.metrics(res, "net_lev")
        m.update(sym=sym, cost=f"{bps}bps_1way", oneway_bps=bps)
        rows.append(m)
    # recorded-spread variant (point-converted): the actual quoted CFD half-spread
    p = E.Params(stop_mode="curr_vwap", sizing="vol_target", use_recorded_spread=True)
    res = E.simulate(dt, p, nmk)
    m = E.metrics(res, "net_lev")
    rec_bps = float(np.nanmedian(0.5 * dt["spread_med"] * p.point_size / dt["open"]) * 1e4)
    m.update(sym=sym, cost="recorded_spread", oneway_bps=round(rec_bps, 3))
    rows.append(m)
    return pd.DataFrame(rows)


def dim_dow(dt, sym, nmk):
    p = E.Params(stop_mode="curr_vwap", sizing="vol_target")
    res = E.simulate(dt, p, nmk)
    sub = res[res.valid].copy()
    sub["dow"] = sub.index.dayofweek
    rows = []
    names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    for d, g in sub.groupby("dow"):
        r = g["lev_ret"].to_numpy()
        r = r[np.isfinite(r)]
        mean = r.mean()
        t = mean / (r.std() / np.sqrt(len(r))) if r.std() > 0 else 0.0
        # this weekday is sampled ~once per week -> annualize by sqrt(52), NOT sqrt(252)
        rows.append({"sym": sym, "dow": names.get(int(d), str(d)), "obs": len(r),
                     "avg_bps": round(1e4 * mean, 1), "t_stat": round(t, 2),
                     "sharpe_wk": round(mean / r.std() * np.sqrt(52), 2) if r.std() > 0 else 0.0,
                     "hit%": round(100 * (r > 0).mean(), 1)})
    return pd.DataFrame(rows)


def dim_regime(dt, sym, nmk):
    """Paper sec 4.1: does risk-adjusted return rise with market vol? Bucket the headline
    vol-target daily returns by PRIOR-DAY realized-vol quartile (causal: sigma_daily is
    as-of t-1). Also split paper-era vs post-pub to test whether the regime effect persists."""
    p = E.Params(stop_mode="curr_vwap", sizing="vol_target")
    res = E.simulate(dt, p, nmk)
    sigma_day = dt["daily_ret"].shift(1).rolling(p.vol_lookback_day).std(ddof=1)
    sub = res[res.valid].copy()
    sub["sig"] = sigma_day.reindex(sub.index)
    sub = sub[np.isfinite(sub["sig"])]
    rows = []
    for wlabel, lo, hi in [("full", None, None), ("paper<=2024-04", None, PAPER_END),
                           ("post>=2024-05", POST_START, None)]:
        w = sub.copy()
        if lo is not None:
            w = w[w.index >= lo]
        if hi is not None:
            w = w[w.index <= hi]
        if len(w) < 40:
            continue
        q = pd.qcut(w["sig"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        for ql, g in w.groupby(q, observed=True):
            r = g["lev_ret"].to_numpy(); r = r[np.isfinite(r)]
            sh = float(r.mean() / r.std() * np.sqrt(E.TRADING_DAYS)) if r.std() > 0 else 0.0
            rows.append({"sym": sym, "window": wlabel, "vol_quartile": ql, "obs": len(r),
                         "avg_bps": round(1e4 * r.mean(), 1), "sharpe": round(sh, 2)})
    return pd.DataFrame(rows)


def main():
    sym = sys.argv[1]
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else DATA_START.get(sym, 2018)
    y1 = int(sys.argv[3]) if len(sys.argv) > 3 else 2026

    win = E.load_sessions(sym, y0, y1)

    def loader(marks):
        return E.build_day_table(win, marks)

    dt = loader(E.MARKS_ET_SEMI)
    nmk = len(E.MARKS_ET_SEMI)
    print(f"{sym} {y0}-{y1}: {dt.index.min().date()}..{dt.index.max().date()}  {len(dt)} days, {len(win):,} bars")

    writers = {
        "escalation": lambda: dim_escalation(dt, sym, nmk),
        "yearly": lambda: dim_yearly(dt, sym, nmk),
        "vm": lambda: dim_vm(dt, sym, nmk),
        "lookback": lambda: dim_lookback(dt, sym, nmk),
        "sizing": lambda: dim_sizing(dt, sym, nmk),
        "stop": lambda: dim_stop(dt, sym, nmk),
        "longshort": lambda: dim_longshort(dt, sym, nmk),
        "cost": lambda: dim_cost(dt, sym, nmk),
        "dow": lambda: dim_dow(dt, sym, nmk),
        "regime": lambda: dim_regime(dt, sym, nmk),
        "freq": lambda: dim_freq(loader, sym),
    }
    for name, fn in writers.items():
        df = fn()
        df.to_csv(OUT / f"{name}__{sym}.csv", index=False)
        print(f"  wrote {name}__{sym}.csv  ({len(df)} rows)")


if __name__ == "__main__":
    main()
