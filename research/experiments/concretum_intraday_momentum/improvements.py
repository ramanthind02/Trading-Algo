"""
Three frictionless improvements to the Part-1 Noise-Area intraday momentum model:
  #1 fast-alpha ENTRY overlay  — delay each breakout entry until a 5-min pullback bar
                                 (the proven QuanTips #2 fill rule, applied to THIS model).
  #2 lower frequency + cooldown — hourly vs semi-hourly marks; suppress re-entry for N marks
                                 after a stop (cuts the curr_vwap whipsaw turnover).
  #3 equity-curve kill-switch   — gate the daily stream by its own equity vs SMA
                                 (execution.equity_curve_filter, lookahead-free).

To let the 5-min overlay interpose, the Noise-Area model is rebuilt natively on the 5-min grid
(same logic as engine.py but mark prices/bands/VWAP measured on 5-min bars). The no-overlay,
semi-hourly, no-cooldown run is the apples-to-apples baseline on this grid.

Lookahead discipline carried over: time-of-day sigma uses .shift(1).rolling (strictly past);
position chosen at a 5-min bar earns the forward 5-min return; leverage uses daily returns to t-1;
the equity gate uses equity through t-1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E
from research.experiments.concretum_intraday_momentum import fast_alpha as FA
from execution.equity_curve_filter import apply_equity_curve_filter

TRADING_DAYS = 252


@dataclass(frozen=True)
class IParams:
    vm: float = 1.0
    lookback: int = 14
    stop_mode: str = "curr_vwap"
    marks_et: tuple = tuple(E.MARKS_ET_SEMI)
    overlay_entry: bool = False
    overlay_exit: bool = False
    cooldown: int = 0                  # marks to block re-entry after a stop-to-flat
    cost_bps_oneway: float = 1.0
    sizing: str = "vol_target"
    eq_filter_window: int = 0          # 0 = off; else SMA window for the equity-curve gate
    long_only: bool = False            # suppress short positions
    ma_filter: int = 0                 # 0 = off; >0 = only trade long when close > SMA(ma_filter)


def _mark_table(b5: pd.DataFrame, marks_et) -> tuple[pd.DataFrame, dict]:
    """Per-day open/prev_close + per-mark (px, vwap, move, 5-min bar index)."""
    mark_s = E.marks_broker_sec(marks_et)
    nmk = len(mark_s)
    rows = []
    bar_idx = {}                       # day -> array of 5-min bar idx for each mark
    for day, g in b5.groupby("day", sort=True):
        sec = g["sec"].to_numpy()
        c = g["close"].to_numpy()
        h = g["high"].to_numpy(); lo = g["low"].to_numpy(); vol = g["vol"].to_numpy()
        if len(g) < 6:
            continue
        open_px = float(g["open"].to_numpy()[0])
        typ = (h + lo + c) / 3.0
        cum_pv = np.cumsum(typ * vol); cum_v = np.cumsum(vol)
        rec = {"day": day, "open": open_px, "close": float(c[-1])}
        idxs = np.full(nmk, -1, dtype=int)
        for k, ms in enumerate(mark_s):
            j = np.searchsorted(sec, ms, side="right") - 1
            idxs[k] = j
            if j < 0:
                rec[f"px_{k}"] = np.nan; rec[f"vwap_{k}"] = np.nan; rec[f"move_{k}"] = np.nan
            else:
                px = float(c[j]); rec[f"px_{k}"] = px
                rec[f"vwap_{k}"] = float(cum_pv[j] / cum_v[j]) if cum_v[j] > 0 else px
                rec[f"move_{k}"] = abs(px / open_px - 1.0)
        rows.append(rec); bar_idx[day] = idxs
    dt = pd.DataFrame(rows).set_index("day").sort_index()
    dt["prev_close"] = dt["close"].shift(1)
    return dt, bar_idx


def _mark_positions(px, ub, lb, vw, stop_mode, cooldown):
    """Noise-Area state machine across marks with a post-stop re-entry cooldown."""
    nmk = len(px)
    out = np.zeros(nmk, dtype=np.int8)
    pos = 0; cd = 0
    for k in range(nmk):
        prev = pos
        npos = E._step(pos, px[k], ub[k], lb[k], vw[k], stop_mode)
        # cooldown: block a fresh entry-from-flat while cooling down
        if cd > 0 and prev == 0 and npos != 0:
            npos = 0
        pos = npos
        out[k] = pos
        # arm cooldown on a stop-to-flat (exit that is not a reversal)
        if prev != 0 and pos == 0 and cooldown > 0:
            cd = cooldown
        elif cd > 0:
            cd -= 1
    return out


def simulate(sym: str, y0: int, y1: int, p: IParams) -> pd.DataFrame:
    b5, d5 = FA.load_5m(sym, y0, y1)
    marks_et = list(p.marks_et)
    dt, bar_idx = _mark_table(b5, marks_et)
    # MA regime gate: 1 if prev_close > SMA(prev_close, ma_filter), else 0
    if p.ma_filter > 0:
        sma = dt["close"].shift(1).rolling(p.ma_filter).mean()
        ma_gate = (dt["close"].shift(1) > sma).astype(int).to_dict()
    else:
        ma_gate = None
    nmk = len(marks_et)
    sig = np.column_stack([dt[f"move_{k}"].shift(1).rolling(p.lookback).mean().to_numpy()
                           for k in range(nmk)])
    open_px = dt["open"].to_numpy(); prev_close = dt["prev_close"].to_numpy()
    close_px = dt["close"].to_numpy()
    px = np.column_stack([dt[f"px_{k}"].to_numpy() for k in range(nmk)])
    vwap = np.column_stack([dt[f"vwap_{k}"].to_numpy() for k in range(nmk)])
    lev_map = d5["lev"].to_dict()
    days = dt.index.to_numpy()
    b5g = {d: g for d, g in b5.groupby("day", sort=True)}

    rows = []
    for di, day in enumerate(days):
        base_hi = max(open_px[di], prev_close[di]); base_lo = min(open_px[di], prev_close[di])
        s = sig[di]
        if not np.isfinite(base_hi) or not np.isfinite(s).all() or not np.isfinite(px[di]).all():
            continue
        ub = base_hi * (1.0 + p.vm * s); lb = base_lo * (1.0 - p.vm * s)
        mark_pos = _mark_positions(px[di], ub, lb, vwap[di], p.stop_mode, p.cooldown)
        if p.long_only:
            mark_pos = np.clip(mark_pos, 0, 1)
        if ma_gate is not None and ma_gate.get(day, 0) == 0:
            mark_pos = np.clip(mark_pos, -1, 0)  # block longs; shorts still allowed if not long_only
        g = b5g[day]; c = g["close"].to_numpy(); ret = g["ret"].to_numpy()
        n = len(c)
        idxs = bar_idx[day]
        # ffill mark positions onto the 5-min grid
        target = np.zeros(n, dtype=np.int8)
        order = np.argsort(idxs)
        for k in order:
            j = idxs[k]
            if j >= 0:
                target[j:] = mark_pos[k]
        target[-1] = 0                                 # EOD flat
        if p.overlay_entry or p.overlay_exit:
            pos = FA._overlay_path(c, ret, target.astype(float), p.overlay_entry, p.overlay_exit, False)
        else:
            pos = target
        fwd = np.empty(n); fwd[:-1] = c[1:] / c[:-1] - 1.0; fwd[-1] = 0.0
        unlev = float(np.dot(pos[:-1].astype(float), fwd[:-1]))
        turn = float(np.abs(np.diff(np.concatenate([[0], pos, [0]]).astype(float))).sum())
        rows.append({"day": day, "unlev": unlev, "turn": turn,
                     "lev": lev_map.get(day, 0.0), "n_tr": int((np.abs(np.diff(pos)) > 0).sum())})
    res = pd.DataFrame(rows).set_index("day")
    res["cost_unlev"] = res["turn"] * p.cost_bps_oneway * 1e-4
    res["net_unlev"] = res["unlev"] - res["cost_unlev"]
    if p.sizing == "vol_target":
        res["gross"] = res["lev"] * res["unlev"]; res["net"] = res["lev"] * res["net_unlev"]
    else:
        res["gross"] = res["unlev"]; res["net"] = res["net_unlev"]
    if p.eq_filter_window >= 2:
        res["net"] = apply_equity_curve_filter(res["net"], p.eq_filter_window)
        res["gross"] = apply_equity_curve_filter(res["gross"], p.eq_filter_window)
    return res


def metrics(res: pd.DataFrame, col: str = "net", lo=None, hi=None) -> dict:
    sub = res
    if lo is not None:
        sub = sub[sub.index >= lo]
    if hi is not None:
        sub = sub[sub.index <= hi]
    r = sub[col].to_numpy(); r = r[np.isfinite(r)]
    if len(r) < 5 or r.std() == 0:
        return {"days": len(r), "sharpe": 0.0}
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"days": len(r), "sharpe": round(r.mean() / r.std() * np.sqrt(TRADING_DAYS), 3),
            "ann%": round(100 * r.mean() * TRADING_DAYS, 2), "vol%": round(100 * r.std() * np.sqrt(TRADING_DAYS), 2),
            "maxDD%": round(100 * dd, 1), "turn/day": round(sub["turn"].mean(), 2)}
