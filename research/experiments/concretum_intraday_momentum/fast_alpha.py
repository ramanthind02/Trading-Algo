"""
QuanTips #2 (Zarattini-Pagani, Feb 2026) "Improving Performance with Fast Alphas" — Phase 0
frictionless-first replication + robustness of the fast-alpha EXECUTION OVERLAY.

Three pieces:
  (A) FAST ALPHA  — 5-min one-bar reversal S_t = -sign(R_t) (+ streak-N conditioning). Strong gross,
      dies on cost = "informational, not monetizable" (paper Fig 1-2, Table 1).
  (B) BASELINE    — ATR(14)-band intraday breakout (Kaufman): bands = session_open +/- 0.5*ATR(14d);
      long if close>upper, short if close<lower; STOP = return to session open; 15-min execution
      (HH:00/15/30/45); EOD flat; vol-target 2% daily (fixed at open). Paper net Sharpe ~0.87.
  (C) OVERLAY     — condition the baseline's EXECUTION on the fast alpha: on a breakout, delay entry
      until a 5-min bar prints in the OPPOSITE (pullback) direction; on a stop, delay the exit until a
      brief counter-move. Mild (single opposite 5-min bar). Paper net Sharpe 0.87 -> 0.99.

Lookahead discipline:
  * ATR(14) uses daily TR up to t-1 (.shift(1)); leverage uses daily returns up to t-1.
  * The fast alpha at bar t uses R_t = close_t/close_{t-1}-1 (known AT close t); the position chosen at
    close t earns the FORWARD 5-min return r_t = close_{t+1}/close_t - 1. Position pos[i] * r[i] with
    r[i] forward — never the bar that informed the decision.
  * Slow breakout decisions use the mark bar's close; held forward; days are independent (start flat).
Data via engine.load_sessions (M1 RTH, broker-time). 5-min bars are clock-aligned (floor sec/300).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E

# 15-min execution marks 9:45..15:45 ET (broker = ET+7); EOD flat at 16:00 ET.
MARK_ET_MIN = {0, 15, 30, 45}
MARK_LO_S = E._bsec(9, 45)
MARK_HI_S = E._bsec(15, 45)
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
def load_5m(sym: str, y0: int, y1: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (bars5, daily). bars5: one row per 5-min RTH bar with day/sec/o/h/l/c/vol/ret/is_mark.
    daily: per-day open/high/low/close + ATR(14, Wilder, strictly-past) + leverage."""
    m1 = E.load_sessions(sym, y0, y1)
    bucket = (m1["sec"] // 300) * 300
    g = m1.groupby([m1["day"], bucket])
    b = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "vol": g["vol"].sum(),
    })
    b.index.set_names(["day", "sec"], inplace=True)
    b = b.reset_index().sort_values(["day", "sec"]).reset_index(drop=True)
    et_min = ((b["sec"] - 7 * 3600) % 3600) // 60
    b["is_mark"] = b["sec"].between(MARK_LO_S, MARK_HI_S) & et_min.isin(list(MARK_ET_MIN))
    # 5-min return within the day (first bar of day -> NaN, treated as 0 sign)
    b["ret"] = b.groupby("day")["close"].pct_change()

    # daily OHLC + ATR(14) Wilder, strictly past
    d = b.groupby("day").agg(open=("open", "first"), high=("high", "max"),
                             low=("low", "min"), close=("close", "last"))
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean().shift(1)      # ATR as of prior close
    d["daily_ret"] = d["close"] / d["close"].shift(1) - 1.0
    sigma_day = d["daily_ret"].shift(1).rolling(14).std(ddof=1)
    d["lev"] = np.minimum(4.0, 0.02 / sigma_day)
    d["lev"] = d["lev"].replace([np.inf, -np.inf], 4.0).clip(lower=0.0).fillna(0.0)
    return b, d


# --------------------------------------------------------------------------- #
# (A) FAST ALPHA — standalone 5-min reversal + streak conditioning
# --------------------------------------------------------------------------- #
def fast_alpha_streaks(b: pd.DataFrame) -> pd.DataFrame:
    """Table-1 replication: avg next-bar return (bps) after >= N consecutive same-sign 5-min bars."""
    rows = []
    for day, g in b.groupby("day"):
        r = g["ret"].to_numpy()
        rows.append(r)
    # build a flat (sign, next_ret) stream per day to avoid cross-day streaks
    recs = []
    for r in rows:
        s = np.where(np.isfinite(r), np.sign(r), 0.0)
        # streak length ending at i (same sign run)
        run = np.zeros(len(r))
        for i in range(len(r)):
            if i > 0 and s[i] == s[i - 1] and s[i] != 0:
                run[i] = run[i - 1] + 1
            else:
                run[i] = 1 if s[i] != 0 else 0
        for i in range(len(r) - 1):
            if s[i] != 0 and np.isfinite(r[i + 1]):
                recs.append((int(s[i]), int(run[i]), float(r[i + 1])))
    df = pd.DataFrame(recs, columns=["sign", "run", "next"])
    out = []
    for N in range(1, 6):
        for sgn, name in [(-1, "Down"), (1, "Up")]:
            sub = df[(df["sign"] == sgn) & (df["run"] >= N)]
            if len(sub) == 0:
                continue
            out.append({"N>=": N, "streak": name, "obs": len(sub),
                        "next_bps": round(1e4 * sub["next"].mean(), 2),
                        "t": round(sub["next"].mean() / (sub["next"].std() / np.sqrt(len(sub))), 2)})
    return pd.DataFrame(out)


def fast_alpha_standalone(b: pd.DataFrame, cost_bps_oneway: float) -> dict:
    """S_t = -sign(R_t), held for the next 5-min bar. pos[i]*r[i] with r forward (lookahead-free)."""
    rr, costs = [], []
    for day, g in b.groupby("day"):
        c = g["close"].to_numpy()
        if len(c) < 3:
            continue
        ret = c[1:] / c[:-1] - 1.0                      # ret[i] = bar i->i+1, for i=0..M-2
        # decision uses the just-closed bar ret[i] (reversal), earns the strictly-NEXT bar ret[i+1]:
        #   pos[i] = -sign(ret[i])  (i=0..M-3)  *  fwd[i] = ret[i+1]   -> forward, no same-bar capture
        pos = -np.sign(ret[:-1])
        fwd = ret[1:]                                   # forward return earned (lookahead-free)
        daypnl = pos * fwd
        turn = np.abs(np.diff(np.concatenate([[0], pos, [0]]))).sum()
        rr.append(daypnl.sum())
        costs.append(turn * cost_bps_oneway * 1e-4)
    r = np.array(rr); cst = np.array(costs)
    net = r - cst
    def sh(x): return float(x.mean() / x.std() * np.sqrt(TRADING_DAYS)) if x.std() > 0 else 0.0
    return {"days": len(r), "gross_sharpe": round(sh(r), 2),
            "net_sharpe": round(sh(net), 2),
            "gross_ann%": round(100 * r.mean() * TRADING_DAYS, 1),
            "net_ann%": round(100 * net.mean() * TRADING_DAYS, 1),
            "turn/day": round(np.mean([np.abs(np.diff(np.concatenate([[0], -np.sign((g['close'].to_numpy()[1:]/g['close'].to_numpy()[:-1]-1)[:-1]), [0]]))).sum() for _, g in b.groupby('day') if len(g) > 3]), 1)}


# --------------------------------------------------------------------------- #
# (B/C) BASELINE breakout + OVERLAY
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BParams:
    atr_mult: float = 0.5
    overlay_entry: bool = False     # delay entries until a pullback 5-min bar
    overlay_exit: bool = False      # delay stop-exits until a counter-move 5-min bar
    placebo: bool = False           # falsification: wait for a SAME-direction bar (should hurt)
    sizing: str = "vol_target"      # 'binary' | 'vol_target'
    cost_bps_oneway: float = 0.0


def _baseline_path(c, sec, is_mark, open_px, upper, lower):
    """Slow desired position on the 5-min grid (decisions at marks, held forward, stop at open)."""
    n = len(c)
    pos = np.zeros(n, dtype=np.int8)
    cur = 0
    for i in range(n):
        if is_mark[i]:
            if cur == 0:
                if c[i] > upper:
                    cur = 1
                elif c[i] < lower:
                    cur = -1
            elif cur == 1:
                if c[i] <= open_px:           # stop: back to session open
                    cur = 0
                    if c[i] < lower:           # straight through to a down-breakout
                        cur = -1
            else:  # cur == -1
                if c[i] >= open_px:
                    cur = 0
                    if c[i] > upper:
                        cur = 1
        pos[i] = cur
    pos[-1] = 0                                # EOD flat (no overnight)
    return pos


def _overlay_path(c, ret, target, ov_entry: bool, ov_exit: bool, placebo: bool = False):
    """Delay baseline transitions until a qualifying 5-min counter-move bar, per leg.
    Long entry waits for ret<0 (pullback); short entry waits for ret>0; long exit waits for ret>0;
    short exit waits for ret<0. placebo=True flips to a SAME-direction wait (falsification — should
    hurt if the gain is reversion-harvesting). ov_*=False -> execute immediately (= baseline leg).
    EOD forced flat. ret[i]=close_i/close_{i-1}-1 (known at close i; position earns forward)."""
    s = -1.0 if placebo else 1.0       # flip the inequality direction for the placebo
    n = len(c)
    pos = np.zeros(n, dtype=np.int8)
    act = 0
    for i in range(n):
        tgt = target[i]
        r = ret[i] if np.isfinite(ret[i]) else 0.0
        eod = (i == n - 1)
        # exit leg
        if act == 1 and tgt != 1:
            if (not ov_exit) or s * r > 0 or eod:
                act = 0
        elif act == -1 and tgt != -1:
            if (not ov_exit) or s * r < 0 or eod:
                act = 0
        # entry leg (only from flat; baseline always visits flat between sides)
        if act == 0 and not eod:
            if tgt == 1 and ((not ov_entry) or s * r < 0):
                act = 1
            elif tgt == -1 and ((not ov_entry) or s * r > 0):
                act = -1
        pos[i] = act
    pos[-1] = 0
    return pos


def simulate_bo(b: pd.DataFrame, d: pd.DataFrame, p: BParams) -> pd.DataFrame:
    rows = []
    lev_map = d["lev"].to_dict()
    open_map = d["open"].to_dict()
    atr_map = d["atr"].to_dict()
    for day, g in b.groupby("day", sort=True):
        atr = atr_map.get(day, np.nan)
        if not np.isfinite(atr) or atr <= 0:
            continue
        c = g["close"].to_numpy()
        sec = g["sec"].to_numpy()
        is_mark = g["is_mark"].to_numpy()
        ret = g["ret"].to_numpy()
        if len(c) < 5:
            continue
        open_px = open_map[day]
        upper = open_px + p.atr_mult * atr
        lower = open_px - p.atr_mult * atr
        base = _baseline_path(c, sec, is_mark, open_px, upper, lower)
        if p.overlay_entry or p.overlay_exit:
            pos = _overlay_path(c, ret, base, p.overlay_entry, p.overlay_exit, p.placebo)
        else:
            pos = base
        fwd = np.empty(len(c)); fwd[:-1] = c[1:] / c[:-1] - 1.0; fwd[-1] = 0.0
        unlev = float(np.dot(pos[:-1].astype(float), fwd[:-1]))
        turn = float(np.abs(np.diff(np.concatenate([[0], pos, [0]]).astype(float))).sum())
        rows.append({"day": day, "unlev": unlev, "turn": turn,
                     "lev": lev_map.get(day, 0.0), "n_tr": int((np.abs(np.diff(pos)) > 0).sum())})
    res = pd.DataFrame(rows).set_index("day")
    res["cost_unlev"] = res["turn"] * p.cost_bps_oneway * 1e-4
    res["net_unlev"] = res["unlev"] - res["cost_unlev"]
    if p.sizing == "vol_target":
        res["ret"] = res["lev"] * res["unlev"]
        res["net"] = res["lev"] * res["net_unlev"]
    else:
        res["ret"] = res["unlev"]; res["net"] = res["net_unlev"]
    return res


def bo_metrics(res: pd.DataFrame, col: str, lo=None, hi=None) -> dict:
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
            "maxDD%": round(100 * dd, 1), "tr/day": round(sub["n_tr"].mean(), 2),
            "turn/day": round(sub["turn"].mean(), 2)}
