"""
Peter / CrackingMarkets "Day Trading Volatility Breakouts Systematically [All Rules Included]"
— the SIMPLE, live-traded variant (distinct from the Zarattini academic Noise-Area model in
engine.py). Built exactly to the published rules so we can test the version a practitioner
actually trades live, rather than the academic one we found friction-fragile.

Published rules (verbatim):
  * ATR(5) on DAILY bars (true range, strictly past).
  * Bands off the session OPEN:  upper = open + k*ATR(5),  lower = open - k*ATR(5),  k = 0.4.
  * Entry = resting STOP order at the band (buy-stop at upper / sell-stop at lower), intrabar.
  * Stop-loss = retrace to the session OPEN (fixed all day). So risk per trade = k*ATR = 1R exactly.
  * ONE long attempt AND ONE short attempt per market per day (no re-entry on the same side).
  * Exit at the protective stop (=open) OR at the end of day (market-on-close).
  * Fixed-fractional risk sizing (0.33%-1% of account per trade). Sharpe is invariant to the
    fraction, so PnL is accounted in R-multiples (1R = k*ATR = the stop distance).

Why this can work where the academic model didn't:
  * One shot per side -> at most 2 trades/day/market (vs the Noise-Area model's ~1.8 turns/day of
    semi-hourly whipsaw). Far less friction.
  * Clean asymmetric payoff: lose exactly 1R, or ride the day's trend to the close.
  * A diversified basket (SPY/IWM/QQQ/GLD/DIA) — diversification is most of the live Sharpe.

Fill realism (conservative — pre-empts the "optimistic fill" critique):
  * Buy-stop fill  = max(upper, bar_open)  (gap through the level fills you worse).
  * Sell-stop fill = min(lower, bar_open).
  * Long protective-stop fill  = min(open_px, bar_open);  short = max(open_px, bar_open).
  * If the SAME minute that triggers the entry also touches the protective stop, the trade is
    booked as a whipsaw loss (-1R). Adverse-order assumption on M1.
  * EOD exit fills at the last RTH bar's close.

Lookahead discipline:
  * ATR(5) uses daily TR up to t-1 (.shift(1)); the open is the first RTH bar's open (known at t).
  * The vol filter (optional) compares ATR/open to a strictly-past rolling median.
  * Entries/exits are driven by intrabar highs/lows in chronological M1 order; no future bar is read.

Data: research.experiments...engine.load_sessions (M1 RTH, broker-time, ET = stored-7h).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.experiments.concretum_intraday_momentum import engine as E

TRADING_DAYS = 252


@dataclass(frozen=True)
class PParams:
    atr_n: int = 5                 # ATR lookback (days)
    k: float = 0.4                 # band/stop multiple of ATR
    longs: bool = True
    shorts: bool = True
    cost_bps_oneway: float = 1.0   # bps of fill price per side (entry + exit = round trip)
    vol_filter_n: int = 0          # 0 = off; else only trade days with ATR/open >= past rolling median
    vol_filter_q: float = 0.5      # quantile threshold for the vol filter (0.5 = median)
    # fast-alpha entry overlay: on a breakout, DON'T chase the band — rest a LIMIT a fraction of R
    # back toward the open, fillable only from the NEXT bar (no intrabar lookahead). 0 = off (band stop).
    entry_pullback: float = 0.0    # fraction of R the limit sits below upper / above lower
    entry_placebo: bool = False    # falsification: put the limit on the WRONG side (chase further)


# --------------------------------------------------------------------------- #
# Daily table: open, RTH OHLC, ATR(n) strictly-past, normalized vol + filter gate
# --------------------------------------------------------------------------- #
def _daily_table(win: pd.DataFrame, p: PParams) -> pd.DataFrame:
    d = (win.groupby("day")
         .agg(open=("open", "first"), high=("high", "max"),
              low=("low", "min"), close=("close", "last"))
         .sort_index())
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()],
                   axis=1).max(axis=1)
    # Wilder ATR(n), as of the PRIOR close (strictly past)
    d["atr"] = tr.ewm(alpha=1.0 / p.atr_n, adjust=False).mean().shift(1)
    d["natr"] = d["atr"] / d["open"]                       # normalized vol proxy
    if p.vol_filter_n > 0:
        thr = d["natr"].shift(1).rolling(p.vol_filter_n).quantile(p.vol_filter_q)
        d["trade_ok"] = (d["natr"] >= thr)
    else:
        d["trade_ok"] = True
    return d


# --------------------------------------------------------------------------- #
# One day: walk M1 bars, one long + one short stop-entry, stop at open, EOD flat.
# Returns list of trade R-multiples (net of round-trip cost).
# --------------------------------------------------------------------------- #
def _day_trades(o, h, l, c, open_px, atr, p: PParams) -> list[float]:
    k = p.k
    R = k * atr                                            # 1R in price = stop distance
    if not np.isfinite(R) or R <= 0:
        return []
    upper = open_px + R
    lower = open_px - R
    cost_px = p.cost_bps_oneway * 1e-4                     # fraction of fill price, per side
    n = len(c)

    long_used = short_used = False
    long_pos = short_pos = False
    l_entry = s_entry = np.nan
    trades: list[float] = []

    def book(entry_fill, exit_fill, direction):
        # R-multiple, net of round-trip cost (cost in price -> R via /R)
        gross = (exit_fill - entry_fill) if direction == 1 else (entry_fill - exit_fill)
        cost = cost_px * (abs(entry_fill) + abs(exit_fill))
        return (gross - cost) / R

    for i in range(n):
        oi, hi, li, ci = o[i], h[i], l[i], c[i]
        last = (i == n - 1)

        # ---- manage an open LONG: protective stop at open_px fires intrabar (even on the
        #      last bar — a resting stop is live all session), else EOD market-on-close ----
        if long_pos:
            if li <= open_px:
                fill = min(open_px, oi)                    # gap-through fills worse
                trades.append(book(l_entry, fill, 1))
                long_pos = False
            elif last:
                trades.append(book(l_entry, ci, 1))        # survived to the close -> MOC
                long_pos = False
        # ---- manage an open SHORT ----
        if short_pos:
            if hi >= open_px:
                fill = max(open_px, oi)
                trades.append(book(s_entry, fill, -1))
                short_pos = False
            elif last:
                trades.append(book(s_entry, ci, -1))
                short_pos = False

        if last:
            break

        # ---- new entries (resting stop orders), one attempt per side ----
        if p.longs and not long_used and not long_pos and hi >= upper:
            l_entry = max(upper, oi)                        # buy-stop, gap fills worse
            long_pos = True
            long_used = True
            # same-bar whipsaw: triggered long also hits the protective stop this minute
            if li <= open_px:
                fill = min(open_px, oi)
                trades.append(book(l_entry, fill, 1))
                long_pos = False
        if p.shorts and not short_used and not short_pos and li <= lower:
            s_entry = min(lower, oi)
            short_pos = True
            short_used = True
            if hi >= open_px:
                fill = max(open_px, oi)
                trades.append(book(s_entry, fill, -1))
                short_pos = False

    return trades


# --------------------------------------------------------------------------- #
# Simulate one market -> per-day frame of net R (and trade count).
# --------------------------------------------------------------------------- #
def simulate(sym: str, y0: int, y1: int, p: PParams) -> pd.DataFrame:
    win = E.load_sessions(sym, y0, y1)
    d = _daily_table(win, p)
    atr_map = d["atr"].to_dict()
    ok_map = d["trade_ok"].to_dict()

    rows = []
    for day, g in win.groupby("day", sort=True):
        atr = atr_map.get(day, np.nan)
        if not np.isfinite(atr) or atr <= 0 or not bool(ok_map.get(day, False)):
            continue
        o = g["open"].to_numpy(); h = g["high"].to_numpy()
        l = g["low"].to_numpy(); c = g["close"].to_numpy()
        if len(c) < 30:
            continue
        open_px = float(o[0])
        if open_px <= 0:
            continue
        trades = _day_trades(o, h, l, c, open_px, atr, p)
        rows.append({"day": day, "R": float(np.sum(trades)), "n_tr": len(trades)})
    return pd.DataFrame(rows).set_index("day").sort_index()


# --------------------------------------------------------------------------- #
# Basket: sum per-day R across markets (each trade risks the same fixed fraction,
# so the combined account PnL is the SUM of per-market R, aligned on date).
# --------------------------------------------------------------------------- #
def simulate_basket(syms, y0: int, y1: int, p: PParams) -> pd.DataFrame:
    frames = {}
    for s in syms:
        try:
            frames[s] = simulate(s, y0, y1, p)["R"]
        except FileNotFoundError:
            continue
    if not frames:
        return pd.DataFrame()
    R = pd.concat(frames, axis=1).sort_index()
    out = pd.DataFrame(index=R.index)
    out["R"] = R.sum(axis=1, min_count=1)                 # combined daily R (fixed risk/trade)
    out["n_mkt"] = R.notna().sum(axis=1)
    out["R"] = out["R"].where(out["n_mkt"] > 0)
    return out.dropna(subset=["R"])


# --------------------------------------------------------------------------- #
# Metrics. R is additive (non-compounded, fixed-fractional risk) -> Sharpe on daily R.
# risk_frac scales returns to account terms (Sharpe-invariant). ann% / maxDD% reported
# at risk_frac per R (default 0.33% = the original post's per-trade risk).
# --------------------------------------------------------------------------- #
def metrics(res: pd.DataFrame, lo=None, hi=None, risk_frac: float = 0.0033) -> dict:
    sub = res
    if lo is not None:
        sub = sub[sub.index >= lo]
    if hi is not None:
        sub = sub[sub.index <= hi]
    r = sub["R"].to_numpy()
    r = r[np.isfinite(r)]
    if len(r) < 5 or r.std() == 0:
        return {"days": len(r), "sharpe": 0.0}
    acc = r * risk_frac                                    # daily account return
    eq = np.cumsum(acc)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    win_tr = r[r > 0]
    return {
        "days": len(r),
        "years": round(len(r) / TRADING_DAYS, 2),
        "sharpe": round(r.mean() / r.std() * np.sqrt(TRADING_DAYS), 3),
        "ann%": round(100 * acc.mean() * TRADING_DAYS, 2),
        "vol%": round(100 * acc.std() * np.sqrt(TRADING_DAYS), 2),
        "maxDD%": round(100 * dd, 2),
        "tot%": round(100 * eq[-1], 1),
        "meanR/day": round(float(r.mean()), 4),
        "tr/day": round(float(sub["n_tr"].mean()), 2) if "n_tr" in sub else np.nan,
        "hit%day": round(100 * (r > 0).mean(), 1),
    }
