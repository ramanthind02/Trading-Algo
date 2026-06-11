"""Industry-Timing METHOD applied to OUR tradeable instruments (futures/CFD book).

This is NOT a replication of Zarattini & Antonacci (2025) on French industries.
It ports the *method* to the instruments we actually trade and tests whether the
edge exists and is robust here.

Method (faithful to the paper, adapted to real OHLC so we use a real ATR instead
of the close-only 1.4x MAD proxy):

  Entry (long-only):  close_t >= UpperBand_{t-1}
      UpperBand = min( DonchianUp(20) , KeltnerUp(20, k) )
      DonchianUp(n)  = rolling max of HIGH over prior n bars
      KeltnerUp(n,k) = EMA(close, n) + k * ATR(n)

  Exit (ratcheted trailing stop, never moves down):
      LowerBand = max( DonchianDown(40) , KeltnerDown(40, k) )
      DonchianDown(n) = rolling min of LOW over prior n bars
      KeltnerDown(n,k)= EMA(close, n) - k * ATR(n)
      stop_t = max(stop_{t-1}, LowerBand_{t-1}) ; exit when close_t < stop_t

  Asymmetry: entry uses 20-bar windows (fast to trigger), exit uses 40-bar
  windows (slow to trigger) -> structural long bias / slow exit.

  Sizing: each *currently-long* instrument gets w_j = (target_vol/N)/sigma_j,
  sigma_j = vol_window-day stdev of daily returns. Total |exposure| capped at
  max_leverage; rescaled if exceeded. Futures are collateralised so the rf/borrow
  term cancels for a relative (Sharpe) comparison -> rf=0.

Lookahead discipline: every band is computed as-of t (inclusive) then .shift(1)
so the day-t decision sees only data <= t-1; the resulting weight earns the
day t+1 return (weights .shift(1) before multiplying returns). Verified two ways
in run_headline.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OHLC_ROOT = REPO_ROOT / "data" / "ohlc_data"
TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    # entry (upper) bands
    entry_don: int = 20
    entry_ema: int = 20
    entry_atr: int = 20
    entry_k: float = 2.0
    # exit (lower) bands
    exit_don: int = 40
    exit_ema: int = 40
    exit_atr: int = 40
    exit_k: float = 2.0
    # sizing
    target_vol_daily: float = 0.015     # total portfolio daily vol target (paper: 1.5%)
    vol_window: int = 14
    max_leverage: float = 2.0
    n_assets: Optional[int] = None       # divisor N; None -> len(universe)
    ratchet: bool = True                 # trailing stop never moves down
    long_only: bool = True               # if False, symmetric (allow short on lower-band break)
    # data / window
    adj: str = "ratio"                   # ratio | default | unadj
    start: Optional[str] = None
    end: Optional[str] = None
    # costs (one-way, basis points on |delta weight|)
    cost_bps: float = 0.0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _file_for(ticker: str, adj: str) -> Path:
    d = OHLC_ROOT / ticker
    suffix = {"ratio": "_ratio", "default": "", "unadj": "_unadj"}[adj]
    cand = d / f"D_{ticker}{suffix}.parquet"
    if cand.exists():
        return cand
    # fall back to the plain file (TLT, AUDNZD have no _ratio variant)
    return d / f"D_{ticker}.parquet"


def load_ohlc(ticker: str, adj: str = "ratio") -> pd.DataFrame:
    df = pd.read_parquet(_file_for(ticker, adj))
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_panel(tickers: list[str], cfg: Config) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        df = load_ohlc(t, cfg.adj)
        if cfg.start:
            df = df[df.index >= pd.Timestamp(cfg.start)]
        if cfg.end:
            df = df[df.index <= pd.Timestamp(cfg.end)]
        out[t] = df
    return out


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Simple-mean ATR over n bars (consistent with the paper's mean-|delta| proxy)."""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def bands(df: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series]:
    """Return (UpperBand_asof_t, LowerBand_asof_t). Caller shifts by 1 for decisions."""
    c = df["close"]
    # Upper (entry)
    don_up = df["high"].rolling(cfg.entry_don, min_periods=cfg.entry_don).max()
    kelt_up = c.ewm(span=cfg.entry_ema, adjust=False, min_periods=cfg.entry_ema).mean() \
        + cfg.entry_k * _atr(df, cfg.entry_atr)
    upper = np.minimum(don_up, kelt_up)
    # Lower (exit)
    don_dn = df["low"].rolling(cfg.exit_don, min_periods=cfg.exit_don).min()
    kelt_dn = c.ewm(span=cfg.exit_ema, adjust=False, min_periods=cfg.exit_ema).mean() \
        - cfg.exit_k * _atr(df, cfg.exit_atr)
    lower = np.maximum(don_dn, kelt_dn)
    return upper, lower


def position_state(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Path-dependent long/flat (or long/short) state machine -> position in {0,1} (or {-1,0,1}).

    State at close t is the position HELD INTO t+1 (so the caller multiplies by
    next-day return, equivalently shift(1) before multiplying same-day return).
    """
    upper_t, lower_t = bands(df, cfg)
    # prior-day bands (data <= t-1) for the day-t decision
    up = upper_t.shift(1).to_numpy()
    lo = lower_t.shift(1).to_numpy()
    close = df["close"].to_numpy()
    n = len(close)
    pos = np.zeros(n, dtype=np.float64)
    state = 0  # 0 flat, 1 long, -1 short (only if not long_only)
    stop = np.nan
    sstop = np.nan  # short trailing stop (upper band ratcheted down)
    for t in range(n):
        u, d = up[t], lo[t]
        px = close[t]
        if state == 1:
            if cfg.ratchet:
                if not np.isnan(d):
                    stop = d if np.isnan(stop) else max(stop, d)
            else:
                stop = d
            if (not np.isnan(stop)) and px < stop:
                state = 0
                stop = np.nan
        elif state == -1:
            if cfg.ratchet:
                if not np.isnan(u):
                    sstop = u if np.isnan(sstop) else min(sstop, u)
            else:
                sstop = u
            if (not np.isnan(sstop)) and px > sstop:
                state = 0
                sstop = np.nan
        # entries (only from flat)
        if state == 0:
            if (not np.isnan(u)) and px >= u:
                state = 1
                stop = d if not np.isnan(d) else np.nan
            elif (not cfg.long_only) and (not np.isnan(d)) and px <= d:
                state = -1
                sstop = u if not np.isnan(u) else np.nan
        pos[t] = state
    return pd.Series(pos, index=df.index)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def _daily_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change()


def backtest(tickers: list[str], cfg: Config) -> dict:
    panel = load_panel(tickers, cfg)
    # align on union of dates
    idx = sorted(set().union(*[df.index for df in panel.values()]))
    idx = pd.DatetimeIndex(idx)

    pos = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    ret = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    vol = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        df = panel[t].reindex(idx)
        r = _daily_returns(panel[t]).reindex(idx)
        ret[t] = r
        vol[t] = r.rolling(cfg.vol_window, min_periods=cfg.vol_window).std()
        pos[t] = position_state(panel[t], cfg).reindex(idx).fillna(0.0)

    N = cfg.n_assets or len(tickers)
    per_asset_target = cfg.target_vol_daily / N
    # raw inverse-vol weight where in-position; guard vol>0 & finite
    safe_vol = vol.where(vol > 0)
    raw_w = (per_asset_target / safe_vol) * pos
    raw_w = raw_w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # leverage cap on gross exposure
    raw_gross = raw_w.abs().sum(axis=1)
    scale = np.where(raw_gross > cfg.max_leverage, cfg.max_leverage / raw_gross.replace(0, np.nan), 1.0)
    scale = pd.Series(scale, index=idx).fillna(1.0)
    w = raw_w.mul(scale, axis=0)
    gross = w.abs().sum(axis=1)   # ACTUAL traded gross (post-cap)

    # weights decided at close t earn return t+1  -> shift(1)
    w_held = w.shift(1).fillna(0.0)
    ret_filled = ret.fillna(0.0)
    gross_ret = (w_held * ret_filled).sum(axis=1)

    # turnover & costs (one-way bps on |delta w|)
    dturn = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = (cfg.cost_bps / 1e4) * dturn
    # cost charged when the new weights are set (close t) -> affects t+1 too; align with gross
    net_ret = gross_ret - cost.shift(1).fillna(0.0)

    # trim warmup (all-zero leading region)
    first_trade = w_held.abs().sum(axis=1)
    nonzero = first_trade[first_trade > 0].index
    if len(nonzero):
        start = nonzero[0]
        gross_ret = gross_ret.loc[start:]
        net_ret = net_ret.loc[start:]
        w = w.loc[start:]
        w_held = w_held.loc[start:]
        ret_filled = ret_filled.loc[start:]
        dturn = dturn.loc[start:]
        pos = pos.loc[start:]
        gross = gross.loc[start:]

    return {
        "cfg": cfg,
        "tickers": tickers,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "weights": w,
        "weights_held": w_held,
        "ret": ret_filled,
        "pos": pos,
        "gross_exposure": gross,
        "turnover_daily": dturn,
        "metrics_gross": metrics(gross_ret, dturn),
        "metrics_net": metrics(net_ret, dturn),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(r: pd.Series, turnover: Optional[pd.Series] = None) -> dict:
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return {"sharpe": np.nan, "ann_return": np.nan, "ann_vol": np.nan,
                "mdd": np.nan, "calmar": np.nan, "sortino": np.nan,
                "hit": np.nan, "skew": np.nan, "n_days": len(r),
                "ann_turnover": np.nan}
    eq = (1.0 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (TRADING_DAYS / n) - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    downside = r[r < 0]
    dd = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    sortino = r.mean() / dd * np.sqrt(TRADING_DAYS) if dd > 0 else np.nan
    roll_max = eq.cummax()
    mdd = (eq / roll_max - 1.0).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    out = {
        "sharpe": float(sharpe),
        "ann_return": float(cagr),
        "ann_vol": float(ann_vol),
        "mdd": float(mdd),
        "calmar": float(calmar),
        "sortino": float(sortino),
        "hit": float((r > 0).mean()),
        "skew": float(r.skew()),
        "n_days": int(n),
    }
    if turnover is not None:
        out["ann_turnover"] = float(turnover.reindex(r.index).fillna(0.0).mean() * TRADING_DAYS)
    return out


# --------------------------------------------------------------------------- #
# Benchmarks (to attribute the edge: is it timing, or just levered long beta?)
# --------------------------------------------------------------------------- #
def benchmark_buyhold(tickers: list[str], cfg: Config) -> dict:
    """Always-long, EW, vol-targeted (timing OFF). The FAIR timing-isolation benchmark:
    identical sizing machinery to backtest(), position pinned to 1. (A true static
    equal-notional hold is a worse benchmark — dominated by the most volatile leg.)
    Costs are applied to its OWN rebalancing turnover so net-vs-net is apples-to-apples."""
    panel = load_panel(tickers, cfg)
    idx = pd.DatetimeIndex(sorted(set().union(*[df.index for df in panel.values()])))
    ret = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    vol = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        r = _daily_returns(panel[t]).reindex(idx)
        ret[t] = r
        vol[t] = r.rolling(cfg.vol_window, min_periods=cfg.vol_window).std()
    N = cfg.n_assets or len(tickers)
    per = cfg.target_vol_daily / N
    safe_vol = vol.where(vol > 0)
    raw_w = (per / safe_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = raw_w.abs().sum(axis=1)
    scale = pd.Series(np.where(gross > cfg.max_leverage, cfg.max_leverage / gross.replace(0, np.nan), 1.0),
                      index=idx).fillna(1.0)
    w = raw_w.mul(scale, axis=0)
    w_held = w.shift(1).fillna(0.0)
    gross_pr = (w_held * ret.fillna(0.0)).sum(axis=1)
    dturn = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = (cfg.cost_bps / 1e4) * dturn
    net_pr = gross_pr - cost.shift(1).fillna(0.0)
    start = w_held.abs().sum(axis=1).gt(0).idxmax()
    gross_pr = gross_pr.loc[start:]
    net_pr = net_pr.loc[start:]
    dturn = dturn.loc[start:]
    return {"ret": gross_pr, "ret_net": net_pr,
            "metrics": metrics(gross_pr, dturn), "metrics_net": metrics(net_pr, dturn)}
