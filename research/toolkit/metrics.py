"""Daily-grid trade metrics for vectorised intraday strategies.

Promotion sources:
  - research/experiments/lafo_kama_mr/engine.py   — metrics(), _sharpe(),
    TRADING_DAYS, daily daily-grid return grid, maxDD computation.
  - research/experiments/lafo_kama_mr/deepen.py   — sharpe(), per_year(),
    daily_series() helper.
  - research/experiments/gold_digger_breakout/poc.py — stats() (R-multiple variant
    of the same day-grid Sharpe), by_year().

Canonical metric
~~~~~~~~~~~~~~~~
**Daily-grid Sharpe** = ``mean(daily_returns) / std(daily_returns) × √252`` where
the return series covers ALL session days in the backtest window and flat days
contribute a return of 0.  This is identical across lafo, gold_digger, and orb_ibs;
the only difference is the unit (pct-of-equity vs R-multiples) which cancels in the
ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252
"""Annualisation constant (trading days per year)."""


# ---------------------------------------------------------------------------
# Core scalar metrics
# ---------------------------------------------------------------------------

def daily_sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe on a daily-grid return array (×√252).

    Returns ``0.0`` when the series has zero variance (all-flat or single bar).
    """
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS))


def profit_factor(net_pnl: pd.Series) -> float:
    """Gross wins / gross losses.  Returns ``float('inf')`` if no losing trades."""
    wins = float(net_pnl[net_pnl > 0].sum())
    losses = float(-net_pnl[net_pnl < 0].sum())
    if losses == 0.0:
        return float("inf")
    return wins / losses


def win_rate(net_pnl: pd.Series) -> float:
    """Fraction of trades with ``net_pnl > 0``."""
    if len(net_pnl) == 0:
        return 0.0
    return float((net_pnl > 0).mean())


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown of the cumulative-sum equity curve.

    Returns the largest peak-to-trough drop in the same units as *returns*.
    """
    if len(returns) == 0:
        return 0.0
    eq = np.cumsum(returns)
    return float((np.maximum.accumulate(eq) - eq).max())


def avg_r(r_multiples: pd.Series) -> float:
    """Mean R-multiple over all trades.  Returns 0.0 for an empty series."""
    return float(r_multiples.mean()) if len(r_multiples) > 0 else 0.0


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def daily_grid_returns(
    trades: pd.DataFrame,
    session_days: np.ndarray,
    ret_col: str = "ret",
) -> np.ndarray:
    """Build a dense daily return array over *session_days*; flat days are 0.

    Per-day returns are **summed** (multiple trades per session day is valid).
    *session_days* must cover every day in the backtest window — not just trade
    days — so flat periods contribute zeros rather than being absent.
    """
    grid = pd.Series(0.0, index=pd.Index(session_days, name="day"))
    if not trades.empty:
        day_ret = trades.groupby("day")[ret_col].sum()
        grid.loc[day_ret.index] = day_ret.values
    return grid.to_numpy()


# ---------------------------------------------------------------------------
# Tabular helpers
# ---------------------------------------------------------------------------

def by_year(
    trades: pd.DataFrame,
    session_days: np.ndarray,
    ret_col: str = "ret",
) -> pd.DataFrame:
    """Per-year table with Sharpe, ret%, maxDD%, and trade count.

    Columns: ``year``, ``sharpe``, ``ret%``, ``maxDD%``, ``trades``.
    """
    grid = daily_grid_returns(trades, session_days, ret_col)
    idx = pd.DatetimeIndex(session_days)
    df_grid = pd.DataFrame({"ret": grid, "year": idx.year})
    rows = []
    for yr, g in df_grid.groupby("year"):
        r = g["ret"].to_numpy()
        n_tr = int((trades["day"].dt.year == yr).sum()) if not trades.empty else 0
        rows.append(
            {
                "year":    yr,
                "sharpe":  round(daily_sharpe(r), 2),
                "ret%":    round(100.0 * r.sum(), 1),
                "maxDD%":  round(100.0 * max_drawdown(r), 1),
                "trades":  n_tr,
            }
        )
    return pd.DataFrame(rows)


def summary(
    trades: pd.DataFrame,
    session_days: np.ndarray,
    ret_col: str = "ret",
) -> dict:
    """Full headline metrics dict.

    Keys mirror ``lafo engine.metrics`` output so existing analysis code
    can substitute ``toolkit.metrics.summary`` directly.

    Keys: ``trades``, ``tr/yr``, ``win%``, ``avgR``, ``PF``, ``sharpe``,
    ``ann_ret%``, ``vol%``, ``maxDD%``, ``tot_ret%``.

    Values are rounded to match the engine (2 dp for Sharpe/PF/ann_ret%/vol%,
    1 dp for win%/maxDD%/tot_ret%, 3 dp for avgR).
    """
    n_days = len(session_days)
    years = max(n_days / TRADING_DAYS, 1e-9)

    if trades.empty:
        return {
            "trades": 0, "tr/yr": 0.0, "win%": 0.0, "avgR": 0.0,
            "PF": 0.0, "sharpe": 0.0, "ann_ret%": 0.0, "vol%": 0.0,
            "maxDD%": 0.0, "tot_ret%": 0.0,
        }

    arr = daily_grid_returns(trades, session_days, ret_col)
    pnl_col = trades["net_pts"] if "net_pts" in trades.columns else trades[ret_col]
    r_col = trades["R"] if "R" in trades.columns else pd.Series(dtype=float)

    tot = float(arr.sum())
    ann_vol = float(arr.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(arr) > 1 else 0.0

    return {
        "trades":   int(len(trades)),
        "tr/yr":    round(len(trades) / years, 1),
        "win%":     round(100.0 * win_rate(pnl_col), 1),
        "avgR":     round(avg_r(r_col), 3),
        "PF":       round(profit_factor(pnl_col), 2),
        "sharpe":   round(daily_sharpe(arr), 2),
        "ann_ret%": round(100.0 * tot / years, 2),
        "vol%":     round(100.0 * ann_vol, 2),
        "maxDD%":   round(100.0 * max_drawdown(arr), 1),
        "tot_ret%": round(100.0 * tot, 1),
    }
