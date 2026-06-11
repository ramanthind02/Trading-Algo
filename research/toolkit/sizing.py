"""Volatility-targeting size-at-entry overlay.

Promotion source: research/experiments/lafo_kama_mr/engine.py (apply_vol_target).

Computes per-day leverage from **strictly-past** daily close-to-close returns
(``shift(1).rolling``), then scales each trade's return by that leverage.  No
intraday rebalancing — the size is fixed at session entry.

Key finding from the LAFO study: vol-targeting HURTS strategies where the edge
lives in high-volatility regimes (deep-dip MR, breakouts).  It throttles position
size exactly when the signal fires most strongly.  Include this overlay only when
the strategy's edge is regime-agnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_vol_target(
    trades: pd.DataFrame,
    bars: pd.DataFrame,
    vol_target: float,
    vol_lookback_days: int = 14,
    lev_cap: float = 4.0,
) -> pd.DataFrame:
    """Add ``lev``, ``ret_sized``, and ``gross_ret_sized`` to *trades*.

    Leverage formula::

        lev = min(lev_cap,  vol_target / σ_day)

    where ``σ_day`` is the rolling standard deviation of daily close-to-close
    returns computed with ``shift(1).rolling(vol_lookback_days, ddof=1)``, so only
    past information is used (strict lookahead-free).

    Parameters
    ----------
    trades:
        Trade ledger; must have columns ``day``, ``ret``, and optionally
        ``gross_ret``.
    bars:
        Intraday bar frame with ``day`` and ``close`` columns (any timeframe;
        the last close per calendar day is used as the daily close).
    vol_target:
        Daily volatility target (e.g. 0.015 = 1.5 % / day).
    vol_lookback_days:
        Rolling window length for σ estimation.
    lev_cap:
        Maximum leverage; applied after the vol-target formula and after clipping.

    Returns
    -------
    pd.DataFrame
        Copy of *trades* with three new columns appended.
        ``lev``             — computed leverage (0.0 where σ is unavailable).
        ``ret_sized``       — ``ret × lev``.
        ``gross_ret_sized`` — ``gross_ret × lev`` (NaN if ``gross_ret`` absent).
    """
    t = trades.copy()

    if t.empty:
        t["lev"] = pd.Series(dtype=float)
        t["ret_sized"] = pd.Series(dtype=float)
        t["gross_ret_sized"] = pd.Series(dtype=float)
        return t

    daily_close = bars.groupby("day")["close"].last()
    daily_ret = daily_close.pct_change()
    sigma = (
        daily_ret
        .shift(1)
        .rolling(vol_lookback_days, min_periods=vol_lookback_days)
        .std(ddof=1)
    )
    # vol_target / sigma; inf (zero-sigma) capped at lev_cap; NaN → 0
    lev = (vol_target / sigma).clip(lower=0.0, upper=lev_cap)
    lev = lev.replace([np.inf, -np.inf], lev_cap).fillna(0.0)

    t["lev"] = t["day"].map(lev).fillna(0.0).to_numpy()
    t["ret_sized"] = t["ret"] * t["lev"]
    t["gross_ret_sized"] = (
        t["gross_ret"] * t["lev"] if "gross_ret" in t.columns else np.nan
    )
    return t
