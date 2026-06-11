"""Equity-curve regime filter — a universal, parameter-light strategy overlay.

Functional core (no I/O). The idea (credited to the Zorro team via Robot Wealth):
trade a sleeve only while its **own theoretical equity curve is above a smoothed
version of itself**; pause it when equity crosses below. It responds to the one
thing that actually matters — whether the strategy is making money in the current
regime — rather than a price-derived proxy that needs a tuned threshold. One knob
only: the smoothing ``window``.

Design notes
------------
* **Theoretical equity, not realized.** The gate is computed from the *ungated*
  strategy returns (the "theoretical" equity), then applied to the trading. This
  is single-pass and avoids the self-referential trap where a gated equity curve
  freezes while flat and can never cross back.
* **Lookahead-free.** The decision for bar ``t`` uses only information through
  ``t-1``: ``gate[t] = 1`` iff ``equity[t-1] >= SMA(equity, window)[t-1]``. The
  return realized at ``t`` never enters its own gate.
* **Warm-up trades.** Before the SMA exists (first ``window`` bars) the gate is
  ON, so the overlay never suppresses the sleeve purely for lack of history.
* **Additive equity.** Uses ``returns.cumsum()`` (sum-of-returns equity); for
  daily fractional returns the crossover points are indistinguishable from a
  compounded curve, and cumsum is robust to large/negative excursions.
"""
from __future__ import annotations

import pandas as pd


def equity_curve_gate(returns: pd.Series, window: int = 50) -> pd.Series:
    """Lookahead-free 0/1 trade gate from the strategy's own equity curve.

    Parameters
    ----------
    returns : pd.Series
        The sleeve's per-bar (ungated, "theoretical") returns, time-ordered.
    window : int
        Smoothing window for the equity moving average (bars). Must be >= 2.

    Returns
    -------
    pd.Series
        Float series of 1.0 (trade) / 0.0 (flat), aligned to ``returns.index``.
        ``gate[t]`` is meant to multiply ``returns[t]`` and depends only on
        equity through ``t-1``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    r = returns.astype(float).fillna(0.0)
    equity = r.cumsum()
    sma = equity.rolling(window, min_periods=window).mean()

    on_now = (equity >= sma).mask(sma.isna(), True)   # warm-up (no SMA) -> ON
    gate = on_now.shift(1, fill_value=True)           # decide t from info <= t-1
    return gate.astype(float)


def apply_equity_curve_filter(returns: pd.Series, window: int = 50) -> pd.Series:
    """Return ``returns`` with the equity-curve gate applied (flat -> 0.0)."""
    return returns.astype(float) * equity_curve_gate(returns, window)


def time_in_market(returns: pd.Series, window: int = 50) -> float:
    """Fraction of bars the gate is ON (1.0 = always invested)."""
    return float(equity_curve_gate(returns, window).mean())
