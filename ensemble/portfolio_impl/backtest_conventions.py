"""Single source of truth for the backtest P&L lanes' temporal conventions.

Two lanes turn the same ``position_fraction`` frame into a return series:

* the **vectorized baseline** —
  :func:`ensemble.portfolio_impl.portfolio_tester.calculate_strategy_returns_from_positions`
  (fast, used for every research sweep / permutation), and
* the **realistic Nautilus lane** —
  :class:`research.portfolio.pnl.nautilus_engine.NautilusPnLEngine`
  (correct-by-construction, event-driven execution).

They must encode the *same* economic model. When that model lived implicitly and
was re-derived independently in each lane, the lanes silently drifted — the cause
of both the one-day-lookahead bug (adapter held the position a day early) and the
stale-open bug (vectorized read an untradeable open). This module states the
conventions once so the lanes consume a single definition, and
``tests/research/test_nautilus_pnl_lane.py::test_nautilus_vs_vectorized_varying_signal_reconciles``
is the gate that fails the build if they ever diverge again (it drives a *varying*
signal — a constant target is shift-invariant and cannot expose a misalignment).

Conventions
-----------
1. **No-lookahead holding shift.** A ``position_fraction`` stamped at session
   ``t`` is the position *decided* using information available through ``t``; it is
   *held* over the NEXT session ``t+1``. Equivalently, the position held during
   session ``X`` is the one decided at ``X-1``. Both lanes apply this one-session
   forward shift — :func:`shift_positions_to_holding` here (Nautilus adapter), and
   the candle-grid ``next_datetime`` merge in the vectorized lane (equivalent when
   the position and candle session dates align). Omitting it makes a lane peek one
   day ahead (lookahead), which silently inflates Sharpe.

2. **Intraday entry/exit prices.** Under the intraday (open→close) window the
   entry is the first *tradeable* price of the session and the exit is the session
   close. For the MT5 CFD feed the first M1 bar's OPEN is a stale price carried
   across the 00:00–01:00 financing dead zone, so the first tradeable price is
   that bar's CLOSE — see
   :func:`data_platform.providers.mt5.cfd_candles._first_tradeable_open`. The
   Nautilus lane fills the entry at that bar's close; the vectorized lane's daily
   ``open`` is de-staled to the same price so ``log(close/open)`` is a genuine
   intraday return.
"""
from __future__ import annotations

import pandas as pd


def shift_positions_to_holding(positions_by_session: pd.Series) -> pd.Series:
    """Apply the no-lookahead holding shift (convention #1) to a session series.

    Given ``position_fraction`` indexed by session, return the position *held*
    during each session: ``held[t] = decided[t-1]``. The index is sorted first and
    the first session is dropped (no prior decision to hold). This is the one
    definition of the shift for the Nautilus lane — do not re-implement it inline.

    Parameters
    ----------
    positions_by_session : pd.Series
        ``position_fraction`` indexed by session (one value per session).

    Returns
    -------
    pd.Series
        The held-position series, shifted forward one session, first session dropped.
    """
    return positions_by_session.sort_index().shift(1).dropna()
