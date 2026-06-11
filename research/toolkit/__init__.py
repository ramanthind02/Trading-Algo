"""Shared vectorised experiment primitives for intraday strategy research.

Each module is self-contained and importable independently.  This package re-exports
the full public API so callers can do ``from research.toolkit import read_m1, daily_sharpe``
without knowing which sub-module each symbol lives in.

Modules
-------
bars      : M1 data I/O, session filtering, intraday resampling, bar-frequency util.
sessions  : ``Session`` enum + broker-wall-clock constants (bsec, RTH/H24 windows).
metrics   : Daily-grid Sharpe, PF, win%, maxDD, per-year table, summary dict.
verify    : Two-way PnL re-derivation lookahead assert.
sizing    : Vol-target leverage overlay (size-at-entry, no intraday rebalance).
costs     : Round-trip cost model + ``FALLBACK_POINT`` live-probed price increments.
"""
from __future__ import annotations

from research.toolkit.bars import (
    INTRADAY_FLOOR,
    RESAMPLE_RULES,
    bars_per_year,
    filter_session,
    load_bars,
    read_m1,
    resample_bars,
)
from research.toolkit.costs import (
    FALLBACK_POINT,
    point_size,
    recorded_spread_cost,
    round_trip_cost,
)
from research.toolkit.metrics import (
    TRADING_DAYS,
    avg_r,
    by_year,
    daily_grid_returns,
    daily_sharpe,
    max_drawdown,
    profit_factor,
    summary,
    win_rate,
)
from research.toolkit.sessions import (
    H24_CLOSE_S,
    H24_OPEN_S,
    RTH_CLOSE_S,
    RTH_OPEN_S,
    Session,
    bsec,
    session_window,
)
from research.toolkit.sizing import apply_vol_target
from research.toolkit.verify import assert_no_lookahead

__all__ = [
    # bars
    "INTRADAY_FLOOR",
    "RESAMPLE_RULES",
    "bars_per_year",
    "filter_session",
    "load_bars",
    "read_m1",
    "resample_bars",
    # costs
    "FALLBACK_POINT",
    "point_size",
    "recorded_spread_cost",
    "round_trip_cost",
    # metrics
    "TRADING_DAYS",
    "avg_r",
    "by_year",
    "daily_grid_returns",
    "daily_sharpe",
    "max_drawdown",
    "profit_factor",
    "summary",
    "win_rate",
    # sessions
    "H24_CLOSE_S",
    "H24_OPEN_S",
    "RTH_CLOSE_S",
    "RTH_OPEN_S",
    "Session",
    "bsec",
    "session_window",
    # sizing
    "apply_vol_target",
    # verify
    "assert_no_lookahead",
]
