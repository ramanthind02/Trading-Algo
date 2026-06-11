"""Round-trip cost model for MT5 CFD instruments.

Promotion sources:
  - research/experiments/lafo_kama_mr/engine.py   — POINT = 0.1, cost formula
    (spread + 2*slip - swap).
  - research/experiments/lafo_kama_mr/costs.py    — recorded-spread + slippage
    sweep; confirmed POINT = 0.1 from 54.2M bid/ask ticks.
  - research/experiments/orb_ibs_nas100/poc.py    — POINT = 0.1, entry_spr +
    2*slip round-trip formula.
  - research/experiments/gold_digger_breakout/ensemble.py — POINT dict for metals
    and JPY crosses.

FALLBACK_POINT
~~~~~~~~~~~~~~
Maps symbol → price increment (the MT5 ``_Point`` / Nautilus ``price_increment``
for that instrument).  These are **live-probed** values confirmed against recorded
tick data.

⚠  WARNING: The Nautilus instruments catalog (``data/instruments/catalog.parquet``)
records **0.01** for NDX and SP500.  That is **WRONG** — likely a copy-paste from a
mini-contract spec.  The correct live-probed value is **0.1**.  Charging spread at
0.01 under-prices it 10× and produces the "spread ≈ 0" artifact noted in prior
research.  ALWAYS use FALLBACK_POINT for these two instruments; do NOT read the
catalog for them.

TODO (ADR-8): when ``data/instruments/catalog.parquet`` is corrected for NDX/SP500,
replace FALLBACK_POINT lookups with the instruments registry.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Live-probed price increments — DO NOT replace with catalog.parquet values
# for NDX and SP500 (catalog has known-wrong 0.01 for both).
# ---------------------------------------------------------------------------
_JPY_CROSSES: list[str] = [
    "USDJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY", "EURJPY",
]

FALLBACK_POINT: dict[str, float] = {
    # ---- Equity-index CFDs ------------------------------------------------
    "NDX":   0.1,    # Darwinex Nasdaq-100 cash CFD (live-probed; catalog = 0.01, WRONG)
    "SP500": 0.1,    # Darwinex S&P 500 cash CFD   (live-probed; catalog = 0.01, WRONG)
    # ---- Precious metals --------------------------------------------------
    "XAUUSD": 0.01,  # Gold CFD; 2-decimal prices → POINT = 0.01
    "XAGUSD": 0.001, # Silver CFD; from gold_digger_breakout/ensemble.py
    # ---- JPY crosses (all 3-decimal quoted) --------------------------------
    **{s: 0.001 for s in _JPY_CROSSES},
}


def point_size(symbol: str) -> float:
    """Return the price increment for *symbol* from ``FALLBACK_POINT``.

    Raises ``KeyError`` with a diagnostic message if the symbol is unknown.
    Add the symbol and its live-probed value to ``FALLBACK_POINT`` if needed.
    """
    try:
        return FALLBACK_POINT[symbol]
    except KeyError as exc:
        known = sorted(FALLBACK_POINT)
        raise KeyError(
            f"Symbol {symbol!r} not in FALLBACK_POINT (known: {known}).  "
            "Add it with its live-probed _Point value."
        ) from exc


def round_trip_cost(
    spread_pts: float,
    slip_pts: float = 0.0,
    *,
    swap_pts: float = 0.0,
) -> float:
    """Total round-trip cost in price points.

    Parameters
    ----------
    spread_pts:
        Full bid/ask spread in price points, **crossed once per round trip**
        (entry + exit together consume one spread).
    slip_pts:
        Per-side slippage in price points (×2 for entry and exit).
    swap_pts:
        Overnight financing cost in price points.
        **Positive = pay** (long paying negative carry).
        **Negative = earn** (short earning carry).

    Formula: ``spread + 2 × slip + swap``
    """
    return spread_pts + 2.0 * slip_pts + swap_pts


def recorded_spread_cost(
    spread_col: np.ndarray,
    symbol: str,
    slip_pts: float = 0.0,
) -> np.ndarray:
    """Per-trade cost array from the MT5 M1 ``spread`` column.

    MT5 stores spread as an integer count of ``_Point`` increments.  Multiply by
    ``FALLBACK_POINT[symbol]`` to convert to price units, then add per-side
    slippage.

    Parameters
    ----------
    spread_col:
        1-D array of MT5 integer (or float) spread values from the parquet
        ``spread`` column.  Each value is in units of ``_Point``.
    symbol:
        Instrument symbol (used to look up the price increment).
    slip_pts:
        Per-side slippage in price points.

    Returns
    -------
    np.ndarray
        Float64 array of per-trade round-trip costs in price units.
        Formula: ``spread_col × POINT + 2 × slip_pts``
    """
    pt = point_size(symbol)
    return spread_col.astype(np.float64) * pt + 2.0 * slip_pts
