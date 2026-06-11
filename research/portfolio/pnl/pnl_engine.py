"""Dual-lane P&L engine: protocol + vectorized default (WP-3 Unit 1).

Both lanes consume the identical ``position_fraction`` frame and terminate at the
same downstream metrics. The vectorized lane is the FROZEN parity baseline; this
module wraps it verbatim behind a ``PnLEngine`` protocol so call sites can be
routed through one indirection without changing behaviour.

The ``nautilus`` lane (realistic intraday/quote-driven execution via
``BacktestEngine``) lives in :mod:`research.portfolio.pnl.nautilus_engine` and is
opt-in (``pnl_engine="nautilus"``); it is additive and never overwrites this
vectorized baseline.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ensemble.portfolio_impl.portfolio_tester import (
    calculate_strategy_returns_from_positions,
)


@runtime_checkable
class PnLEngine(Protocol):
    """Convert a ``position_fraction`` frame + candles into a return series."""

    def returns_from_positions(
        self,
        positions_df: pd.DataFrame,
        candles_df: pd.DataFrame,
    ) -> pd.Series:
        ...


class VectorizedPnLEngine:
    """DEFAULT lane — delegates verbatim to the frozen vectorized baseline.

    Wraps :func:`calculate_strategy_returns_from_positions` with its default
    ``instrument_return_kind='log_intraday'`` convention. No math is reimplemented
    here; behaviour is byte-identical to calling the frozen function directly.
    """

    def returns_from_positions(
        self,
        positions_df: pd.DataFrame,
        candles_df: pd.DataFrame,
    ) -> pd.Series:
        return calculate_strategy_returns_from_positions(
            positions_df,
            candles_df,
            instrument_return_kind="log_intraday",
        )


def make_pnl_engine(
    kind: str,
    *,
    multi_ticker: bool = False,
    **engine_kwargs: object,
) -> PnLEngine:
    """Build the P&L engine selected by ``kind``.

    Parameters
    ----------
    kind : {"vectorized", "nautilus"}
        ``"vectorized"`` returns the frozen-baseline wrapper (default research
        path). ``"nautilus"`` returns the realistic ``BacktestEngine`` lane.
    multi_ticker : bool
        For ``"nautilus"``: wrap the single-instrument lane in
        :class:`MultiTickerNautilusPnLEngine` so a multi-ticker ``position_fraction``
        frame (the portfolio pipeline's combined positions) is handled per
        instrument and combined. The vectorized lane is multi-ticker natively.
    **engine_kwargs
        Forwarded to ``NautilusPnLEngine`` (e.g. ``window_policy``,
        ``execution_policy``, ``rollover_minute``, ``measure_spread``).
    """
    if kind == "vectorized":
        return VectorizedPnLEngine()
    if kind == "nautilus":
        # Imported lazily so the default (vectorized) path never pays the
        # Nautilus import cost, and the parity baseline stays decoupled.
        from research.portfolio.pnl.nautilus_engine import (
            MultiTickerNautilusPnLEngine,
            NautilusPnLEngine,
        )

        base = NautilusPnLEngine(**engine_kwargs)  # type: ignore[arg-type]
        return MultiTickerNautilusPnLEngine(base=base) if multi_ticker else base
    raise ValueError(
        f"Unknown pnl_engine kind {kind!r}; expected 'vectorized' or 'nautilus'."
    )
