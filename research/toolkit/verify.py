"""Trade-ledger lookahead verification.

Promotion source: research/experiments/lafo_kama_mr/engine.py (assert_no_lookahead).

Two-way PnL re-derivation: for each trade the recorded ``net_pts`` must equal the
independently computed ``(exit − entry) × dir − cost_pts``.  A mismatch indicates
a bar-alignment bug or cost-accounting error in the strategy engine — it does NOT
certify temporal causality (bar-windowing discipline handles that), but it is the
cheapest mechanical sanity check and catches the most common lookahead mistakes.
"""
from __future__ import annotations

import pandas as pd


def assert_no_lookahead(
    trades: pd.DataFrame,
    *,
    tol: float = 1e-6,
) -> None:
    """Assert ``net_pts == (exit − entry) × dir − cost_pts`` across all trades.

    Required columns: ``net_pts``, ``exit``, ``entry``, ``dir``, ``cost_pts``.
    Passes silently when *trades* is empty.

    Raises ``AssertionError`` with a diagnostic message on failure.
    """
    if trades.empty:
        return
    derived = (trades["exit"] - trades["entry"]) * trades["dir"] - trades["cost_pts"]
    a = float(trades["net_pts"].sum())
    b = float(derived.sum())
    assert abs(a - b) < tol, (
        f"PnL re-derivation mismatch: net_pts sum = {a:.8f}, "
        f"(exit-entry)*dir-cost sum = {b:.8f}, "
        f"diff = {abs(a - b):.2e} (tol {tol:.0e})"
    )
