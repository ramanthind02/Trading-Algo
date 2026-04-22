"""
Pure delta math for ETF rebalancing.

Given target shares (from the forecast), current positions (from IB), and
latest prices (from IB snapshots), produce a deterministic list of
:class:`OrderIntent` objects. No I/O, no mutation, no side effects.

Isolating this in a pure function is deliberate: the money math is the
single most dangerous piece of the execution path, and the only way to
trust it is to test it exhaustively without mocking IB or Telegram.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from typing import Dict, List

from execution.models import ExecutionConfig, OrderIntent, OrderSide


_SHARE_QUANTUM = Decimal("0.0001")


def compute_order_intents(
    target_shares: Dict[str, Decimal],
    current_positions: Dict[str, Decimal],
    prices: Dict[str, float],
    config: ExecutionConfig,
) -> List[OrderIntent]:
    """Return the minimal set of orders that moves current → target.

    Parameters
    ----------
    target_shares
        Desired share count per ETF symbol (Decimal). Missing key = 0.
    current_positions
        Currently held share count per ETF (Decimal). Missing key = 0.
    prices
        Latest reference price per ETF (float). Required for every ETF that
        appears in either ``target_shares`` or ``current_positions`` with a
        non-zero share count, else ``KeyError`` is raised (fail loud).
    config
        Supplies ``min_rebalance_shares`` and ``min_rebalance_notional_usd``
        dead-band thresholds that suppress micro-churn.

    Returns
    -------
    list[OrderIntent]
        Sorted by ETF symbol for deterministic ordering.
    """
    intents: List[OrderIntent] = []
    etfs = set(target_shares) | set(current_positions)

    for etf in sorted(etfs):
        target = target_shares.get(etf, Decimal("0"))
        current = current_positions.get(etf, Decimal("0"))
        delta = target - current
        abs_delta = abs(delta)

        if abs_delta < config.min_rebalance_shares:
            continue

        if etf not in prices:
            raise KeyError(f"Missing price for {etf} (delta={delta})")
        price = prices[etf]
        notional = float(abs_delta) * price

        if notional < config.min_rebalance_notional_usd:
            continue

        shares_q = abs_delta.quantize(_SHARE_QUANTUM, rounding=ROUND_HALF_EVEN)
        if shares_q <= Decimal("0"):
            continue

        intents.append(OrderIntent(
            etf=etf,
            side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
            shares=shares_q,
            est_price=price,
            est_notional=float(shares_q) * price,
        ))

    return intents
