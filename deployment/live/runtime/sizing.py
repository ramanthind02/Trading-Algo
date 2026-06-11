"""Position-sizing bridge: target ``position_fraction`` → Nautilus lot orders.

The vault portfolio emits a per-instrument ``position_fraction`` (signed target
exposure as a fraction of sizing capital). Translating that into a broker lot
count is *exactly* the tested logic already used by the legacy CFD path, so this
module is a thin, reusable bridge over
:func:`execution.mt5_rebalancer.compute_target_signed_lots` — it does NOT
re-derive the math.

Two adaptations are made for the Nautilus world:

1. ``symbol_info_from_nautilus_instrument`` builds the
   :class:`execution.mt5_models.MT5SymbolInfo` the sizer expects from a loaded
   Nautilus ``Instrument``. The vendored MT5 adapter stores the MT5
   ``trade_contract_size`` as the instrument's ``lot_size``, ``volume_step`` as
   ``size_increment`` and ``volume_min/max`` as ``min/max_quantity`` — and order
   quantities are expressed in **lots** — so the mapping is direct and the
   returned signed-lot count drops straight into a Nautilus order ``Quantity``.
2. ``net_rebalance`` computes the single delta order for a **netting** account
   (Nautilus tracks one net position per instrument): ``delta = target −
   current``, snapped to the lot step, with the same ``min_rebalance_lots`` /
   ``min_rebalance_notional_usd`` dead-band the legacy planner applies. (The
   hedging per-ticket close path lives in ``mt5_rebalancer.plan_symbol_actions``
   and is reused for the future real-order tier, not here.)
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from execution.mt5_models import MT5SymbolInfo, OrderSide, TradeMode
from execution.mt5_rebalancer import SizingConfig, compute_target_signed_lots

_TOL = 1e-9


def _snap(value: float, step: float) -> float:
    """Round ``value`` to the nearest multiple of ``step`` (step > 0)."""
    if step <= 0:
        raise ValueError(f"volume_step must be > 0, got {step}")
    rounded = round(value / step) * step
    digits = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(rounded, digits + 2)


def symbol_info_from_nautilus_instrument(instrument: object) -> MT5SymbolInfo:
    """Derive the sizer's :class:`MT5SymbolInfo` from a Nautilus ``Instrument``.

    Best-effort: ``trade_mode``/``spread`` are not carried by the Nautilus
    instrument, so they default to ``FULL`` / ``0``. When the live MT5
    connection is available the caller may build a richer ``MT5SymbolInfo``
    directly from ``mt5.symbol_info`` instead; the sizing math only reads
    ``trade_contract_size`` and the ``volume_*`` fields.
    """
    raw_symbol = getattr(instrument, "raw_symbol", None)
    name = str(raw_symbol) if raw_symbol is not None else str(getattr(instrument, "id", ""))
    contract_size = float(getattr(instrument, "lot_size"))
    step = float(getattr(instrument, "size_increment"))
    min_qty = getattr(instrument, "min_quantity", None)
    max_qty = getattr(instrument, "max_quantity", None)
    price_increment = float(getattr(instrument, "price_increment"))
    return MT5SymbolInfo(
        name=name,
        trade_mode=TradeMode.FULL,
        point=price_increment,
        digits=int(getattr(instrument, "price_precision", 0)),
        trade_contract_size=contract_size,
        volume_min=float(min_qty) if min_qty is not None else step,
        volume_max=float(max_qty) if max_qty is not None else 1e12,
        volume_step=step,
        spread=0,
        visible=True,
    )


def build_sizing_config(
    *,
    sizing_basis_usd: float,
    lot_size_ceiling: float = 100.0,
    min_rebalance_lots: float = 0.0,
    min_rebalance_notional_usd: float = 0.0,
) -> SizingConfig:
    """Construct the :class:`SizingConfig` from account equity + dead-band knobs."""
    return SizingConfig(
        sizing_basis_usd=sizing_basis_usd,
        lot_size_ceiling=lot_size_ceiling,
        min_rebalance_lots=min_rebalance_lots,
        min_rebalance_notional_usd=min_rebalance_notional_usd,
    )


def target_signed_lots(
    *,
    position_fraction: float,
    price: float,
    symbol: MT5SymbolInfo,
    sizing: SizingConfig,
) -> Optional[float]:
    """Signed target lots for one instrument (delegates to the tested sizer).

    Returns ``None`` when the target rounds below ``volume_min`` (skip), else a
    signed lot count (``0.0`` means "go flat").
    """
    return compute_target_signed_lots(
        position_fraction=position_fraction,
        price=price,
        symbol=symbol,
        sizing=sizing,
    )


@dataclass(frozen=True)
class RebalanceOrder:
    """A single netting delta order to bring the net position toward target."""

    side: OrderSide
    volume: float  # lots, always > 0


def net_rebalance(
    *,
    target_signed_lots: float,
    current_signed_lots: float,
    symbol: MT5SymbolInfo,
    price: float,
    min_rebalance_lots: float = 0.0,
    min_rebalance_notional_usd: float = 0.0,
) -> Optional[RebalanceOrder]:
    """Delta order for a netting account, or ``None`` when within the dead-band.

    ``delta = target − current`` (signed lots), snapped to ``volume_step``.
    A non-zero delta is suppressed only when it is below BOTH a meaningful size
    (``min_rebalance_lots``) and notional (``min_rebalance_notional_usd``) — i.e.
    pure noise. A delta that crosses through zero (direction flip) is fine: under
    netting a single order of size ``|delta|`` both closes the old side and opens
    the new one.
    """
    delta = _snap(target_signed_lots - current_signed_lots, symbol.volume_step)
    if abs(delta) <= _TOL:
        return None

    abs_delta = abs(delta)
    delta_notional = abs_delta * price * symbol.trade_contract_size
    below_lots = abs_delta < min_rebalance_lots
    below_notional = delta_notional < min_rebalance_notional_usd
    if below_lots and below_notional:
        return None

    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    return RebalanceOrder(side=side, volume=abs_delta)


@dataclass(frozen=True)
class PlannedOrder:
    """A keyed rebalance order produced by the pure planner."""

    key: str          # caller's instrument key (canonical or InstrumentId str)
    side: OrderSide
    volume: float     # lots, > 0


def plan_rebalance(
    *,
    targets: Mapping[str, float],
    prices: Mapping[str, float],
    current_lots: Mapping[str, float],
    symbols: Mapping[str, MT5SymbolInfo],
    sizing: SizingConfig,
    min_rebalance_lots: float = 0.0,
    min_rebalance_notional_usd: float = 0.0,
) -> list[PlannedOrder]:
    """Pure rebalance decision: targets + market state → netting delta orders.

    The functional core of the strategy (the imperative shell merely gathers the
    inputs from Nautilus and submits the returned orders). For each keyed target:

    * skip when its price or symbol info is missing (cannot size safely);
    * skip when the sizer returns ``None`` (target rounds below ``volume_min`` —
      treated as "no opinion", neither open nor churn an existing position);
    * otherwise emit the single netting delta order from :func:`net_rebalance`
      (``None`` inside the dead-band).

    Keys are opaque (canonical ticker or ``InstrumentId`` string) and pass
    straight through onto the returned :class:`PlannedOrder`.
    """
    orders: list[PlannedOrder] = []
    for key, fraction in targets.items():
        price = prices.get(key)
        symbol = symbols.get(key)
        if price is None or symbol is None:
            continue
        target = target_signed_lots(
            position_fraction=fraction, price=price, symbol=symbol, sizing=sizing
        )
        if target is None:
            continue
        order = net_rebalance(
            target_signed_lots=target,
            current_signed_lots=current_lots.get(key, 0.0),
            symbol=symbol,
            price=price,
            min_rebalance_lots=min_rebalance_lots,
            min_rebalance_notional_usd=min_rebalance_notional_usd,
        )
        if order is not None:
            orders.append(PlannedOrder(key=key, side=order.side, volume=order.volume))
    return orders


__all__ = [
    "PlannedOrder",
    "RebalanceOrder",
    "build_sizing_config",
    "net_rebalance",
    "plan_rebalance",
    "symbol_info_from_nautilus_instrument",
    "target_signed_lots",
]
