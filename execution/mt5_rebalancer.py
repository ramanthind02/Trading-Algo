"""
Pure functions that translate ``forecasts_df + live MT5 state`` into
ticket-level rebalance actions for a single account, per the rules
pinned in ``plan.md §7``.

This module **must not** import the ``MetaTrader5`` package — it's pure
logic, fully unit-testable on any platform with synthetic
:class:`MT5SymbolInfo` / :class:`MT5Tick` / :class:`MT5Position`.

Key invariants
--------------
- Same-direction-decrease emits ``CLOSE_TICKET`` actions, never an
  opposite-side ``OPEN_NEW``. This matters because most CFD prop firms
  run MT5 in hedging mode where an opposite-side market order without a
  ``position=<ticket>`` reference opens a new hedge ticket rather than
  reducing the existing one.
- Direction-flip → close all our tickets first, then open fresh on the
  target side (avoids partial residual hedge tickets).
- Dead-band: if the |delta notional| is below the per-account threshold,
  return ``NO_OP`` (no daily spread bleed on cosmetic forecast wiggles).
- Lots are floored at ``volume_min`` (no trade if rounded below) unless
  the account opts into ``force_min_lot_if_signal``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from execution.mt5_models import (
    MT5Position,
    MT5SymbolInfo,
    MT5Tick,
    OrderSide,
    RebalanceAction,
    RebalanceActionKind,
)


_TOL = 1e-9


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizingConfig:
    """Per-account knobs that affect ``target_signed_lots`` and rounding."""

    sizing_basis_usd: float
    lot_size_ceiling: float = 100.0
    force_min_lot_if_signal: bool = False
    min_rebalance_lots: float = 0.0
    min_rebalance_notional_usd: float = 0.0


def _sign(x: float) -> int:
    if x > _TOL:
        return +1
    if x < -_TOL:
        return -1
    return 0


def _round_to_step(value: float, step: float) -> float:
    """Round ``value`` to the nearest multiple of ``step`` (always positive)."""
    if step <= 0:
        raise ValueError(f"volume_step must be > 0, got {step}")
    return round(value / step) * step


def _quantize(value: float, step: float) -> float:
    """Snap to step with finite-precision normalisation for nicer printing."""
    rounded = _round_to_step(value, step)
    # Strip trailing float noise like 0.30000000000000004.
    n = max(0, -int(math.floor(math.log10(step))) if step < 1 else 0)
    return round(rounded, n + 2)


def compute_target_signed_lots(
    *,
    position_fraction: float,
    price: float,
    symbol: MT5SymbolInfo,
    sizing: SizingConfig,
) -> Optional[float]:
    """Translate ``position_fraction`` → signed lot count for one symbol.

    Returns
    -------
    Optional[float]
        ``None`` if the trade should be skipped (e.g. rounded volume is
        below ``volume_min`` and the caller did not opt into
        ``force_min_lot_if_signal``). Otherwise a signed lot count
        (positive = long, negative = short, may be 0.0 to indicate
        "go flat").
    """
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")
    if symbol.trade_contract_size <= 0:
        raise ValueError(
            f"trade_contract_size must be > 0, got {symbol.trade_contract_size}"
        )

    if abs(position_fraction) < _TOL:
        return 0.0  # explicit "go flat"

    target_notional_usd = position_fraction * sizing.sizing_basis_usd
    notional_per_lot_usd = price * symbol.trade_contract_size
    lots_fractional = target_notional_usd / notional_per_lot_usd

    abs_lots = abs(lots_fractional)
    rounded = _quantize(abs_lots, symbol.volume_step)

    if rounded < symbol.volume_min - _TOL:
        if sizing.force_min_lot_if_signal:
            rounded = symbol.volume_min
        else:
            return None

    if rounded > symbol.volume_max:
        rounded = symbol.volume_max
    if rounded > sizing.lot_size_ceiling:
        rounded = sizing.lot_size_ceiling

    # Final snap so ceiling-clamped values still respect volume_step.
    rounded = _quantize(rounded, symbol.volume_step)

    return rounded * _sign(position_fraction)


# ---------------------------------------------------------------------------
# Per-symbol rebalance plan
# ---------------------------------------------------------------------------


def _signed_current_lots(positions: Sequence[MT5Position]) -> float:
    return sum(p.signed_volume for p in positions)


def _our_positions(
    all_positions: Sequence[MT5Position], symbol: str, magic: int
) -> List[MT5Position]:
    return [p for p in all_positions if p.symbol == symbol and p.magic == magic]


def _unmanaged_positions(
    all_positions: Sequence[MT5Position], symbol: str, magic: int
) -> List[MT5Position]:
    return [p for p in all_positions if p.symbol == symbol and p.magic != magic]


def _build_partial_close_actions(
    our_tickets: Sequence[MT5Position],
    reduce_lots: float,
    symbol: str,
    volume_step: float,
) -> List[RebalanceAction]:
    """Emit CLOSE_TICKET actions summing to ``reduce_lots`` (oldest first).

    The last ticket in the chain may be partially closed if the remaining
    reduction is smaller than that ticket's volume.
    """
    if reduce_lots <= _TOL:
        return []

    actions: List[RebalanceAction] = []
    remaining = reduce_lots
    # Oldest first so older, more-likely-stop-loss-hit tickets get freed up.
    ordered = sorted(our_tickets, key=lambda p: p.open_time_utc)
    for ticket in ordered:
        if remaining <= _TOL:
            break
        close_volume = min(ticket.volume, remaining)
        close_volume = _quantize(close_volume, volume_step)
        if close_volume < _TOL:
            continue
        actions.append(
            RebalanceAction(
                kind=RebalanceActionKind.CLOSE_TICKET,
                symbol=symbol,
                side=ticket.side,         # original side; executor sends opposite
                volume=close_volume,
                ticket=ticket.ticket,
                reason=f"reduce_to_target by {close_volume}",
            )
        )
        remaining -= close_volume
    return actions


def plan_symbol_actions(
    *,
    symbol_name: str,
    target_signed_lots: Optional[float],
    all_positions: Sequence[MT5Position],
    symbol: MT5SymbolInfo,
    tick: MT5Tick,
    magic_number: int,
    abort_if_unmanaged_position: bool,
    min_rebalance_lots: float = 0.0,
    min_rebalance_notional_usd: float = 0.0,
) -> List[RebalanceAction]:
    """Compute the rebalance actions for a single symbol on one account.

    See module docstring + ``plan.md §7.1`` for the decision rules.
    """
    unmanaged = _unmanaged_positions(all_positions, symbol_name, magic_number)
    if unmanaged and abort_if_unmanaged_position:
        return [
            RebalanceAction(
                kind=RebalanceActionKind.ABORT_UNMANAGED,
                symbol=symbol_name,
                reason=(
                    f"{len(unmanaged)} non-magic ticket(s) on {symbol_name} "
                    f"(tickets={[p.ticket for p in unmanaged]}); refusing to trade."
                ),
            )
        ]

    if target_signed_lots is None:
        # Skipped by sizer (e.g. rounded below volume_min). Treat as "no
        # opinion": don't open AND don't close anything we already have.
        return [
            RebalanceAction(
                kind=RebalanceActionKind.NO_OP,
                symbol=symbol_name,
                reason="target lots below volume_min; no trade.",
            )
        ]

    our_tickets = _our_positions(all_positions, symbol_name, magic_number)
    current = _signed_current_lots(our_tickets)
    delta = target_signed_lots - current

    # Going flat
    if abs(target_signed_lots) <= _TOL and abs(current) > _TOL:
        return [
            RebalanceAction(
                kind=RebalanceActionKind.CLOSE_TICKET,
                symbol=symbol_name,
                side=p.side,
                volume=p.volume,
                ticket=p.ticket,
                reason="target=0, closing all our tickets",
            )
            for p in sorted(our_tickets, key=lambda t: t.open_time_utc)
        ]

    # Both flat (or current already at target)
    if abs(target_signed_lots) <= _TOL and abs(current) <= _TOL:
        return [
            RebalanceAction(
                kind=RebalanceActionKind.NO_OP,
                symbol=symbol_name,
                reason="target=0 and current=0",
            )
        ]

    # Direction flip
    if _sign(target_signed_lots) != _sign(current) and abs(current) > _TOL:
        close_actions = [
            RebalanceAction(
                kind=RebalanceActionKind.CLOSE_TICKET,
                symbol=symbol_name,
                side=p.side,
                volume=p.volume,
                ticket=p.ticket,
                reason="direction flip; closing existing ticket",
            )
            for p in sorted(our_tickets, key=lambda t: t.open_time_utc)
        ]
        new_side = OrderSide.from_signed(target_signed_lots)
        open_action = RebalanceAction(
            kind=RebalanceActionKind.OPEN_NEW,
            symbol=symbol_name,
            side=new_side,
            volume=abs(target_signed_lots),
            reason=f"direction flip; opening fresh {new_side.value}",
        )
        return [*close_actions, open_action]

    # Same direction (or coming from flat to nonzero)
    same_side = OrderSide.from_signed(target_signed_lots) if abs(target_signed_lots) > _TOL else None
    if abs(current) <= _TOL:
        # Currently flat; opening fresh
        volume = abs(target_signed_lots)
        delta_notional = volume * tick.for_side(same_side) * symbol.trade_contract_size  # type: ignore[arg-type]
        if (
            volume < min_rebalance_lots
            or delta_notional < min_rebalance_notional_usd
        ):
            return [
                RebalanceAction(
                    kind=RebalanceActionKind.NO_OP,
                    symbol=symbol_name,
                    reason="below dead-band; skipping new open",
                )
            ]
        return [
            RebalanceAction(
                kind=RebalanceActionKind.OPEN_NEW,
                symbol=symbol_name,
                side=same_side,
                volume=_quantize(volume, symbol.volume_step),
                reason="fresh open from flat",
            )
        ]

    # Same direction, both nonzero: compare magnitudes
    abs_target = abs(target_signed_lots)
    abs_current = abs(current)
    delta_lots = abs_target - abs_current
    delta_notional = abs(delta_lots) * tick.mid() * symbol.trade_contract_size

    # Exact match (or within float noise): nothing to do.
    if abs(delta_lots) <= _TOL:
        return [
            RebalanceAction(
                kind=RebalanceActionKind.NO_OP,
                symbol=symbol_name,
                reason="current already matches target",
            )
        ]

    if (
        abs(delta_lots) < min_rebalance_lots
        or delta_notional < min_rebalance_notional_usd
    ):
        return [
            RebalanceAction(
                kind=RebalanceActionKind.NO_OP,
                symbol=symbol_name,
                reason=(
                    f"within dead-band: |Δ lots|={abs(delta_lots):.4f}, "
                    f"|Δ notional|=${delta_notional:,.0f}"
                ),
            )
        ]

    if delta_lots > _TOL:
        # Increase exposure: open additional ticket for the delta
        return [
            RebalanceAction(
                kind=RebalanceActionKind.OPEN_NEW,
                symbol=symbol_name,
                side=same_side,
                volume=_quantize(delta_lots, symbol.volume_step),
                reason="increase same-side exposure",
            )
        ]

    # Decrease exposure: partial close on existing tickets (NEVER opposite OPEN_NEW)
    return _build_partial_close_actions(
        our_tickets, reduce_lots=abs(delta_lots), symbol=symbol_name, volume_step=symbol.volume_step
    )


# ---------------------------------------------------------------------------
# Whole-account plan
# ---------------------------------------------------------------------------


def plan_account_actions(
    *,
    targets_signed_lots_by_symbol: Dict[str, Optional[float]],
    all_positions: Sequence[MT5Position],
    symbol_infos: Dict[str, MT5SymbolInfo],
    ticks: Dict[str, MT5Tick],
    magic_number: int,
    abort_if_unmanaged_position: bool,
    min_rebalance_lots: float = 0.0,
    min_rebalance_notional_usd: float = 0.0,
) -> Dict[str, List[RebalanceAction]]:
    """Compose per-symbol plans into a single account-wide plan.

    Symbols missing a tick or symbol_info are skipped with a NO_OP record so
    the caller can surface it.
    """
    out: Dict[str, List[RebalanceAction]] = {}
    for symbol_name, target in targets_signed_lots_by_symbol.items():
        sym = symbol_infos.get(symbol_name)
        tick = ticks.get(symbol_name)
        if sym is None or tick is None:
            out[symbol_name] = [
                RebalanceAction(
                    kind=RebalanceActionKind.NO_OP,
                    symbol=symbol_name,
                    reason="symbol_info or tick missing from broker; cannot plan",
                )
            ]
            continue
        out[symbol_name] = plan_symbol_actions(
            symbol_name=symbol_name,
            target_signed_lots=target,
            all_positions=all_positions,
            symbol=sym,
            tick=tick,
            magic_number=magic_number,
            abort_if_unmanaged_position=abort_if_unmanaged_position,
            min_rebalance_lots=min_rebalance_lots,
            min_rebalance_notional_usd=min_rebalance_notional_usd,
        )
    return out
