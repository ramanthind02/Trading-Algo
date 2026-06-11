"""Universal, position-aware rollover swap-avoidance overlay — pure decision core.

This is the *functional core* (no MT5, no I/O) of the overlay studied in
``research/rollover_cost/``. Every decision is a function of **live quantities MT5
exposes for any symbol** — bid/ask, swap rate, and the current position — so it
generalises across an arbitrary universe with **no per-ticker configuration**.

What the study settled (see ``research/rollover_cost/README.md`` for the full
write-up across Darwinex *and* FTMO)
-------------------------------------------------------------------------------
**Use market orders on both legs.** Passive limit "market-making" was tested six
independent ways (parametric model, static tick-replay, the Nautilus matching
engine, an entry-price+chase replay, an optimistic repricing model, and Carver's
passive→aggressive algo with a full threshold sweep) and **never beats market
orders at the rollover.** The reopen is a uniquely adverse moment for a passive
order: a transient *spike* spread (not a standing one), thin liquidity (you only
fill passively ~30–60 % of the time), and a systematic post-reopen **up-drift** —
so the nights you miss the fill are exactly your best long nights (a measured
+1.4 vs +0.1 bps on miss vs fill nights). Skipping a miss is the *worst* policy;
chasing keeps you in your winners; market-in just sidesteps the whole game.

The governing principle
-----------------------
> **Overlay a leg (flatten before the rollover, re-enter after) only when the
> round-trip half-spread you'd pay is smaller than the swap you'd save tonight.**
> Otherwise hold the leg through and eat the swap. Earn the carry on
> positive-carry legs by holding them.

That one comparison, evaluated on each symbol's *live* spread and swap, reproduces
the per-ticker behaviour from the study (tight index legs overlay; a wide-spread
leg whose round-trip exceeds its swap is held) with **no hard-coded ticker logic**.
On triple-swap nights the swap is ×3, so the threshold rises and more legs overlay
— automatically.

Execution timing (the measured optimum, both brokers)
-----------------------------------------------------
* **Exit:** one **market** order at **T-15 min** before the 00:00-broker rollover.
  Earlier than the close (price drifts *down* into the close, and the spread
  widens), so T-15 both pays a tighter spread and sells before the dip.
* **Entry:** one **market** order at the reopen. *Immediate* for tight-reopen
  names (indices); for wide-reopen names (e.g. gold) wait ``entry_settle_min`` for
  the spike to decay (measured ~+0.7 bps on gold). Never rest a limit; never skip.

In the research backtests this policy took the vault book from a hold-through Sharpe
of **0.33** to **~0.80** on Darwinex (swap avoidance + this execution timing) and
from **0.07 → ~0.70** on FTMO.

.. warning::
   **STATUS: studied / NOT yet integrated into the live trading node.** The live
   runtime (``deployment/live/vault_strategy.py``) currently performs a plain
   session rebalance (``plan_rebalance``) with **no** T-15 flatten / reopen re-entry,
   so this overlay is exercised only by its unit tests and the research P&L lane
   (``research/portfolio/pnl/nautilus_engine.py``, ``ROLLOVER_FLATTEN_REENTER``).
   Wiring it into the live node requires an intraday rollover-time handler (it fires
   at T-15 before, and just after, the 00:00-broker rollover — the current strategy is
   session/decision-clock driven, not bar-driven at the rollover) and **must be
   validated on a demo account before arming live**. Do not treat the Sharpe figures
   above as live-realised until that integration lands.

Position-state awareness (hard requirement)
-------------------------------------------
The overlay is **delta-based**: it never assumes it starts flat. Under the
carry-aware gate we deliberately *hold* positive-carry (and not-worth-overlaying)
legs through the rollover, so at re-entry the order is always
``target - current_actual_position`` (read live), never ``target - 0`` — otherwise
a still-held leg would be doubled.

The three universal scalars (the *only* tunables, shared by all symbols):
``OverlayParams`` — exit lead, entry settle, cross-margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Carry(Enum):
    """Sign of the swap for a held position's direction."""
    NEGATIVE = "negative"   # holding over the rollover PAYS swap → candidate to overlay
    POSITIVE = "positive"   # holding EARNS swap → hold through to collect it
    ZERO = "zero"


class LegAction(Enum):
    """What to do for one leg at a decision point."""
    HOLD = "hold"                  # leave the position on: positive carry, or overlay not worth the spread
    CROSS_MARKET = "cross_market"  # send a MARKET order now (T-15 flatten, or reopen re-establish)
    NONE = "none"                  # no order needed (flat, or delta already zero)


@dataclass(frozen=True)
class OverlayParams:
    """The only tunables — universal across the whole universe (no per-ticker values)."""
    exit_lead_min: int = 15      # send the exit MARKET order this many minutes before the rollover
    entry_settle_min: int = 0    # wait this long after the reopen before the entry MARKET order
    #                              (0 = immediate, optimal for tight-reopen names; raise for
    #                              wide-reopen names — e.g. gold ~5 — to let the spike decay)
    cross_margin: float = 1.0    # overlay a leg iff round-trip half-spread < cross_margin * |swap|
    #                              (≤1 = stricter: only overlay when clearly swap-positive)


# ---------------------------------------------------------------------------
# Live-quantity helpers (universal — bps of notional from bid/ask/swap)
# ---------------------------------------------------------------------------

def half_spread_bps(bid: float, ask: float) -> float:
    """Half the bid/ask spread as bps of mid. Universal cost of crossing once."""
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return float("inf")
    mid = 0.5 * (bid + ask)
    return (ask - bid) / 2.0 / mid * 1e4


def swap_bps_for_position(
    *, position: float, mid: float, point: float, contract_size: float,
    swap_long_points: float, swap_short_points: float, triple: bool,
) -> float:
    """Tonight's swap in bps of notional for the *held* position's direction (signed).

    POINTS-mode (Darwinex / FTMO CFDs): a fixed points charge per lot at the
    rollover. Negative = you pay; positive = you earn. ``triple`` applies the ×3
    weekend/midweek multiplier. Universal — driven entirely by ``symbol_info`` fields.
    """
    if position == 0.0 or mid <= 0.0:
        return 0.0
    pts = swap_long_points if position > 0 else swap_short_points
    mult = 3.0 if triple else 1.0
    return mult * pts * point / mid * 1e4


def carry_of(swap_bps: float) -> Carry:
    if swap_bps < 0.0:
        return Carry.NEGATIVE
    if swap_bps > 0.0:
        return Carry.POSITIVE
    return Carry.ZERO


# ---------------------------------------------------------------------------
# The universal decisions
# ---------------------------------------------------------------------------

def should_overlay_leg(round_trip_half_spread_bps: float, swap_saved_bps: float,
                       params: OverlayParams = OverlayParams()) -> bool:
    """The one rule: overlay a leg iff its round-trip cost is cheaper than the swap saved.

    ``round_trip_half_spread_bps`` is the total spread you'd pay to flatten *and*
    re-enter (exit half-spread + the expected reopen half-spread). ``swap_saved_bps``
    is the magnitude (≥0) of tonight's swap for the leg, ×3 on triple nights. When the
    round-trip exceeds the swap, holding through and eating the swap is cheaper.
    """
    return round_trip_half_spread_bps < params.cross_margin * abs(swap_saved_bps)


def decide_exit_leg(
    *, position: float, swap_bps_position: float, round_trip_half_spread_bps: float,
    params: OverlayParams = OverlayParams(),
) -> LegAction:
    """Pre-rollover leg decision at **T-15** (carry-aware, universal, MARKET-only).

    - Flat: NONE.
    - Positive / zero carry: HOLD — collect the swap (or nothing to avoid).
    - Negative carry: CROSS_MARKET (flatten now) iff the round-trip half-spread is
      cheaper than tonight's swap; otherwise HOLD and eat the swap (flattening a leg
      whose round-trip exceeds its swap loses money). No passive limit — limits lose
      at the rollover (see module docstring).
    """
    if position == 0.0:
        return LegAction.NONE
    if carry_of(swap_bps_position) is not Carry.NEGATIVE:
        return LegAction.HOLD
    return (LegAction.CROSS_MARKET
            if should_overlay_leg(round_trip_half_spread_bps, swap_bps_position, params)
            else LegAction.HOLD)


def reentry_delta(target_position: float, current_position: float) -> float:
    """Universal, position-aware re-entry size: trade the DELTA, never assume flat."""
    return target_position - current_position


def decide_entry_leg(
    *, target_position: float, current_position: float,
) -> tuple[LegAction, float]:
    """Post-rollover re-entry at **reopen + entry_settle_min** (MARKET, delta-based).

    Returns ``(action, delta)`` where ``delta = target - current`` read live. A
    still-held leg (positive-carry or not-overlaid) has ``current == target`` so the
    delta is 0 → NONE (never doubled). A flattened leg has ``current == 0`` so the
    delta is the full target → CROSS_MARKET. We **always** re-establish with a market
    order — never rest a limit, never skip a miss: skipping forgoes the held day, and
    the study shows the missed nights are systematically the best long nights.
    """
    delta = reentry_delta(target_position, current_position)
    if delta == 0.0:
        return LegAction.NONE, 0.0
    return LegAction.CROSS_MARKET, delta
