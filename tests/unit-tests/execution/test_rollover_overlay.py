"""Unit tests for the universal, position-aware rollover overlay decision core.

These prove the design guarantees of the **market-order** policy the study settled on:
  1. UNIVERSAL — one ``should_overlay_leg`` rule reproduces the per-ticker behaviour
     (a leg whose round-trip spread < its swap overlays; a wide-round-trip leg is held)
     from *live inputs only*, with no per-ticker constants.
  2. CARRY-AWARE — positive-carry legs are held to collect the swap; only negative-carry
     legs that are swap-positive to round-trip are flattened.
  3. POSITION-AWARE — re-entry trades ``target - current`` (delta), never ``target - 0``,
     so a still-held (held-through) leg is not doubled.
  4. MARKET-ONLY — exit and entry are single market orders (no passive limits, no chase,
     no skip); limits lose at the rollover (see rollover_overlay module docstring).
"""
from __future__ import annotations

import pytest

from execution.rollover_overlay import (
    Carry, LegAction, OverlayParams,
    half_spread_bps, swap_bps_for_position, carry_of, should_overlay_leg,
    decide_exit_leg, reentry_delta, decide_entry_leg,
)

P = OverlayParams()


# --- live-quantity helpers -------------------------------------------------

def test_half_spread_bps():
    # mid 100, spread 0.10 → half 0.05 → 5 bps
    assert half_spread_bps(99.95, 100.05) == pytest.approx(5.0, rel=1e-3)
    assert half_spread_bps(0.0, 1.0) == float("inf")      # bad quote guarded


def test_swap_sign_and_triple():
    base = dict(mid=100.0, point=0.1, contract_size=10.0,
                swap_long_points=-10.0, swap_short_points=5.0)
    long_bps = swap_bps_for_position(position=1.0, triple=False, **base)
    short_bps = swap_bps_for_position(position=-1.0, triple=False, **base)
    assert long_bps < 0 and short_bps > 0                 # longs pay, shorts earn
    assert carry_of(long_bps) is Carry.NEGATIVE
    assert carry_of(short_bps) is Carry.POSITIVE
    # triple multiplies magnitude by 3
    assert swap_bps_for_position(position=1.0, triple=True, **base) == pytest.approx(3 * long_bps)
    assert swap_bps_for_position(position=0.0, triple=False, **base) == 0.0


# --- THE universal rule (round-trip spread vs swap) ------------------------

def test_should_overlay_is_a_pure_live_comparison():
    assert should_overlay_leg(0.5, 1.6) is True           # cheap round-trip < swap → overlay
    assert should_overlay_leg(3.1, 1.5) is False          # wide round-trip > swap → hold/eat swap
    # same rule overlays a wide-spread leg on a triple night (swap ×3) with NO ticker logic
    assert should_overlay_leg(3.1, 1.5 * 3) is True


def test_one_rule_generalizes_across_assets_no_per_ticker_constants():
    """Index-like (tight round-trip) overlays; a wide leg eats swap — same call, live inputs."""
    index_like = dict(round_trip_half_spread_bps=0.5, swap_saved_bps=1.6)
    wide_like = dict(round_trip_half_spread_bps=3.1, swap_saved_bps=1.5)
    assert should_overlay_leg(**index_like) is True
    assert should_overlay_leg(**wide_like) is False


# --- exit leg (carry-aware, MARKET at T-15) --------------------------------

def test_exit_positive_carry_holds_to_collect_swap():
    a = decide_exit_leg(position=-1.0, swap_bps_position=+0.6, round_trip_half_spread_bps=0.5)
    assert a is LegAction.HOLD


def test_exit_negative_carry_crosses_or_eats():
    common = dict(position=1.0, swap_bps_position=-1.6)
    # cheap round-trip → market-out now (cheaper than the swap)
    assert decide_exit_leg(**common, round_trip_half_spread_bps=0.5) is LegAction.CROSS_MARKET
    # WIDE round-trip → hold / eat the swap (overlaying costs more than it saves)
    assert decide_exit_leg(**common, round_trip_half_spread_bps=3.0) is LegAction.HOLD


def test_exit_flat_is_noop():
    assert decide_exit_leg(position=0.0, swap_bps_position=-1.6,
                           round_trip_half_spread_bps=0.5) is LegAction.NONE


# --- re-entry is DELTA-based (never assume flat), MARKET-only --------------

def test_reentry_delta_off_held_position_not_zero():
    # held a -2.0 short through the rollover; new target is -3.0 short
    assert reentry_delta(target_position=-3.0, current_position=-2.0) == pytest.approx(-1.0)
    # flattened leg (current 0) → full target
    assert reentry_delta(target_position=4.0, current_position=0.0) == pytest.approx(4.0)


def test_entry_uses_delta_and_markets_it():
    # held -2.0, target -2.0 → delta 0 → NONE (no double!)
    act, d = decide_entry_leg(target_position=-2.0, current_position=-2.0)
    assert act is LegAction.NONE and d == 0.0
    # flattened (current 0), target +4 → market the full +4
    act, d = decide_entry_leg(target_position=4.0, current_position=0.0)
    assert act is LegAction.CROSS_MARKET and d == pytest.approx(4.0)
    # partially held (current +1), target +4 → market the +3 delta
    act, d = decide_entry_leg(target_position=4.0, current_position=1.0)
    assert act is LegAction.CROSS_MARKET and d == pytest.approx(3.0)


def test_overlay_params_defaults_match_measured_optimum():
    # exit at T-15, immediate entry by default, strict swap-positive gate
    assert P.exit_lead_min == 15
    assert P.entry_settle_min == 0
    assert P.cross_margin == 1.0
