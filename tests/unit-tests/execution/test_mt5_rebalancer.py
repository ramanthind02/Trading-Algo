"""Unit tests for :mod:`execution.mt5_rebalancer`.

These pin the §7.1 / §7.3 spec in ``plan.md`` so a regression of the
critical CFD-prop rebalance rules (especially the hedging-mode close
semantics) fails loudly.

All fixtures are synthetic — no MT5 dependency.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from execution.mt5_models import (
    MT5Position,
    MT5SymbolInfo,
    MT5Tick,
    OrderSide,
    RebalanceAction,
    RebalanceActionKind,
    TradeMode,
)
from execution.mt5_rebalancer import (
    SizingConfig,
    compute_target_signed_lots,
    plan_account_actions,
    plan_symbol_actions,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

UTC = timezone.utc
T0 = datetime(2026, 1, 5, 21, 30, tzinfo=UTC)


def _sym(name="US100.cash", contract=1.0, vol_min=0.01, vol_max=100.0, vol_step=0.01,
         trade_mode=TradeMode.FULL, spread=10, visible=True, digits=2):
    return MT5SymbolInfo(
        name=name,
        trade_mode=trade_mode,
        point=10 ** -digits,
        digits=digits,
        trade_contract_size=contract,
        volume_min=vol_min,
        volume_max=vol_max,
        volume_step=vol_step,
        spread=spread,
        visible=visible,
    )


def _tick(symbol="US100.cash", bid=18000.0, ask=18001.0, t=T0):
    return MT5Tick(symbol=symbol, bid=bid, ask=ask, time_utc=t)


def _pos(ticket, symbol, side: OrderSide, volume, magic=90420, when: Optional[datetime] = None):
    return MT5Position(
        ticket=ticket,
        symbol=symbol,
        side=side,
        volume=volume,
        open_price=18000.0,
        magic=magic,
        open_time_utc=when or T0,
    )


# ---------------------------------------------------------------------------
# Sizing — pin the §7.3 worked examples to the cent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,price,contract,equity,target_pct,expected",
    [
        # XAUUSD: 100 oz/lot. tiny account → raw 0.00833 → rounds to 0.01 (snap up).
        # (Note: plan.md §7.3 originally labelled this "skip"; in practice
        # round-to-nearest snaps it to volume_min so we do trade.)
        ("XAUUSD_50k_+5pct_snaps_to_min", 3000.0, 100.0, 50_000.0, +0.05, +0.01),
        # XAUUSD: 100k account → 0.0333 raw → snaps to 0.03
        ("XAUUSD_100k_+10pct", 3000.0, 100.0, 100_000.0, +0.10, +0.03),
        # XAGUSD: 5000 oz/lot
        ("XAGUSD_100k_+10pct", 30.0, 5000.0, 100_000.0, +0.10, +0.07),
        # US500: 1 unit/lot (raw 2.0)
        ("US500_100k_-10pct", 5000.0, 1.0, 100_000.0, -0.10, -2.00),
        # US100: 1 unit/lot (raw 0.833)
        ("US100_100k_+15pct", 18_000.0, 1.0, 100_000.0, +0.15, +0.83),
        # Tiny enough that round-to-nearest produces 0.00 → genuinely skipped.
        ("XAUUSD_50k_+0.1pct_skip", 3000.0, 100.0, 50_000.0, +0.001, None),
    ],
)
def test_sizing_table_pinned(label, price, contract, equity, target_pct, expected):
    """If any of these break, plan §7.3 is out of sync with code."""
    sym = _sym(contract=contract)
    sizing = SizingConfig(sizing_basis_usd=equity)
    out = compute_target_signed_lots(
        position_fraction=target_pct,
        price=price,
        symbol=sym,
        sizing=sizing,
    )
    if expected is None:
        assert out is None, f"{label}: expected skip, got {out}"
    else:
        assert out is not None, f"{label}: unexpected skip"
        assert out == pytest.approx(expected, abs=1e-9), f"{label}: got {out}"


def test_sizing_zero_target_returns_zero_not_none() -> None:
    """A zero forecast means 'go flat' (close all), not 'skip'."""
    out = compute_target_signed_lots(
        position_fraction=0.0,
        price=18000.0,
        symbol=_sym(),
        sizing=SizingConfig(sizing_basis_usd=100_000.0),
    )
    assert out == 0.0


def test_sizing_force_min_lot_if_signal_overrides_skip() -> None:
    """``force_min_lot_if_signal=True`` floors-up tiny computed volumes."""
    out = compute_target_signed_lots(
        position_fraction=+0.001,           # 0.0001 lots raw on a 100k account
        price=18000.0,
        symbol=_sym(),
        sizing=SizingConfig(sizing_basis_usd=100_000.0, force_min_lot_if_signal=True),
    )
    assert out == pytest.approx(0.01)


def test_sizing_lot_ceiling_clamp() -> None:
    """``lot_size_ceiling`` caps the rounded volume."""
    out = compute_target_signed_lots(
        position_fraction=+1.0,             # 100% of equity → huge raw lots
        price=10.0,
        symbol=_sym(contract=1.0, vol_max=1000.0),
        sizing=SizingConfig(sizing_basis_usd=1_000_000.0, lot_size_ceiling=5.0),
    )
    assert out == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Rebalance rules — pin each branch in §7.1
# ---------------------------------------------------------------------------


def test_no_op_when_both_flat() -> None:
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=0.0,
        all_positions=[],
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    assert len(actions) == 1
    assert actions[0].kind is RebalanceActionKind.NO_OP


def test_open_from_flat_emits_single_open() -> None:
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=+0.50,
        all_positions=[],
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    assert len(actions) == 1
    a = actions[0]
    assert a.kind is RebalanceActionKind.OPEN_NEW
    assert a.side is OrderSide.BUY
    assert a.volume == pytest.approx(0.50)


def test_short_from_flat_uses_sell_side() -> None:
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=-0.25,
        all_positions=[],
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    assert actions[0].side is OrderSide.SELL
    assert actions[0].volume == pytest.approx(0.25)


def test_go_flat_emits_close_for_every_our_ticket() -> None:
    positions = [
        _pos(1, "US100.cash", OrderSide.BUY, 0.10, when=T0),
        _pos(2, "US100.cash", OrderSide.BUY, 0.20, when=T0 + timedelta(minutes=5)),
    ]
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=0.0,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    assert all(a.kind is RebalanceActionKind.CLOSE_TICKET for a in actions)
    assert {a.ticket for a in actions} == {1, 2}
    # Each close carries the ticket's ORIGINAL side so the executor knows
    # to send the opposite side with position=<ticket>.
    assert all(a.side is OrderSide.BUY for a in actions)


def test_direction_flip_closes_all_then_opens_new() -> None:
    """Going from net long → net short closes existing longs and opens a
    fresh short. CRITICAL: opposite-side OPEN_NEW happens only AFTER all
    close orders, not as a single hedging order."""
    positions = [
        _pos(1, "US100.cash", OrderSide.BUY, 0.30, when=T0),
        _pos(2, "US100.cash", OrderSide.BUY, 0.20, when=T0 + timedelta(minutes=3)),
    ]
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=-0.40,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    closes = [a for a in actions if a.kind is RebalanceActionKind.CLOSE_TICKET]
    opens = [a for a in actions if a.kind is RebalanceActionKind.OPEN_NEW]
    assert len(closes) == 2
    assert len(opens) == 1
    assert opens[0].side is OrderSide.SELL
    assert opens[0].volume == pytest.approx(0.40)
    # And the action ordering: close first, then open.
    assert all(actions.index(c) < actions.index(opens[0]) for c in closes)


def test_same_direction_increase_emits_single_open_for_delta() -> None:
    positions = [_pos(1, "US100.cash", OrderSide.BUY, 0.10)]
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=+0.30,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    assert len(actions) == 1
    a = actions[0]
    assert a.kind is RebalanceActionKind.OPEN_NEW
    assert a.side is OrderSide.BUY
    assert a.volume == pytest.approx(0.20)


def test_same_direction_decrease_emits_close_not_opposite_open_regression() -> None:
    """REGRESSION TEST for the hedging-mode bug.

    In MT5 hedging mode (default for most CFD prop firms), placing an
    opposite-side market order WITHOUT ``position=<ticket>`` opens a new
    hedge ticket. The rebalancer MUST instead emit ``CLOSE_TICKET``
    actions, each carrying the original ticket id so the executor can
    set ``position=<ticket>`` on the closing deal.
    """
    positions = [
        _pos(1, "US100.cash", OrderSide.BUY, 0.30, when=T0),
        _pos(2, "US100.cash", OrderSide.BUY, 0.20, when=T0 + timedelta(minutes=5)),
    ]
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=+0.30,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    assert all(a.kind is RebalanceActionKind.CLOSE_TICKET for a in actions), (
        "REGRESSION: same-direction decrease must NOT emit OPEN_NEW on the "
        "opposite side; that would hedge in hedging-mode accounts."
    )
    # Must close 0.20 lots total, taken oldest-first.
    total_close = sum(a.volume or 0 for a in actions)
    assert total_close == pytest.approx(0.20)
    # First close hits ticket #1 (older)
    assert actions[0].ticket == 1
    # Each close carries the original side so the executor sends the
    # OPPOSITE side with position=<ticket> set.
    assert all(a.side is OrderSide.BUY for a in actions)


def test_partial_close_walks_tickets_oldest_first() -> None:
    positions = [
        _pos(1, "US100.cash", OrderSide.BUY, 0.10, when=T0),
        _pos(2, "US100.cash", OrderSide.BUY, 0.10, when=T0 + timedelta(minutes=1)),
        _pos(3, "US100.cash", OrderSide.BUY, 0.10, when=T0 + timedelta(minutes=2)),
    ]
    # Reduce from 0.30 to 0.10 (need to close 0.20)
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=+0.10,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    # First two tickets fully closed; ticket #3 untouched.
    closed_tickets = [a.ticket for a in actions]
    assert closed_tickets == [1, 2]
    total = sum(a.volume for a in actions)
    assert total == pytest.approx(0.20)


def test_partial_close_handles_last_ticket_partial() -> None:
    positions = [
        _pos(1, "US100.cash", OrderSide.BUY, 0.10, when=T0),
        _pos(2, "US100.cash", OrderSide.BUY, 0.10, when=T0 + timedelta(minutes=1)),
    ]
    # Reduce from 0.20 to 0.15 (close exactly 0.05).
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=+0.15,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    # Should be a single partial close of 0.05 on ticket #1.
    assert len(actions) == 1
    assert actions[0].ticket == 1
    assert actions[0].volume == pytest.approx(0.05)


def test_dead_band_skip_when_delta_too_small() -> None:
    positions = [_pos(1, "US100.cash", OrderSide.BUY, 0.10)]
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=+0.11,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(bid=18000.0, ask=18001.0),
        magic_number=90420,
        abort_if_unmanaged_position=True,
        # delta_notional ≈ 0.01 * 18000 * 1 = $180; dead-band $500 skips.
        min_rebalance_notional_usd=500.0,
    )
    assert len(actions) == 1
    assert actions[0].kind is RebalanceActionKind.NO_OP
    assert "dead-band" in actions[0].reason


def test_unmanaged_position_aborts_when_gate_on() -> None:
    positions = [
        _pos(1, "US100.cash", OrderSide.BUY, 0.10, magic=12345),  # someone else's
    ]
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=+0.50,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    assert len(actions) == 1
    assert actions[0].kind is RebalanceActionKind.ABORT_UNMANAGED
    assert "12345" not in actions[0].reason   # we report tickets not magic
    assert "1" in actions[0].reason            # ticket #1 mentioned


def test_unmanaged_position_proceeds_when_gate_off() -> None:
    positions = [_pos(99, "US100.cash", OrderSide.BUY, 0.10, magic=12345)]
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=+0.50,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=False,
    )
    # Unmanaged ticket #99 ignored; rebalancer opens 0.50 fresh.
    assert len(actions) == 1
    assert actions[0].kind is RebalanceActionKind.OPEN_NEW
    assert actions[0].volume == pytest.approx(0.50)


def test_target_none_emits_no_op_does_not_close_existing() -> None:
    """If the sizer returned None (below volume_min, no force), the
    rebalancer should not touch existing positions either way."""
    positions = [_pos(1, "US100.cash", OrderSide.BUY, 0.10)]
    actions = plan_symbol_actions(
        symbol_name="US100.cash",
        target_signed_lots=None,
        all_positions=positions,
        symbol=_sym(),
        tick=_tick(),
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    assert len(actions) == 1
    assert actions[0].kind is RebalanceActionKind.NO_OP


# ---------------------------------------------------------------------------
# Whole-account plan
# ---------------------------------------------------------------------------


def test_plan_account_actions_handles_missing_symbol_info_gracefully() -> None:
    out = plan_account_actions(
        targets_signed_lots_by_symbol={"US100.cash": +0.10},
        all_positions=[],
        symbol_infos={},                  # broker did not return info
        ticks={},
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    assert out["US100.cash"][0].kind is RebalanceActionKind.NO_OP
    assert "missing" in out["US100.cash"][0].reason.lower()


def test_plan_account_actions_composes_multiple_symbols_independently() -> None:
    positions = [_pos(1, "US100.cash", OrderSide.BUY, 0.20)]
    out = plan_account_actions(
        targets_signed_lots_by_symbol={
            "US100.cash": +0.20,  # current = target → no-op (within dead-band 0)
            "US500.cash": -0.10,  # fresh short
        },
        all_positions=positions,
        symbol_infos={
            "US100.cash": _sym(name="US100.cash"),
            "US500.cash": _sym(name="US500.cash"),
        },
        ticks={
            "US100.cash": _tick(symbol="US100.cash"),
            "US500.cash": _tick(symbol="US500.cash", bid=5000.0, ask=5001.0),
        },
        magic_number=90420,
        abort_if_unmanaged_position=True,
    )
    us100_actions = out["US100.cash"]
    us500_actions = out["US500.cash"]
    assert len(us100_actions) == 1 and us100_actions[0].kind is RebalanceActionKind.NO_OP
    assert len(us500_actions) == 1 and us500_actions[0].kind is RebalanceActionKind.OPEN_NEW
    assert us500_actions[0].side is OrderSide.SELL
    assert us500_actions[0].volume == pytest.approx(0.10)
