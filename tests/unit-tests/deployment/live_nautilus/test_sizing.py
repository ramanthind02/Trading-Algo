"""Unit tests for deployment.live.runtime.sizing (synthetic instruments)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from deployment.live.runtime import sizing
from execution.mt5_models import MT5SymbolInfo, OrderSide, TradeMode
from execution.mt5_rebalancer import compute_target_signed_lots


def _symbol(contract_size: float = 100.0, step: float = 0.01) -> MT5SymbolInfo:
    return MT5SymbolInfo(
        name="US500.cash",
        trade_mode=TradeMode.FULL,
        point=0.01,
        digits=2,
        trade_contract_size=contract_size,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=step,
        spread=0,
        visible=True,
    )


def test_symbol_info_from_nautilus_instrument_maps_lot_size_to_contract_size() -> None:
    inst = SimpleNamespace(
        raw_symbol="XAUUSD",
        id="XAUUSD.FTMO",
        lot_size=100.0,        # MT5 trade_contract_size lives here
        size_increment=0.01,   # MT5 volume_step
        min_quantity=0.01,
        max_quantity=50.0,
        price_increment=0.01,
        price_precision=2,
    )
    info = sizing.symbol_info_from_nautilus_instrument(inst)
    assert info.name == "XAUUSD"
    assert info.trade_contract_size == 100.0
    assert info.volume_step == 0.01
    assert info.volume_min == 0.01
    assert info.volume_max == 50.0


def test_target_signed_lots_matches_underlying_sizer() -> None:
    sym = _symbol()
    cfg = sizing.build_sizing_config(sizing_basis_usd=1_000_000.0, lot_size_ceiling=100.0)
    got = sizing.target_signed_lots(position_fraction=0.5, price=100.0, symbol=sym, sizing=cfg)
    pinned = compute_target_signed_lots(
        position_fraction=0.5, price=100.0, symbol=sym, sizing=cfg
    )
    assert got == pinned == 50.0


def test_net_rebalance_open_from_flat() -> None:
    order = sizing.net_rebalance(
        target_signed_lots=0.5, current_signed_lots=0.0,
        symbol=_symbol(), price=100.0,
    )
    assert order == sizing.RebalanceOrder(side=OrderSide.BUY, volume=0.5)


def test_net_rebalance_reduce_same_side() -> None:
    order = sizing.net_rebalance(
        target_signed_lots=0.3, current_signed_lots=0.5,
        symbol=_symbol(), price=100.0,
    )
    assert order is not None
    assert order.side is OrderSide.SELL
    assert order.volume == pytest.approx(0.2)


def test_net_rebalance_direction_flip_single_order() -> None:
    # Netting: a flip from +0.5 to -0.4 is one SELL of 0.9 (closes + opens).
    order = sizing.net_rebalance(
        target_signed_lots=-0.4, current_signed_lots=0.5,
        symbol=_symbol(), price=100.0,
    )
    assert order is not None
    assert order.side is OrderSide.SELL
    assert order.volume == pytest.approx(0.9)


def test_net_rebalance_exact_match_is_noop() -> None:
    assert sizing.net_rebalance(
        target_signed_lots=0.5, current_signed_lots=0.5,
        symbol=_symbol(), price=100.0,
    ) is None


def test_net_rebalance_deadband_suppresses_noise() -> None:
    # Delta 0.001 lots is below both the lot floor AND the notional floor -> None.
    assert sizing.net_rebalance(
        target_signed_lots=0.501, current_signed_lots=0.5,
        symbol=_symbol(), price=100.0,
        min_rebalance_lots=0.05, min_rebalance_notional_usd=1e9,
    ) is None


def test_net_rebalance_meaningful_notional_passes_lot_floor() -> None:
    # Delta below the lot floor but with large notional still trades (OR semantics).
    order = sizing.net_rebalance(
        target_signed_lots=0.5, current_signed_lots=0.46,
        symbol=_symbol(contract_size=100.0), price=100.0,
        min_rebalance_lots=0.05, min_rebalance_notional_usd=100.0,
    )
    assert order is not None  # 0.04 lots * 100 * 100 = $400 notional >= $100
    assert order.side is OrderSide.BUY
    assert order.volume == pytest.approx(0.04)


def test_net_rebalance_snaps_to_step() -> None:
    order = sizing.net_rebalance(
        target_signed_lots=0.333, current_signed_lots=0.0,
        symbol=_symbol(step=0.01), price=100.0,
    )
    assert order is not None
    assert order.volume == pytest.approx(0.33)
