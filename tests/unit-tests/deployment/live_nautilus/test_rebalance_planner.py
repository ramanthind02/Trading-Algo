"""Unit tests for the pure rebalance planner deployment.live.runtime.sizing.plan_rebalance."""
from __future__ import annotations

import pytest

from deployment.live.runtime import sizing
from execution.mt5_models import MT5SymbolInfo, OrderSide, TradeMode


def _symbol(name: str = "US500.cash", contract_size: float = 1.0) -> MT5SymbolInfo:
    return MT5SymbolInfo(
        name=name,
        trade_mode=TradeMode.FULL,
        point=0.01,
        digits=2,
        trade_contract_size=contract_size,
        volume_min=0.01,
        volume_max=1000.0,
        volume_step=0.01,
        spread=0,
        visible=True,
    )


def _sizing(equity: float = 100_000.0) -> sizing.SizingConfig:
    return sizing.build_sizing_config(sizing_basis_usd=equity, lot_size_ceiling=1000.0)


def test_plan_opens_from_flat_for_each_target() -> None:
    orders = sizing.plan_rebalance(
        targets={"ES": 0.5, "GC": -0.25},
        prices={"ES": 5000.0, "GC": 2000.0},
        current_lots={"ES": 0.0, "GC": 0.0},
        symbols={"ES": _symbol("US500.cash"), "GC": _symbol("XAUUSD", contract_size=100.0)},
        sizing=_sizing(),
    )
    by_key = {o.key: o for o in orders}
    assert by_key["ES"].side is OrderSide.BUY
    assert by_key["GC"].side is OrderSide.SELL
    assert all(o.volume > 0 for o in orders)


def test_plan_skips_missing_price_or_symbol() -> None:
    orders = sizing.plan_rebalance(
        targets={"ES": 0.5, "NQ": 0.5},
        prices={"ES": 5000.0},                # NQ price missing
        current_lots={},
        symbols={"ES": _symbol(), "NQ": _symbol()},
        sizing=_sizing(),
    )
    assert [o.key for o in orders] == ["ES"]


def test_plan_skips_subthreshold_target_without_churning() -> None:
    # Tiny fraction on a huge-contract instrument rounds below volume_min -> no order.
    orders = sizing.plan_rebalance(
        targets={"ES": 1e-6},
        prices={"ES": 5000.0},
        current_lots={"ES": 0.0},
        symbols={"ES": _symbol(contract_size=100.0)},
        sizing=_sizing(equity=1000.0),
    )
    assert orders == []


def test_plan_emits_noop_free_deltas_only() -> None:
    # current already equals target -> dead-band -> no order emitted at all.
    sym = _symbol(contract_size=1.0)
    # target lots for fraction 0.5 @ price 100, equity 100k, cs 1 = 500 lots
    orders = sizing.plan_rebalance(
        targets={"ES": 0.5},
        prices={"ES": 100.0},
        current_lots={"ES": 500.0},
        symbols={"ES": sym},
        sizing=_sizing(equity=100_000.0),
    )
    assert orders == []


def test_plan_direction_flip_single_netting_order() -> None:
    sym = _symbol(contract_size=1.0)
    # fraction +0.5 @ price 100, equity 100k -> +500 lots target; current -200 -> BUY 700
    orders = sizing.plan_rebalance(
        targets={"ES": 0.5},
        prices={"ES": 100.0},
        current_lots={"ES": -200.0},
        symbols={"ES": sym},
        sizing=_sizing(equity=100_000.0),
    )
    assert len(orders) == 1
    assert orders[0].side is OrderSide.BUY
    assert orders[0].volume == pytest.approx(700.0)
