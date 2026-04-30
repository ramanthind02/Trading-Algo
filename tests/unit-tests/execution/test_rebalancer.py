"""Unit tests for execution.rebalancer.compute_order_intents."""

from decimal import Decimal

import pytest

from execution.models import ExecutionConfig, OrderSide
from execution.rebalancer import compute_order_intents


def _cfg(
    min_rebalance_shares: Decimal = Decimal("0.01"),
    min_rebalance_notional_usd: float = 5.0,
) -> ExecutionConfig:
    return ExecutionConfig(
        ib_account_id="DU_TEST",
        max_orders_per_run=100,
        min_rebalance_shares=min_rebalance_shares,
        min_rebalance_notional_usd=min_rebalance_notional_usd,
        authorized_telegram_user_ids=[1],
        approval_timeout_seconds=600,
    )


def test_empty_in_empty_out() -> None:
    assert compute_order_intents({}, {}, {}, _cfg()) == []


def test_buy_from_zero() -> None:
    out = compute_order_intents(
        target_shares={"SPY": Decimal("0.5")},
        current_positions={},
        prices={"SPY": 600.0},
        config=_cfg(),
    )
    assert len(out) == 1
    assert out[0].etf == "SPY"
    assert out[0].side == OrderSide.BUY
    assert out[0].shares == Decimal("0.5000")


def test_full_exit() -> None:
    out = compute_order_intents(
        target_shares={"SPY": Decimal("0")},
        current_positions={"SPY": Decimal("0.8")},
        prices={"SPY": 600.0},
        config=_cfg(),
    )
    assert len(out) == 1
    assert out[0].side == OrderSide.SELL
    assert out[0].shares == Decimal("0.8000")


def test_sign_flip_long_to_short_produces_single_order() -> None:
    out = compute_order_intents(
        target_shares={"SPY": Decimal("-0.3")},
        current_positions={"SPY": Decimal("0.5")},
        prices={"SPY": 600.0},
        config=_cfg(),
    )
    assert len(out) == 1
    assert out[0].side == OrderSide.SELL
    assert out[0].shares == Decimal("0.8000")


def test_deadband_shares_suppresses_micro_order() -> None:
    out = compute_order_intents(
        target_shares={"SPY": Decimal("0.505")},
        current_positions={"SPY": Decimal("0.500")},
        prices={"SPY": 600.0},
        config=_cfg(min_rebalance_shares=Decimal("0.01")),
    )
    assert out == []


def test_deadband_notional_suppresses_cheap_order() -> None:
    out = compute_order_intents(
        target_shares={"CHEAP": Decimal("2")},
        current_positions={"CHEAP": Decimal("1")},
        prices={"CHEAP": 2.0},
        config=_cfg(min_rebalance_notional_usd=5.0),
    )
    assert out == []


def test_multiple_tickers_sorted_and_independent() -> None:
    out = compute_order_intents(
        target_shares={"QQQ": Decimal("0.2"), "SPY": Decimal("0.25")},
        current_positions={"GLD": Decimal("0.1")},
        prices={"QQQ": 500.0, "SPY": 600.0, "GLD": 280.0},
        config=_cfg(),
    )
    assert [i.etf for i in out] == ["GLD", "QQQ", "SPY"]
    assert out[0].side == OrderSide.SELL
    assert out[1].side == OrderSide.BUY
    assert out[2].side == OrderSide.BUY


def test_missing_price_raises() -> None:
    with pytest.raises(KeyError, match="SPY"):
        compute_order_intents(
            target_shares={"SPY": Decimal("1")},
            current_positions={},
            prices={},
            config=_cfg(),
        )


def test_zero_delta_skipped_without_price_lookup() -> None:
    out = compute_order_intents(
        target_shares={"SPY": Decimal("0.5")},
        current_positions={"SPY": Decimal("0.5")},
        prices={},
        config=_cfg(),
    )
    assert out == []


def test_decimal_precision_no_float_drift() -> None:
    target = Decimal("0.1") + Decimal("0.2")
    out = compute_order_intents(
        target_shares={"SPY": target},
        current_positions={},
        prices={"SPY": 600.0},
        config=_cfg(),
    )
    assert out[0].shares == Decimal("0.3000")


def test_notional_rounding_quantum_4dp() -> None:
    out = compute_order_intents(
        target_shares={"SPY": Decimal("0.123456789")},
        current_positions={},
        prices={"SPY": 600.0},
        config=_cfg(),
    )
    assert out[0].shares == Decimal("0.1235")


def test_deadband_at_exact_threshold_includes_order() -> None:
    out = compute_order_intents(
        target_shares={"SPY": Decimal("0.01")},
        current_positions={},
        prices={"SPY": 600.0},
        config=_cfg(min_rebalance_shares=Decimal("0.01"), min_rebalance_notional_usd=5.0),
    )
    assert len(out) == 1


def test_only_current_no_target_full_liquidation() -> None:
    out = compute_order_intents(
        target_shares={},
        current_positions={"SPY": Decimal("0.5"), "QQQ": Decimal("0.2")},
        prices={"SPY": 600.0, "QQQ": 500.0},
        config=_cfg(),
    )
    assert len(out) == 2
    assert all(i.side == OrderSide.SELL for i in out)
