"""Unit tests for :mod:`execution.mt5_order_safety`.

All gates are pure functions returning Optional[str]; tests check both the
pass case (returns None) and at least one fail case (returns informative
reason).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from execution.mt5_models import (
    MT5AccountInfo,
    MT5SymbolInfo,
    MT5Tick,
    OrderSide,
    RebalanceAction,
    RebalanceActionKind,
    TradeMode,
)
from execution.mt5_order_safety import (
    check_account_login_matches,
    check_action_compatible_with_trade_mode,
    check_daily_loss_floor,
    check_margin_budget,
    check_symbol_tradeable,
    check_trade_allowed,
    run_preflight,
)


UTC = timezone.utc
T0 = datetime(2026, 1, 5, 22, 0, tzinfo=UTC)


def _acct(login=12345678, server="FTMO-Demo", equity=100_000.0, margin_free=80_000.0,
          balance=100_000.0, trade_allowed=True, trade_expert=True):
    return MT5AccountInfo(
        login=login,
        server=server,
        currency="USD",
        balance=balance,
        equity=equity,
        margin=balance - margin_free,
        margin_free=margin_free,
        margin_level=200.0,
        leverage=100,
        trade_allowed=trade_allowed,
        trade_expert=trade_expert,
    )


def _sym(name="US100.cash", trade_mode=TradeMode.FULL, spread=10, visible=True):
    return MT5SymbolInfo(
        name=name,
        trade_mode=trade_mode,
        point=0.01,
        digits=2,
        trade_contract_size=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread=spread,
        visible=visible,
    )


def _tick(symbol="US100.cash", bid=18000.0, ask=18001.0, t=T0):
    return MT5Tick(symbol=symbol, bid=bid, ask=ask, time_utc=t)


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def test_login_match_passes_on_exact_match() -> None:
    assert check_account_login_matches(_acct(), expected_login=12345678, expected_server="FTMO-Demo") is None


def test_login_match_fails_on_wrong_login() -> None:
    msg = check_account_login_matches(_acct(login=99), expected_login=12345678, expected_server="FTMO-Demo")
    assert msg is not None
    assert "12345678" in msg
    assert "99" in msg


def test_login_match_fails_on_wrong_server() -> None:
    msg = check_account_login_matches(_acct(server="FTMO-Live"), expected_login=12345678, expected_server="FTMO-Demo")
    assert msg is not None
    assert "FTMO-Live" in msg
    assert "FTMO-Demo" in msg


def test_trade_allowed_gate() -> None:
    assert check_trade_allowed(_acct()) is None
    assert check_trade_allowed(_acct(trade_allowed=False)) is not None
    assert "trade_allowed" in check_trade_allowed(_acct(trade_allowed=False))  # type: ignore


def test_symbol_tradeable_invisible_fails() -> None:
    msg = check_symbol_tradeable(_sym(visible=False))
    assert msg is not None
    assert "visible" in msg


def test_symbol_tradeable_disabled_mode_fails() -> None:
    assert check_symbol_tradeable(_sym(trade_mode=TradeMode.DISABLED)) is not None


def test_action_compatible_long_only_blocks_sell_open() -> None:
    sym = _sym(trade_mode=TradeMode.LONG_ONLY)
    sell = RebalanceAction(kind=RebalanceActionKind.OPEN_NEW, symbol=sym.name, side=OrderSide.SELL, volume=0.1)
    buy = RebalanceAction(kind=RebalanceActionKind.OPEN_NEW, symbol=sym.name, side=OrderSide.BUY, volume=0.1)
    assert check_action_compatible_with_trade_mode(sell, sym) is not None
    assert check_action_compatible_with_trade_mode(buy, sym) is None


def test_action_compatible_close_allowed_in_close_only() -> None:
    sym = _sym(trade_mode=TradeMode.CLOSE_ONLY)
    close = RebalanceAction(
        kind=RebalanceActionKind.CLOSE_TICKET, symbol=sym.name, side=OrderSide.BUY, volume=0.1, ticket=1,
    )
    assert check_action_compatible_with_trade_mode(close, sym) is None
    # But cannot OPEN_NEW in CLOSE_ONLY mode
    open_a = RebalanceAction(kind=RebalanceActionKind.OPEN_NEW, symbol=sym.name, side=OrderSide.BUY, volume=0.1)
    assert check_action_compatible_with_trade_mode(open_a, sym) is not None


def test_spread_is_no_longer_gated() -> None:
    """We removed the spread gate entirely; verify the symbol w/ huge spread
    still passes the remaining symbol checks (visible + tradeable)."""
    assert check_symbol_tradeable(_sym(spread=99_999)) is None


def test_margin_budget_within_cap_passes() -> None:
    assert check_margin_budget(requested_margin_usd=10_000, account_info=_acct(margin_free=80_000)) is None


def test_margin_budget_over_cap_fails() -> None:
    msg = check_margin_budget(
        requested_margin_usd=80_000, account_info=_acct(margin_free=80_000), max_margin_usage_pct=0.5,
    )
    assert msg is not None
    assert "80,000" in msg


def test_daily_loss_floor_within_limit_passes() -> None:
    # equity 99_000 vs SOD 100_000 → 1% drawdown; cap 5% → OK
    assert check_daily_loss_floor(
        account_info=_acct(equity=99_000),
        sod_balance_usd=100_000.0,
        max_daily_loss_pct=0.05,
    ) is None


def test_daily_loss_floor_over_limit_fails() -> None:
    msg = check_daily_loss_floor(
        account_info=_acct(equity=94_000),  # 6% drawdown
        sod_balance_usd=100_000.0,
        max_daily_loss_pct=0.05,
    )
    assert msg is not None
    assert "6.00%" in msg


def test_daily_loss_floor_disabled_when_either_none() -> None:
    assert check_daily_loss_floor(
        account_info=_acct(equity=50_000), sod_balance_usd=None, max_daily_loss_pct=0.05,
    ) is None
    assert check_daily_loss_floor(
        account_info=_acct(equity=50_000), sod_balance_usd=100_000.0, max_daily_loss_pct=None,
    ) is None


# ---------------------------------------------------------------------------
# Aggregate run_preflight
# ---------------------------------------------------------------------------


def test_run_preflight_clean_pass() -> None:
    failures = run_preflight(
        account_info=_acct(),
        expected_login=12345678,
        expected_server="FTMO-Demo",
        actions_by_symbol={
            "US100.cash": [
                RebalanceAction(kind=RebalanceActionKind.OPEN_NEW, symbol="US100.cash", side=OrderSide.BUY, volume=0.1),
            ]
        },
        symbol_infos={"US100.cash": _sym()},
        ticks={"US100.cash": _tick()},
        requested_margin_usd=1_000.0,
        max_margin_usage_pct=0.95,
    )
    assert failures == []


def test_run_preflight_collects_multiple_failures() -> None:
    failures = run_preflight(
        account_info=_acct(login=99, server="WRONG", trade_allowed=False),  # 2 account-level fails
        expected_login=12345678,
        expected_server="FTMO-Demo",
        actions_by_symbol={
            "US100.cash": [
                RebalanceAction(kind=RebalanceActionKind.OPEN_NEW, symbol="US100.cash", side=OrderSide.SELL, volume=0.1),
            ]
        },
        symbol_infos={"US100.cash": _sym(trade_mode=TradeMode.LONG_ONLY, spread=500, visible=True)},
        ticks={"US100.cash": _tick(t=T0)},
        requested_margin_usd=1_000.0,
    )
    # Should have collected: login mismatch (short-circuits server) +
    # trade_allowed + LONG_ONLY blocks SELL open = at least 3
    assert len(failures) >= 3
    # And the LONG_ONLY action incompatibility must be captured.
    assert any("LONG_ONLY" in f for f in failures)


def test_run_preflight_skips_symbol_gates_for_no_op_only_actions() -> None:
    """If a symbol has only NO_OP actions, trade-mode is irrelevant — gate
    must not flag it."""
    failures = run_preflight(
        account_info=_acct(),
        expected_login=12345678,
        expected_server="FTMO-Demo",
        actions_by_symbol={
            "US100.cash": [
                RebalanceAction(kind=RebalanceActionKind.NO_OP, symbol="US100.cash", reason="within dead-band"),
            ]
        },
        # Huge spread; previously would fail spread gate (now removed).
        symbol_infos={"US100.cash": _sym(spread=99999)},
        ticks={"US100.cash": _tick(t=T0)},
        requested_margin_usd=0.0,
    )
    assert failures == []
