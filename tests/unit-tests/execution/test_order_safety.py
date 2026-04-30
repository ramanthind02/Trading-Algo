"""Unit tests for execution.order_safety gates."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from execution.models import ExecutionConfig, OrderIntent, OrderSide
from execution.order_safety import (
    PreflightContext,
    SafetyViolation,
    check_account_match,
    check_live_port_requires_flag,
    check_lock_file,
    check_market_hours,
    check_max_orders_per_run,
    run_all_preflight_checks,
)

_ET = ZoneInfo("America/New_York")


def _config(**overrides) -> ExecutionConfig:
    defaults = dict(
        ib_account_id="DU1234567",
        max_orders_per_run=10,
        min_rebalance_shares=Decimal("0.01"),
        min_rebalance_notional_usd=5.0,
        authorized_telegram_user_ids=[111],
        approval_timeout_seconds=600,
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def _ctx(tmp_path, **overrides) -> PreflightContext:
    defaults = dict(
        ib_port=7497,
        live_flag=False,
        managed_accounts=["DU1234567"],
        intents=[_intent("SPY", OrderSide.BUY, "0.25", 600.0)],
        now_et=datetime(2026, 4, 21, 15, 45, tzinfo=_ET),
        lock_file=tmp_path / "lock.json",
    )
    defaults.update(overrides)
    return PreflightContext(**defaults)


def _intent(etf: str, side: OrderSide, shares: str, price: float) -> OrderIntent:
    s = Decimal(shares)
    return OrderIntent(etf=etf, side=side, shares=s, est_price=price, est_notional=float(s) * price)


# --- lock file ---

def test_lock_file_missing_passes(tmp_path) -> None:
    check_lock_file(_ctx(tmp_path))


def test_lock_file_present_raises(tmp_path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    with pytest.raises(SafetyViolation, match="Lock file"):
        check_lock_file(_ctx(tmp_path, lock_file=lock))


def test_lock_file_allow_rerun_overrides(tmp_path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    check_lock_file(_ctx(tmp_path, lock_file=lock, allow_rerun=True))


# --- live port ---

def test_paper_port_passes(tmp_path) -> None:
    check_live_port_requires_flag(_ctx(tmp_path, ib_port=7497))


def test_live_port_without_flag_raises(tmp_path) -> None:
    with pytest.raises(SafetyViolation, match="LIVE"):
        check_live_port_requires_flag(_ctx(tmp_path, ib_port=7496))


def test_live_port_with_flag_passes(tmp_path) -> None:
    check_live_port_requires_flag(_ctx(tmp_path, ib_port=7496, live_flag=True))


# --- account ---

def test_account_match_pass(tmp_path) -> None:
    check_account_match(_ctx(tmp_path), _config())


def test_account_placeholder_raises(tmp_path) -> None:
    with pytest.raises(SafetyViolation, match="not set"):
        check_account_match(_ctx(tmp_path), _config(ib_account_id="REPLACE_ME"))


def test_account_mismatch_raises(tmp_path) -> None:
    with pytest.raises(SafetyViolation, match="not in IB"):
        check_account_match(_ctx(tmp_path, managed_accounts=["DU9999999"]), _config())


# --- market hours ---

def test_market_hours_inside_passes(tmp_path) -> None:
    check_market_hours(_ctx(tmp_path, now_et=datetime(2026, 4, 21, 15, 45, tzinfo=_ET)), _config())


def test_market_hours_before_open_raises(tmp_path) -> None:
    with pytest.raises(SafetyViolation, match="outside trading"):
        check_market_hours(_ctx(tmp_path, now_et=datetime(2026, 4, 21, 9, 0, tzinfo=_ET)), _config())


def test_market_hours_after_close_raises(tmp_path) -> None:
    with pytest.raises(SafetyViolation, match="outside trading"):
        check_market_hours(_ctx(tmp_path, now_et=datetime(2026, 4, 21, 16, 30, tzinfo=_ET)), _config())


# --- max orders ---

def test_max_orders_under_limit_passes(tmp_path) -> None:
    intents = [_intent(f"E{i}", OrderSide.BUY, "0.1", 100.0) for i in range(5)]
    check_max_orders_per_run(_ctx(tmp_path, intents=intents), _config(max_orders_per_run=10))


def test_max_orders_over_limit_raises(tmp_path) -> None:
    intents = [_intent(f"E{i}", OrderSide.BUY, "0.1", 100.0) for i in range(15)]
    with pytest.raises(SafetyViolation, match="exceeds max_orders_per_run"):
        check_max_orders_per_run(_ctx(tmp_path, intents=intents), _config(max_orders_per_run=10))


# --- orchestrator happy path ---

def test_run_all_preflight_checks_happy_path(tmp_path) -> None:
    run_all_preflight_checks(_ctx(tmp_path), _config())
