"""Unit tests for the per-broker MT5 config accessors (no terminal contact).

Covers the additions to ``data_platform/providers/mt5/brokers.py``:
timezone, asset-class mapping, per-asset-class sessions / market hours,
execution + prop-firm risk rules, and the schedule check ``is_market_open``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from data_platform.providers.mt5 import brokers
from data_platform.providers.mt5.brokers import (
    AccountMode,
    AssetClass,
    FillingMode,
)

UTC = timezone.utc


# ── config integrity ───────────────────────────────────────────────────────────

def test_known_brokers_include_confirmed_and_stub():
    known = brokers.known_brokers()
    assert {"darwinex", "ftmo", "fundednext"} <= set(known)


def test_validate_config_passes():
    brokers.validate_config()  # raises on any malformed broker block


# ── timezone ────────────────────────────────────────────────────────────────────

def test_broker_timezone_label_and_zoneinfo():
    assert brokers.broker_timezone("ftmo") == "EET"
    assert brokers.broker_tzinfo("ftmo") == ZoneInfo("Europe/Athens")


# ── asset class ─────────────────────────────────────────────────────────────────

def test_asset_class_by_canonical():
    assert brokers.asset_class("ftmo", "NQ") == AssetClass.INDEX
    assert brokers.asset_class("ftmo", "GC") == AssetClass.METAL
    assert brokers.asset_class("ftmo", "CL") == AssetClass.ENERGY
    assert brokers.asset_class("darwinex", "EURUSD") == AssetClass.FX


def test_asset_class_by_native_symbol():
    # US100.cash -> canonical NQ -> index
    assert brokers.asset_class("ftmo", "US100.cash") == AssetClass.INDEX
    # SP500 is Darwinex's native ES symbol
    assert brokers.asset_class("darwinex", "SP500") == AssetClass.INDEX


def test_asset_class_unknown_is_other():
    assert brokers.asset_class("ftmo", "NOPE") == AssetClass.OTHER


# ── sessions / market hours ─────────────────────────────────────────────────────

def test_fx_session_is_weekdays_only():
    s = brokers.session("ftmo", AssetClass.FX)
    assert s.weekdays == frozenset(range(5))   # Mon..Fri
    assert s.tz == "EET"


def test_crypto_session_is_24_7():
    s = brokers.session("ftmo", "crypto")
    assert s.weekdays == frozenset(range(7))   # Mon..Sun
    # Saturday midday is inside a 24/7 crypto session
    assert s.contains(5, 12 * 60) is True


def test_fx_rollover_break_excludes_midnight_minute():
    s = brokers.session("ftmo", AssetClass.FX)
    # 00:01 on a Wednesday falls inside the wrapping 23:58-00:02 rollover break
    assert s.contains(2, 1) is False
    # ...but midday is open
    assert s.contains(2, 12 * 60) is True


def test_index_session_has_premarket_gap():
    s = brokers.session("ftmo", AssetClass.INDEX)
    assert s.contains(2, 0) is False        # 00:00 before the 01:00 open
    assert s.contains(2, 12 * 60) is True   # midday open


def test_session_unknown_asset_class_raises():
    with pytest.raises(KeyError):
        brokers.session("ftmo", AssetClass.OTHER)


# ── is_market_open (pure schedule) ──────────────────────────────────────────────

def test_is_market_open_weekday_vs_weekend_fx():
    wed = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)   # Wednesday
    sat = datetime(2026, 6, 13, 12, 0, tzinfo=UTC)   # Saturday
    assert brokers.is_market_open("ftmo", "EURUSD", at=wed) is True
    assert brokers.is_market_open("ftmo", "EURUSD", at=sat) is False


def test_is_market_open_naive_datetime_treated_as_utc():
    naive_sat = datetime(2026, 6, 13, 12, 0)   # Saturday, no tzinfo
    assert brokers.is_market_open("ftmo", "EURUSD", at=naive_sat) is False


def test_is_market_open_unmapped_symbol_is_closed():
    wed = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    assert brokers.is_market_open("ftmo", "NOPE", at=wed) is False


# ── rules ───────────────────────────────────────────────────────────────────────

def test_execution_rules_ftmo():
    er = brokers.execution_rules("ftmo")
    assert er.filling_mode == FillingMode.IOC
    assert er.account_mode == AccountMode.HEDGING
    assert er.weekend_holding is False
    assert er.magic_number == 510


def test_execution_rules_darwinex_allows_weekend_holding():
    assert brokers.execution_rules("darwinex").weekend_holding is True


def test_risk_rules_ftmo_are_prop_limits():
    rr = brokers.risk_rules("ftmo")
    assert rr is not None
    assert rr.max_daily_loss_pct == 5.0
    assert rr.max_total_loss_pct == 10.0
    assert rr.profit_target_pct == 10.0
    assert rr.news_trading_restricted is True


def test_risk_rules_none_for_live_retail():
    assert brokers.risk_rules("darwinex") is None


def test_rules_bundles_execution_and_risk():
    r = brokers.rules("ftmo")
    assert r.execution.filling_mode == FillingMode.IOC
    assert r.risk is not None and r.risk.max_leverage == 30


def test_terminal_path_points_at_broker_exe():
    p = brokers.terminal_path("ftmo")
    assert p is not None and p.lower().endswith("terminal64.exe")
    assert "ftmo" in p.lower()
    # Darwinex terminal path is distinct (different install) — never the same exe.
    assert brokers.terminal_path("darwinex") != p


# ── frozen / immutable surface ──────────────────────────────────────────────────

def test_dataclasses_are_frozen():
    er = brokers.execution_rules("ftmo")
    with pytest.raises(Exception):
        er.magic_number = 999  # type: ignore[misc]
