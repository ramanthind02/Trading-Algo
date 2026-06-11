"""Unit tests for the live-monitoring API (frontend/api/live.py).

The live node publishes a JSON snapshot; this API only reads it and writes command
files. Tests redirect the per-broker live_state dir to a tmp path and assert the
derived prop-firm gauges, the snapshot reads, and the flatten safety gates
(confirm token, node-online, demo-only).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from deployment.live.monitoring import live_state
from frontend.api import live

BROKER = "ftmo"  # a real known broker with prop-firm risk_rules


def _now_iso(offset_secs: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_secs)).isoformat()


@pytest.fixture
def live_dir(tmp_path, monkeypatch):
    """Redirect every broker's live_state dir under a tmp path."""
    monkeypatch.setattr(live.live_state, "live_state_dir", lambda b: tmp_path / b)
    return tmp_path


def _write_snapshot(live_dir, broker=BROKER, *, tier="demo", age_secs=0.0, **over):
    state_dir = live_dir / broker
    snap = {
        "schema_version": 1,
        "broker": broker,
        "venue": "MT5",
        "exec_tier": tier,
        "account": {"login": 1, "server": "FTMO-Demo", "currency": "USD",
                    "balance": 100_000.0, "floating_pnl": -500.0, "equity": 99_500.0,
                    "leverage_in_use": 0.4},
        "positions": [],
        "gross_notional": 40_000.0,
        "marks_fresh": True,
        "targets": [],
        "warmup": None,
        "strategy_state": "IDLE",
        "halted": False,
        "ready_to_trade": True,
        "initial_balance": 100_000.0,
        "risk_baseline": {"account_start_equity": 100_000.0, "account_start_at": "t0",
                          "day_start_equity": 100_000.0, "day_start_balance": 100_000.0,
                          "day_start_date": "2026-06-08", "day_start_estimated": False},
        "last_command_result": None,
        "node_started_at": _now_iso(-3600),
        "ts": _now_iso(-age_secs),
    }
    snap.update(over)
    live_state.write_snapshot(live_state.snapshot_path(state_dir), snap)
    return snap


# ── compute_risk_gauges (pure) ──────────────────────────────────────────────────

def test_gauges_full_ftmo_basis():
    rules = live.brokers.risk_rules(BROKER)  # FTMO 5/10/10
    # FTMO basis: limits are dollars = pct of the INITIAL balance (100k), not of
    # day-start equity. day anchor = max(day_start_balance, day_start_equity).
    g = live.compute_risk_gauges(
        equity=97_000.0, account_start=100_000.0, day_start_equity=99_000.0,
        day_start_balance=99_000.0, initial_balance=100_000.0,
        gross_notional=50_000.0, rules=rules,
    )
    # daily drawdown = (99k - 97k)/100k = 2.0% of initial; used = 2k / (5% of 100k) = 40%
    assert g["daily_drawdown_pct"] == pytest.approx(2.0)
    assert g["daily_used_pct"] == pytest.approx(40.0)
    assert g["total_drawdown_pct"] == pytest.approx(3.0)            # (100k-97k)/100k
    assert g["total_headroom_pct"] == pytest.approx(7.0)            # 10% - 3%
    assert g["leverage_in_use"] == pytest.approx(50_000 / 97_000)


def test_gauges_day_anchor_uses_max_balance_equity():
    rules = live.brokers.risk_rules(BROKER)
    # Held overnight in profit: day-start equity (101k) > balance (100k); anchor = 101k.
    g = live.compute_risk_gauges(
        equity=99_000.0, account_start=100_000.0, day_start_equity=101_000.0,
        day_start_balance=100_000.0, initial_balance=100_000.0,
        gross_notional=None, rules=rules,
    )
    assert g["daily_drawdown_pct"] == pytest.approx(2.0)  # (101k - 99k)/100k


def test_gauges_none_inputs_safe():
    g = live.compute_risk_gauges(
        equity=None, account_start=None, day_start_equity=None, day_start_balance=None,
        initial_balance=None, gross_notional=None, rules=None,
    )
    assert g["daily_pnl_pct"] is None
    assert g["total_drawdown_pct"] is None
    assert g["leverage_in_use"] is None


def test_gauges_no_limits_still_reports_pnl():
    g = live.compute_risk_gauges(
        equity=101_000.0, account_start=100_000.0, day_start_equity=100_000.0,
        day_start_balance=100_000.0, initial_balance=100_000.0,
        gross_notional=None, rules=None,
    )
    assert g["total_pnl_pct"] == pytest.approx(1.0)
    assert g["daily_limit_pct"] is None  # no rules → no limit


# ── snapshot reads ──────────────────────────────────────────────────────────────

def test_get_snapshot_missing_raises(live_dir):
    with pytest.raises(FileNotFoundError):
        live.get_snapshot(BROKER)


def test_get_snapshot_fresh_is_online(live_dir):
    _write_snapshot(live_dir, age_secs=2.0)
    snap = live.get_snapshot(BROKER)
    assert snap["online"] is True
    assert snap["account"]["equity"] == 99_500.0


def test_get_snapshot_stale_is_offline(live_dir):
    _write_snapshot(live_dir, age_secs=120.0)
    snap = live.get_snapshot(BROKER)
    assert snap["online"] is False


def test_get_snapshot_unknown_broker_raises(live_dir):
    with pytest.raises(ValueError):
        live.get_snapshot("not_a_broker")


def test_get_risk_uses_snapshot_equity_and_rules(live_dir):
    _write_snapshot(live_dir)
    risk = live.get_risk(BROKER)
    assert risk["has_limits"] is True
    assert risk["limits"]["max_daily_loss_pct"] == 5.0
    # equity 99_500 vs day_start 100_000 → 0.5% daily drawdown
    assert risk["gauges"]["daily_drawdown_pct"] == pytest.approx(0.5)


def test_list_brokers_marks_snapshot_presence(live_dir):
    _write_snapshot(live_dir, age_secs=1.0)
    by_name = {b["broker"]: b for b in live.list_brokers()["brokers"]}
    assert by_name[BROKER]["has_snapshot"] is True
    assert by_name[BROKER]["online"] is True


def test_get_equity_series(live_dir):
    state_dir = live_dir / BROKER
    live_state.append_equity_sample(live_state.equity_path(state_dir), ts_iso="t0", equity=100_000.0)
    out = live.get_equity(BROKER)
    assert out["series"][0]["equity"] == 100_000.0


# ── flatten safety gates ────────────────────────────────────────────────────────

def test_flatten_confirm_mismatch_raises(live_dir):
    _write_snapshot(live_dir)
    with pytest.raises(ValueError):
        live.flatten(BROKER, confirm="wrong")


def test_flatten_no_node_raises(live_dir):
    with pytest.raises(FileNotFoundError):
        live.flatten(BROKER, confirm=BROKER)


def test_flatten_stale_node_raises(live_dir):
    _write_snapshot(live_dir, age_secs=120.0)
    with pytest.raises(RuntimeError):
        live.flatten(BROKER, confirm=BROKER)


def test_flatten_live_account_refused(live_dir):
    _write_snapshot(live_dir, tier="live", age_secs=2.0)
    with pytest.raises(PermissionError):
        live.flatten(BROKER, confirm=BROKER)


def test_flatten_demo_writes_command(live_dir):
    _write_snapshot(live_dir, tier="demo", age_secs=2.0)
    out = live.flatten(BROKER, confirm=BROKER)
    assert out["status"] == "submitted"
    cmd = live_state.read_command(live_state.command_path(live_dir / BROKER))
    assert cmd is not None
    assert cmd.action == "flatten"
    assert cmd.id == out["command_id"]
    assert cmd.confirm == BROKER
