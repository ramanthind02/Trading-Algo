"""Unit tests for the live-state functional core (deployment/live/monitoring/live_state.py).

Pure module: file paths, risk-baseline math, command parsing, equity history, and
atomic snapshot/baseline IO — no NautilusTrader / MetaTrader5 needed.
"""
from __future__ import annotations

import json

import pytest

from deployment.live.monitoring import live_state


# ── risk baseline transitions ──────────────────────────────────────────────────

def test_baseline_sets_account_anchor_once_from_first_equity():
    b0 = live_state.RiskBaseline()
    b1 = live_state.update_baseline(b0, equity=100_000.0, broker_date="2026-06-08", now_iso="t0")
    assert b1.account_start_equity == 100_000.0
    assert b1.day_start_equity == 100_000.0
    assert b1.day_start_date == "2026-06-08"

    # Later (same day, lower equity): anchors are preserved.
    b2 = live_state.update_baseline(b1, equity=98_000.0, broker_date="2026-06-08", now_iso="t1")
    assert b2.account_start_equity == 100_000.0  # account anchor never moves
    assert b2.day_start_equity == 100_000.0      # day anchor held within the day


def test_baseline_account_anchor_uses_initial_balance_when_given():
    b0 = live_state.RiskBaseline()
    b1 = live_state.update_baseline(
        b0, equity=97_500.0, broker_date="2026-06-08", now_iso="t0", initial_balance=100_000.0
    )
    assert b1.account_start_equity == 100_000.0  # the configured initial balance, not equity


def test_baseline_day_anchor_resets_on_date_rollover():
    b1 = live_state.update_baseline(
        live_state.RiskBaseline(), equity=100_000.0, broker_date="2026-06-08", now_iso="t0"
    )
    b1 = live_state.update_baseline(b1, equity=98_000.0, broker_date="2026-06-08", now_iso="t1")
    b2 = live_state.update_baseline(b1, equity=98_000.0, broker_date="2026-06-09", now_iso="t2")
    assert b2.day_start_equity == 98_000.0        # reset to current equity at the new day
    assert b2.day_start_date == "2026-06-09"
    assert b2.account_start_equity == 100_000.0   # account anchor still original


def test_baseline_records_day_start_balance():
    b = live_state.update_baseline(
        live_state.RiskBaseline(), equity=99_000.0, balance=99_500.0,
        broker_date="2026-06-08", now_iso="t0",
    )
    assert b.day_start_equity == 99_000.0
    assert b.day_start_balance == 99_500.0


def test_baseline_cold_start_into_new_day_is_estimated():
    prev = live_state.RiskBaseline(
        account_start_equity=100_000.0, account_start_at="t0",
        day_start_equity=100_000.0, day_start_balance=100_000.0, day_start_date="2026-06-08",
    )
    # prev_observed_date None == first tick after a restart into a NEW broker day.
    nb = live_state.update_baseline(
        prev, equity=97_000.0, balance=98_000.0, broker_date="2026-06-09",
        now_iso="t", prev_observed_date=None,
    )
    assert nb.day_start_date == "2026-06-09"
    assert nb.day_start_equity == 97_000.0
    assert nb.day_start_estimated is True


def test_baseline_witnessed_rollover_not_estimated():
    prev = live_state.RiskBaseline(
        account_start_equity=100_000.0, account_start_at="t0",
        day_start_equity=100_000.0, day_start_balance=100_000.0, day_start_date="2026-06-08",
    )
    nb = live_state.update_baseline(
        prev, equity=99_000.0, balance=99_500.0, broker_date="2026-06-09",
        now_iso="t", prev_observed_date="2026-06-08",  # node was running yesterday
    )
    assert nb.day_start_estimated is False
    assert nb.day_start_equity == 99_000.0


def test_baseline_persist_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    b = live_state.RiskBaseline(
        account_start_equity=100_000.0, account_start_at="t0",
        day_start_equity=99_000.0, day_start_balance=99_500.0,
        day_start_date="2026-06-08", day_start_estimated=True,
    )
    live_state.save_baseline(path, b)
    assert live_state.load_baseline(path) == b


def test_baseline_load_missing_is_empty(tmp_path):
    assert live_state.load_baseline(tmp_path / "nope.json") == live_state.RiskBaseline()


# ── halt persistence (durable kill switch) ──────────────────────────────────────

def test_halt_roundtrip_and_clear(tmp_path):
    path = tmp_path / "halt.json"
    assert live_state.load_halt(path) == live_state.HaltState()
    live_state.save_halt(path, live_state.HaltState(halted=True, by_command_id="x", at="t"))
    loaded = live_state.load_halt(path)
    assert loaded.halted is True
    assert loaded.by_command_id == "x"
    live_state.clear_halt(path)
    assert live_state.load_halt(path).halted is False
    live_state.clear_halt(path)  # idempotent on missing


# ── command channel ─────────────────────────────────────────────────────────────

def test_command_roundtrip(tmp_path):
    path = tmp_path / "command.json"
    cmd = live_state.Command(id="abc", action="flatten", issued_at="2026-06-08T12:00:00+00:00", confirm="ftmo")
    live_state.write_command(path, cmd)
    assert live_state.read_command(path) == cmd


def test_command_missing_is_none(tmp_path):
    assert live_state.read_command(tmp_path / "command.json") is None


def test_command_clear_consumes_file(tmp_path):
    path = tmp_path / "command.json"
    live_state.write_command(path, live_state.Command(id="a", action="flatten", issued_at="t"))
    assert live_state.read_command(path) is not None
    live_state.clear_command(path)
    assert live_state.read_command(path) is None  # never re-read after a restart
    live_state.clear_command(path)  # idempotent on missing


def test_command_malformed_is_none(tmp_path):
    path = tmp_path / "command.json"
    path.write_text(json.dumps({"id": "x"}), encoding="utf-8")  # missing action/issued_at
    assert live_state.read_command(path) is None


# ── equity history ──────────────────────────────────────────────────────────────

def test_equity_append_and_read(tmp_path):
    path = tmp_path / "equity.jsonl"
    live_state.append_equity_sample(path, ts_iso="t0", equity=100_000.0)
    live_state.append_equity_sample(path, ts_iso="t1", equity=100_500.0)
    series = live_state.read_equity_series(path)
    assert [p["equity"] for p in series] == [100_000.0, 100_500.0]
    assert series[0]["ts"] == "t0"


def test_equity_read_tolerates_torn_final_line(tmp_path):
    path = tmp_path / "equity.jsonl"
    live_state.append_equity_sample(path, ts_iso="t0", equity=100_000.0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"ts": "t1", "equity": 100')  # crash mid-append → torn line
    series = live_state.read_equity_series(path)
    assert len(series) == 1 and series[0]["equity"] == 100_000.0


def test_equity_read_skips_json_valid_non_numeric_value(tmp_path):
    path = tmp_path / "equity.jsonl"
    live_state.append_equity_sample(path, ts_iso="t0", equity=100.0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"ts":"t1","equity":"NaNstr"}\n')  # valid JSON, bad value
    live_state.append_equity_sample(path, ts_iso="t2", equity=200.0)
    series = live_state.read_equity_series(path)
    assert [p["equity"] for p in series] == [100.0, 200.0]  # bad line skipped, not fatal


def test_equity_read_limit_keeps_latest(tmp_path):
    path = tmp_path / "equity.jsonl"
    for i in range(10):
        live_state.append_equity_sample(path, ts_iso=f"t{i}", equity=float(i))
    series = live_state.read_equity_series(path, limit=3)
    assert [p["equity"] for p in series] == [7.0, 8.0, 9.0]


def test_equity_read_missing_is_empty(tmp_path):
    assert live_state.read_equity_series(tmp_path / "nope.jsonl") == []


# ── snapshot IO ─────────────────────────────────────────────────────────────────

def test_snapshot_roundtrip(tmp_path):
    path = tmp_path / "snapshot.json"
    snap = {"schema_version": 1, "broker": "ftmo", "account": {"equity": 99_000.0}, "positions": []}
    live_state.write_snapshot(path, snap)
    assert live_state.read_snapshot(path) == snap


def test_snapshot_missing_is_none(tmp_path):
    assert live_state.read_snapshot(tmp_path / "snapshot.json") is None


def test_paths_are_under_live_state_dir():
    d = live_state.live_state_dir("ftmo")
    assert d.name == "live_state"
    assert live_state.snapshot_path(d).name == "snapshot.json"
    assert live_state.command_path(d).name == "command.json"
    assert live_state.baseline_path(d).name == "baseline.json"
    assert live_state.equity_path(d).name == "equity_history.jsonl"
