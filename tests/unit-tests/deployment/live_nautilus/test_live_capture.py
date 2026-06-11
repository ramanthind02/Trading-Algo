"""Node-side durable capture (migration plan §7.1 + §7.2 + §7.4 + §7.5):
the forecast decision log, the exec client's deal log, the submit-result log,
and the durable equity log. All are append-only JSONL whose failure must never
propagate into the trading loop.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.live.monitoring import forecast_log
from deployment.live.monitoring import live_state
from deployment.live.monitoring import slippage


# ── forecasts.jsonl ─────────────────────────────────────────────────────────


def test_record_forecasts_appends_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(forecast_log, "live_state_dir", lambda broker: tmp_path / broker)
    rows = [
        {"as_of": "2026-06-09", "canonical": "NQ", "forecast_score": 1.2,
         "target_fraction": 0.05, "engine_config_hash": "abc", "vault_root": "vault",
         "warmup_ready": 1, "ts": "t1"},
        {"as_of": "2026-06-09", "canonical": "GC", "forecast_score": -0.4,
         "target_fraction": -0.02, "engine_config_hash": "abc", "vault_root": "vault",
         "warmup_ready": 1, "ts": "t1"},
    ]
    forecast_log.record_forecasts("ftmo", rows)
    forecast_log.record_forecasts("ftmo", rows[:1])  # second decision appends

    lines = forecast_log.forecasts_jsonl_path("ftmo").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["schema_version"] == forecast_log.SCHEMA_VERSION
    assert first["broker"] == "ftmo"
    assert first["canonical"] == "NQ"
    assert first["target_fraction"] == 0.05


def test_record_forecasts_empty_rows_is_noop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(forecast_log, "live_state_dir", lambda broker: tmp_path / broker)
    forecast_log.record_forecasts("ftmo", [])
    assert not forecast_log.forecasts_jsonl_path("ftmo").exists()


# ── deals.jsonl (vendored exec client) ──────────────────────────────────────


def _exec_client_class():
    # The vendored adapter is not installed; mirror node_builder's sys.path
    # injection of deployment/nautilus_mt5/vendor/mt5-connect.
    import sys

    from deployment.live.runtime.node_builder import vendored_adapter_path

    adapter_root = str(vendored_adapter_path())
    if adapter_root not in sys.path:
        sys.path.insert(0, adapter_root)
    from mt5connect.execution import MT5LiveExecutionClient  # vendored

    return MT5LiveExecutionClient


def _fake_deal() -> SimpleNamespace:
    return SimpleNamespace(
        ticket=12345, order=678, time=1_780_000_000, time_msc=1_780_000_000_123,
        type=1, entry=1, magic=510, position_id=999, reason=0, volume=0.02,
        price=21500.5, commission=-0.08, swap=-1.23, fee=0.0, profit=14.5,
        symbol="NDX", comment="O-20260609-001", external_id="",
    )


def _fake_self(deals_log_path: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        _config=SimpleNamespace(deals_log_path=deals_log_path),
        _log=logging.getLogger("test_deal_capture"),
    )


def test_append_deal_log_writes_full_record(tmp_path: Path) -> None:
    cls = _exec_client_class()
    log_path = tmp_path / "live_state" / "deals.jsonl"
    fake = _fake_self(str(log_path))

    cls._append_deal_log(fake, _fake_deal())
    cls._append_deal_log(fake, _fake_deal())

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    # The fields the Nautilus fill event DROPS must be present here.
    assert rec["swap"] == -1.23
    assert rec["profit"] == 14.5
    assert rec["fee"] == 0.0
    assert rec["entry"] == 1
    assert rec["position_id"] == 999
    assert rec["ticket"] == 12345
    assert rec["comment"] == "O-20260609-001"
    assert "captured_at" in rec


def test_append_deal_log_disabled_when_no_path(tmp_path: Path) -> None:
    cls = _exec_client_class()
    fake = _fake_self(None)
    cls._append_deal_log(fake, _fake_deal())  # must be a silent no-op
    assert not list(tmp_path.iterdir())


def test_append_deal_log_never_raises(tmp_path: Path) -> None:
    cls = _exec_client_class()
    # Point the log at a path whose parent is a FILE -> mkdir/open must fail,
    # and the failure must be swallowed (capture never breaks the poll loop).
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    fake = _fake_self(str(blocker / "deals.jsonl"))
    cls._append_deal_log(fake, _fake_deal())  # no exception


# ── slippage.record_submit intent + target_fraction (migration plan §7.2) ────


def test_record_submit_with_intent_and_target_fraction(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(slippage, "live_state_dir", lambda broker: tmp_path / broker / "live_state")
    slippage.record_submit(
        "ftmo", canonical="NQ", side="BUY", qty=1.0,
        bid=21000.0, ask=21001.0, client_order_id="O-001",
        broker_time_iso="2026-06-09T18:00:00",
        intent="ENTRY", target_fraction=0.05,
    )
    rec = json.loads(slippage.submits_jsonl_path("ftmo").read_text(encoding="utf-8").strip())
    assert rec["intent"] == "ENTRY"
    assert rec["target_fraction"] == pytest.approx(0.05)
    assert rec["canonical"] == "NQ"


def test_record_submit_without_intent_produces_legacy_shape(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(slippage, "live_state_dir", lambda broker: tmp_path / broker / "live_state")
    slippage.record_submit(
        "ftmo", canonical="ES", side="SELL", qty=0.5,
        bid=5200.0, ask=5201.0, client_order_id="O-002",
        broker_time_iso="2026-06-09T18:00:00",
    )
    rec = json.loads(slippage.submits_jsonl_path("ftmo").read_text(encoding="utf-8").strip())
    # Legacy keys present, optional keys absent
    assert "client_order_id" in rec
    assert "intent" not in rec
    assert "target_fraction" not in rec


# ── _append_submit_result_log (migration plan §7.2) ──────────────────────────


def _fake_self_submit(submit_results_log_path: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        _config=SimpleNamespace(submit_results_log_path=submit_results_log_path),
        _log=logging.getLogger("test_submit_capture"),
    )


def _fake_result() -> SimpleNamespace:
    return SimpleNamespace(
        retcode=10009, deal=99, order=678, volume=0.02,
        price=21500.5, bid=21500.0, ask=21501.0,
    )


def test_append_submit_result_log_writes_full_record(tmp_path: Path) -> None:
    cls = _exec_client_class()
    log_path = tmp_path / "live_state" / "submit_results.jsonl"
    fake = _fake_self_submit(str(log_path))

    cls._append_submit_result_log(fake, "O-20260609-001", _fake_result())
    cls._append_submit_result_log(fake, "O-20260609-002", _fake_result())

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["client_order_id"] == "O-20260609-001"
    assert rec["retcode"] == 10009
    assert rec["deal"] == 99
    assert rec["order"] == 678
    assert rec["volume"] == pytest.approx(0.02)
    assert rec["price"] == pytest.approx(21500.5)
    assert rec["bid"] == pytest.approx(21500.0)
    assert rec["ask"] == pytest.approx(21501.0)
    assert "captured_at" in rec


def test_append_submit_result_log_disabled_when_no_path(tmp_path: Path) -> None:
    cls = _exec_client_class()
    fake = _fake_self_submit(None)
    cls._append_submit_result_log(fake, "O-001", _fake_result())  # must be silent no-op
    assert not list(tmp_path.iterdir())


def test_append_submit_result_log_never_raises(tmp_path: Path) -> None:
    cls = _exec_client_class()
    # Parent is a file, not a directory — mkdir/open will fail; must be swallowed.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    fake = _fake_self_submit(str(blocker / "submit_results.jsonl"))
    cls._append_submit_result_log(fake, "O-001", _fake_result())  # no exception


# ── append_equity_sample_durable (migration plan §7.5) ───────────────────────


def test_durable_equity_grows_past_capped_compaction(tmp_path: Path, monkeypatch) -> None:
    # Drive compaction by making the byte budget tiny and the kept-sample cap small.
    monkeypatch.setattr(live_state, "MAX_EQUITY_SAMPLES", 2)
    monkeypatch.setattr(live_state, "_EQUITY_COMPACT_BYTES", 5)

    capped_path = tmp_path / "equity_history.jsonl"
    durable_path = tmp_path / "equity.jsonl"

    for i in range(5):
        ts = f"2026-06-09T{i:02d}:00:00"
        eq = 100_000.0 + i
        live_state.append_equity_sample(str(capped_path), ts_iso=ts, equity=eq)
        live_state.append_equity_sample_durable(
            str(durable_path), ts_iso=ts, equity=eq,
            balance=eq - 50.0, floating_pnl=50.0,
            gross_notional=200_000.0, marks_fresh=True,
        )

    capped_lines = capped_path.read_text(encoding="utf-8").splitlines()
    durable_lines = durable_path.read_text(encoding="utf-8").splitlines()

    # Capped file is compacted to MAX_EQUITY_SAMPLES=2; durable keeps all 5.
    assert len(capped_lines) <= 2
    assert len(durable_lines) == 5
    assert len(durable_lines) > len(capped_lines)

    # Durable records carry the extra fields.
    for line in durable_lines:
        rec = json.loads(line)
        assert "balance" in rec
        assert "floating_pnl" in rec
        assert "gross_notional" in rec
        assert "marks_fresh" in rec
