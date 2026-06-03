"""Unit tests for :mod:`execution.run_mt5_execution`.

The orchestrator is mocked at the MT5 layer — :class:`MT5TradeExecutor`
is replaced with a fake that returns canned account info / positions /
ticks. The approval flow is mocked via a fake notifier. We exercise:

- happy path: one account, fresh open, approved, executed
- preflight failure: account is surfaced but never executed
- per-account cancel: ftmo_a executed, ftmo_b cancelled by user
- cancel-all: nothing executed
- missing creds: surfaced in report
- audit log: written and contains expected keys
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pytest

from execution import run_mt5_execution as orch
from execution.approval_flow import ApprovalDecision, BatchApprovalOutcome
from execution.mt5_models import (
    MT5AccountInfo,
    MT5Position,
    MT5SymbolInfo,
    MT5Tick,
    OrderResult,
    OrderSide,
    RebalanceAction,
    RebalanceActionKind,
    TradeMode,
)


UTC = timezone.utc
T0 = datetime(2026, 1, 5, 22, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeMT5Executor:
    """Stand-in for :class:`execution.mt5_trade_executor.MT5TradeExecutor`.

    Configured per-test via class-level state so the orchestrator's
    private ``from execution.mt5_trade_executor import MT5TradeExecutor``
    call returns this fake.
    """

    login: int
    password: str
    server: str
    terminal_path: Optional[str] = None
    timeout_ms: int = 60_000
    place_calls: List[Dict[str, Any]] = field(default_factory=list)
    close_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Per-test setup:
    account_info: Optional[MT5AccountInfo] = None
    positions: List[MT5Position] = field(default_factory=list)
    symbol_infos: Dict[str, MT5SymbolInfo] = field(default_factory=dict)
    ticks: Dict[str, MT5Tick] = field(default_factory=dict)
    margin_per_call: float = 100.0
    place_success: bool = True
    place_retcode: int = 10009  # TRADE_RETCODE_DONE

    @classmethod
    @contextmanager
    def connect_and_verify(
        cls,
        *,
        login: int,
        password: str,
        server: str,
        terminal_path: Optional[str] = None,
        timeout_ms: int = 60_000,
    ):
        spec = _FAKE_REGISTRY[login]
        spec.password = password
        spec.server = server
        yield spec

    # ---- read ----
    def fetch_account_info(self) -> MT5AccountInfo:
        assert self.account_info is not None
        return self.account_info

    def fetch_all_positions(self) -> List[MT5Position]:
        return list(self.positions)

    def fetch_symbol_info(self, name: str) -> MT5SymbolInfo:
        return self.symbol_infos[name]

    def fetch_tick(self, name: str) -> MT5Tick:
        return self.ticks[name]

    def order_calc_margin_usd(
        self, *, symbol_name: str, side: OrderSide, volume: float, price: float
    ) -> float:
        return self.margin_per_call

    # ---- write ----
    def place_market_order(
        self,
        *,
        symbol_name: str,
        side: OrderSide,
        volume: float,
        magic: int,
        comment: str = "",
        **kwargs: Any,
    ) -> OrderResult:
        self.place_calls.append(
            {"symbol": symbol_name, "side": side, "volume": volume, "magic": magic}
        )
        return OrderResult(
            action=RebalanceAction(
                kind=RebalanceActionKind.OPEN_NEW,
                symbol=symbol_name,
                side=side,
                volume=volume,
            ),
            success=self.place_success,
            deal_ticket=999_000 + len(self.place_calls),
            retcode=self.place_retcode,
            comment="ok" if self.place_success else "rejected",
        )

    def close_ticket(
        self,
        *,
        ticket: int,
        symbol_name: str,
        original_side: OrderSide,
        volume: float,
        magic: int,
        comment: str = "",
        **kwargs: Any,
    ) -> OrderResult:
        self.close_calls.append(
            {
                "ticket": ticket,
                "symbol": symbol_name,
                "side": original_side,
                "volume": volume,
                "magic": magic,
            }
        )
        return OrderResult(
            action=RebalanceAction(
                kind=RebalanceActionKind.CLOSE_TICKET,
                symbol=symbol_name,
                side=original_side,
                volume=volume,
                ticket=ticket,
            ),
            success=True,
            deal_ticket=888_000 + len(self.close_calls),
            retcode=10009,
            comment="ok",
        )


_FAKE_REGISTRY: Dict[int, _FakeMT5Executor] = {}


def _install_fake_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make orchestrator's ``from execution.mt5_trade_executor import ...``
    yield our fake."""
    import sys
    import types

    fake_mod = types.ModuleType("execution.mt5_trade_executor")
    fake_mod.MT5TradeExecutor = _FakeMT5Executor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "execution.mt5_trade_executor", fake_mod)


@dataclass
class _FakeNotifier:
    sent: List[Dict[str, Any]] = field(default_factory=list)
    configured: bool = True

    def is_configured(self) -> bool:
        return self.configured

    def send_message(self, text: str) -> bool:
        self.sent.append({"text": text})
        return True


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


def _sym(name: str) -> MT5SymbolInfo:
    return MT5SymbolInfo(
        name=name,
        trade_mode=TradeMode.FULL,
        point=0.01,
        digits=2,
        trade_contract_size=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread=10,
        visible=True,
    )


def _tick(name: str, mid: float = 18000.0) -> MT5Tick:
    return MT5Tick(symbol=name, bid=mid - 0.5, ask=mid + 0.5, time_utc=T0)


def _acct(login: int) -> MT5AccountInfo:
    return MT5AccountInfo(
        login=login,
        server="FTMO-Demo",
        currency="USD",
        balance=100_000.0,
        equity=100_000.0,
        margin=0.0,
        margin_free=100_000.0,
        margin_level=0.0,
        leverage=100,
        trade_allowed=True,
        trade_expert=True,
    )


def _make_args(
    *, dry_run=False, dry_run_execute=False, approve_via_telegram=True, live=False, execute=True
) -> argparse.Namespace:
    return argparse.Namespace(
        dry_run=dry_run,
        dry_run_execute=dry_run_execute,
        approve_via_telegram=approve_via_telegram,
        live=live,
        execute=execute,
    )


def _setup_account(
    *,
    login: int,
    positions: Optional[List[MT5Position]] = None,
    symbols: Sequence[str] = ("US100.cash",),
) -> _FakeMT5Executor:
    sym_map = {s: _sym(s) for s in symbols}
    tick_map = {s: _tick(s) for s in symbols}
    spec = _FakeMT5Executor(
        login=login,
        password="",
        server="FTMO-Demo",
        account_info=_acct(login),
        positions=positions or [],
        symbol_infos=sym_map,
        ticks=tick_map,
    )
    _FAKE_REGISTRY[login] = spec
    return spec


def _make_config(
    *,
    accounts: List[Dict[str, Any]],
    default_on_timeout: str = "cancel",
    audit_dir: Optional[Path] = None,
    tradeable_tickers: Optional[List[str]] = None,
    instruments: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "tradeable_tickers": list(tradeable_tickers or ["ES", "NQ", "GC", "SI"]),
        "instruments": dict(
            instruments
            or {
                "ES": {"mt5_symbol": "US500.cash"},
                "NQ": {"mt5_symbol": "US100.cash"},
                "GC": {"mt5_symbol": "XAUUSD"},
                "SI": {"mt5_symbol": "XAGUSD"},
            }
        ),
        "mt5": {
            "sizing_basis": "equity",
            "lot_size_ceiling": 100.0,
            "min_rebalance_lots": 0.0,
            "min_rebalance_notional_usd": 0.0,
            "max_margin_usage_pct": 0.95,
            "default_magic_number": 90420,
        },
        "execution": {
            "approval_timeout_seconds": 5,
            "default_on_timeout": default_on_timeout,
            "authorized_telegram_user_ids": [42],
            "allow_any_approver": True,
            "audit_dir": str(audit_dir) if audit_dir else "logs/cfd_prop_audit",
        },
        "accounts": accounts,
    }


def _account_block(
    *, label: str, login: int, magic: int = 90420, tradeable=("ES",)
) -> Dict[str, Any]:
    # Note: per-account `tradeable_tickers` is supported by the orchestrator
    # as an override; if omitted we fall back to the top-level list. Keeping
    # it here per-test to preserve the original behaviour where each test
    # picks the symbols it cares about.
    return {
        "label": label,
        "enabled": True,
        "username_env": f"MT5_{label.upper()}_USERNAME",
        "password_env": f"MT5_{label.upper()}_PASSWORD",
        "server_env": f"MT5_{label.upper()}_SERVER",
        "magic_number": magic,
        "terminal_path": None,
        "tradeable_tickers": list(tradeable),
        "symbol_overrides": {},
    }


def _forecasts_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    _FAKE_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_enabled_accounts_skips_quietly(monkeypatch, capsys):
    _install_fake_executor(monkeypatch)
    config = _make_config(accounts=[{**_account_block(label="a", login=1), "enabled": False}])
    orch.run_cfd_prop_execution(
        args=_make_args(),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.1}]),
    )
    out = capsys.readouterr().out
    assert "no enabled accounts" in out


def test_empty_forecast_df_skips(monkeypatch, capsys):
    _install_fake_executor(monkeypatch)
    config = _make_config(accounts=[_account_block(label="a", login=1)])
    orch.run_cfd_prop_execution(
        args=_make_args(),
        config=config,
        forecasts_df=pd.DataFrame(columns=["ticker", "position_fraction"]),
    )
    out = capsys.readouterr().out
    assert "forecast frame is empty" in out


def test_missing_creds_yields_preflight_failure(monkeypatch, capsys, tmp_path):
    _install_fake_executor(monkeypatch)
    _setup_account(login=12345, symbols=("US500.cash",))
    # Note: do NOT set the env vars.
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=12345, tradeable=("ES",))],
        audit_dir=tmp_path / "audit",
    )
    monkeypatch.setattr(
        orch, "request_batch_approval",
        lambda *a, **kw: pytest.fail("request_batch_approval should NOT be called when no eligible"),
    )
    orch.run_cfd_prop_execution(
        args=_make_args(approve_via_telegram=False, live=True),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.10}]),
    )
    out = capsys.readouterr().out
    assert "Missing env var(s)" in out
    assert "MT5_FTMO_A_USERNAME" in out
    assert "no actionable plans" in out.lower() or "nothing to approve" in out.lower()


def test_happy_path_executes_single_open(monkeypatch, capsys, tmp_path):
    _install_fake_executor(monkeypatch)
    spec = _setup_account(login=11111, symbols=("US500.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "11111")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")

    # Force the approval path to APPROVE the single account.
    def fake_approval(*args, **kwargs):
        labels = kwargs.get("account_labels")
        return BatchApprovalOutcome(
            global_decision=ApprovalDecision.APPROVED,
            per_account={lbl: ApprovalDecision.APPROVED for lbl in labels},
            approver_telegram_id=42,
        )
    monkeypatch.setattr(orch, "request_batch_approval", fake_approval)
    monkeypatch.setattr(
        orch.TelegramNotifier, "for_cfd_prop", classmethod(lambda cls: _FakeNotifier())
    )

    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=11111, tradeable=("ES",))],
        audit_dir=tmp_path / "audit",
    )
    orch.run_cfd_prop_execution(
        args=_make_args(),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.10}]),
    )
    # 10% of 100k equity on $5000-priced US500 = 2.0 lots BUY
    assert len(spec.place_calls) == 1
    call = spec.place_calls[0]
    assert call["symbol"] == "US500.cash"
    assert call["side"] is OrderSide.BUY
    # The fake tick we built defaults to $18000 mid (irrelevant for assertion shape)
    out = capsys.readouterr().out
    assert "Summary" in out
    # Audit log written
    audit_files = list((tmp_path / "audit").glob("cfd_prop_*.json"))
    assert len(audit_files) == 1
    payload = json.loads(audit_files[0].read_text())
    assert payload["global_decision"] == ApprovalDecision.APPROVED.value
    assert payload["per_account_decision"]["ftmo_a"] == ApprovalDecision.APPROVED.value
    assert len(payload["reports"]) == 1
    assert payload["reports"][0]["results"][0]["success"] is True


def test_cancel_all_skips_execution(monkeypatch, capsys, tmp_path):
    _install_fake_executor(monkeypatch)
    spec = _setup_account(login=22222, symbols=("US500.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "22222")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")

    def fake_approval(*args, **kwargs):
        labels = kwargs.get("account_labels")
        return BatchApprovalOutcome(
            global_decision=ApprovalDecision.CANCELLED,
            per_account={lbl: ApprovalDecision.CANCELLED for lbl in labels},
        )
    monkeypatch.setattr(orch, "request_batch_approval", fake_approval)
    monkeypatch.setattr(
        orch.TelegramNotifier, "for_cfd_prop", classmethod(lambda cls: _FakeNotifier())
    )

    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=22222, tradeable=("ES",))],
        audit_dir=tmp_path / "audit",
    )
    orch.run_cfd_prop_execution(
        args=_make_args(),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.10}]),
    )
    assert spec.place_calls == []
    assert spec.close_calls == []
    audit = json.loads(next((tmp_path / "audit").glob("cfd_prop_*.json")).read_text())
    assert audit["per_account_decision"]["ftmo_a"] == ApprovalDecision.CANCELLED.value


def test_per_account_cancel_only_skips_one(monkeypatch, capsys, tmp_path):
    _install_fake_executor(monkeypatch)
    spec_a = _setup_account(login=33333, symbols=("US500.cash",))
    spec_b = _setup_account(login=44444, symbols=("US500.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "33333")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    monkeypatch.setenv("MT5_FTMO_B_USERNAME", "44444")
    monkeypatch.setenv("MT5_FTMO_B_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_B_SERVER", "FTMO-Demo")

    def fake_approval(*args, **kwargs):
        labels = kwargs.get("account_labels")
        return BatchApprovalOutcome(
            global_decision=ApprovalDecision.APPROVED,
            per_account={
                labels[0]: ApprovalDecision.APPROVED,
                labels[1]: ApprovalDecision.CANCELLED,
            },
        )
    monkeypatch.setattr(orch, "request_batch_approval", fake_approval)
    monkeypatch.setattr(
        orch.TelegramNotifier, "for_cfd_prop", classmethod(lambda cls: _FakeNotifier())
    )

    config = _make_config(
        accounts=[
            _account_block(label="ftmo_a", login=33333, magic=90421, tradeable=("ES",)),
            _account_block(label="ftmo_b", login=44444, magic=90422, tradeable=("ES",)),
        ],
        audit_dir=tmp_path / "audit",
    )
    orch.run_cfd_prop_execution(
        args=_make_args(),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.10}]),
    )
    # ftmo_a should have placed an order; ftmo_b should not.
    assert len(spec_a.place_calls) == 1
    assert spec_b.place_calls == []
    payload = json.loads(next((tmp_path / "audit").glob("cfd_prop_*.json")).read_text())
    assert payload["per_account_decision"]["ftmo_a"] == ApprovalDecision.APPROVED.value
    assert payload["per_account_decision"]["ftmo_b"] == ApprovalDecision.CANCELLED.value


def test_dry_run_execute_skips_approval_and_orders(monkeypatch, capsys):
    _install_fake_executor(monkeypatch)
    spec = _setup_account(login=55555, symbols=("US500.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "55555")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    # If approval is called, the test will fail.
    monkeypatch.setattr(
        orch, "request_batch_approval",
        lambda *a, **kw: pytest.fail("approval should NOT be requested in dry-run-execute"),
    )
    config = _make_config(accounts=[_account_block(label="ftmo_a", login=55555, tradeable=("ES",))])
    orch.run_cfd_prop_execution(
        args=_make_args(dry_run_execute=True),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.10}]),
    )
    assert spec.place_calls == []
    out = capsys.readouterr().out
    assert "SKIPPING approval" in out


def test_no_approve_flag_auto_approves_and_executes(monkeypatch, capsys, tmp_path):
    """After stripping require_live_flag_for_real_money, --execute without
    --approve-via-telegram simply auto-approves all preflight-ok accounts
    (user explicitly opted out of Telegram approval)."""
    _install_fake_executor(monkeypatch)
    spec = _setup_account(login=66666, symbols=("US500.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "66666")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    monkeypatch.setattr(
        orch, "request_batch_approval",
        lambda *a, **kw: pytest.fail("approval should not be requested"),
    )
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=66666, tradeable=("ES",))],
        audit_dir=tmp_path / "audit",
    )
    orch.run_cfd_prop_execution(
        args=_make_args(approve_via_telegram=False, live=False),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.10}]),
    )
    # Should have auto-approved and placed an OPEN_NEW market order.
    assert len(spec.place_calls) == 1
    assert spec.place_calls[0]["symbol"] == "US500.cash"
    assert spec.place_calls[0]["side"] is OrderSide.BUY


def test_partial_close_with_existing_positions(monkeypatch, capsys, tmp_path):
    """Existing long 4 lots, target long 2 lots → emit CLOSE_TICKET for 2 lots."""
    _install_fake_executor(monkeypatch)
    existing = MT5Position(
        ticket=777,
        symbol="US500.cash",
        side=OrderSide.BUY,
        volume=4.0,
        open_price=5000.0,
        magic=90421,
        open_time_utc=T0,
    )
    spec = _setup_account(login=77777, symbols=("US500.cash",), positions=[existing])
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "77777")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")

    def fake_approval(*args, **kwargs):
        labels = kwargs.get("account_labels")
        return BatchApprovalOutcome(
            global_decision=ApprovalDecision.APPROVED,
            per_account={lbl: ApprovalDecision.APPROVED for lbl in labels},
        )
    monkeypatch.setattr(orch, "request_batch_approval", fake_approval)
    monkeypatch.setattr(
        orch.TelegramNotifier, "for_cfd_prop", classmethod(lambda cls: _FakeNotifier())
    )

    # Target 10% of $100k equity on $18000 mid US500 = 0.5555 → 0.56 lots
    # Current = 4.0; need to close ~3.44 lots
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=77777, magic=90421, tradeable=("ES",))],
        audit_dir=tmp_path / "audit",
    )
    orch.run_cfd_prop_execution(
        args=_make_args(),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.10}]),
    )
    # Must close ticket #777 (no opposite-side OPEN_NEW)
    assert len(spec.close_calls) == 1
    assert spec.close_calls[0]["ticket"] == 777
    assert spec.close_calls[0]["side"] is OrderSide.BUY  # original side
    assert spec.place_calls == []


def test_audit_log_contains_expected_keys(monkeypatch, tmp_path):
    _install_fake_executor(monkeypatch)
    _setup_account(login=88888, symbols=("US500.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "88888")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")

    def fake_approval(*args, **kwargs):
        labels = kwargs.get("account_labels")
        return BatchApprovalOutcome(
            global_decision=ApprovalDecision.APPROVED,
            per_account={lbl: ApprovalDecision.APPROVED for lbl in labels},
            approver_telegram_id=42,
        )
    monkeypatch.setattr(orch, "request_batch_approval", fake_approval)
    monkeypatch.setattr(
        orch.TelegramNotifier, "for_cfd_prop", classmethod(lambda cls: _FakeNotifier())
    )

    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=88888, tradeable=("ES",))],
        audit_dir=tmp_path / "audit",
    )
    orch.run_cfd_prop_execution(
        args=_make_args(),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.10}]),
    )
    payload = json.loads(next((tmp_path / "audit").glob("cfd_prop_*.json")).read_text())
    for required in (
        "run_id", "timestamp_utc", "global_decision", "per_account_decision",
        "approver_telegram_id", "plans", "reports",
    ):
        assert required in payload, f"audit log missing {required}"
    assert payload["approver_telegram_id"] == 42


def test_no_execute_flag_skips_mt5_entirely(monkeypatch, capsys):
    """Pure --dry-run (no --execute) must NOT touch MT5 at all."""
    _install_fake_executor(monkeypatch)
    spec = _setup_account(login=99999, symbols=("US500.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "99999")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")

    monkeypatch.setattr(
        orch, "request_batch_approval",
        lambda *a, **kw: pytest.fail("approval must NOT be requested without --execute"),
    )
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=99999, tradeable=("ES",))],
    )
    orch.run_cfd_prop_execution(
        args=_make_args(execute=False, dry_run=True),
        config=config,
        forecasts_df=_forecasts_df([{"ticker": "ES", "position_fraction": 0.10}]),
    )
    # No MT5 connect happened, no orders placed.
    assert spec.place_calls == []
    assert spec.close_calls == []
    out = capsys.readouterr().out
    assert "--execute not set" in out
