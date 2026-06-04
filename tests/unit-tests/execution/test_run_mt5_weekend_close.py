"""Unit tests for :mod:`execution.run_mt5_weekend_close`.

Tests the weekend-close orchestrator end-to-end with a mocked MT5 layer.
Mirrors ``tests/unit-tests/execution/test_run_mt5_execution.py`` patterns
(same fakes / fixtures style) but for the close-everything flow.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from execution import run_mt5_weekend_close as wc
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
    login: int
    password: str
    server: str
    terminal_path: Optional[str] = None
    timeout_ms: int = 60_000
    close_calls: List[Dict[str, Any]] = field(default_factory=list)
    account_info: Optional[MT5AccountInfo] = None
    positions: List[MT5Position] = field(default_factory=list)
    symbol_infos: Dict[str, MT5SymbolInfo] = field(default_factory=dict)
    close_success: bool = True
    close_retcode: int = 10009

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

    def fetch_account_info(self) -> MT5AccountInfo:
        assert self.account_info is not None
        return self.account_info

    def fetch_all_positions(self) -> List[MT5Position]:
        return list(self.positions)

    def fetch_symbol_info(self, name: str) -> MT5SymbolInfo:
        return self.symbol_infos[name]

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
            success=self.close_success,
            deal_ticket=888_000 + len(self.close_calls),
            retcode=self.close_retcode,
            comment="ok" if self.close_success else "rejected",
        )


_FAKE_REGISTRY: Dict[int, _FakeMT5Executor] = {}


def _install_fake_executor(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _sym(name: str, *, trade_mode: TradeMode = TradeMode.FULL) -> MT5SymbolInfo:
    return MT5SymbolInfo(
        name=name,
        trade_mode=trade_mode,
        point=0.01,
        digits=2,
        trade_contract_size=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread=10,
        visible=True,
    )


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


def _pos(ticket: int, symbol: str, side: OrderSide, volume: float, *,
         magic: int = 90420, when: Optional[datetime] = None) -> MT5Position:
    return MT5Position(
        ticket=ticket,
        symbol=symbol,
        side=side,
        volume=volume,
        open_price=18000.0,
        magic=magic,
        open_time_utc=when or T0,
    )


def _make_args(
    *,
    dry_run: bool = False,
    dry_run_execute: bool = False,
    approve_via_telegram: bool = True,
    execute: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        dry_run=dry_run,
        dry_run_execute=dry_run_execute,
        approve_via_telegram=approve_via_telegram,
        execute=execute,
    )


def _setup_account(
    *,
    login: int,
    positions: Optional[List[MT5Position]] = None,
    symbols: Sequence[str] = ("US100.cash",),
) -> _FakeMT5Executor:
    sym_map = {s: _sym(s) for s in symbols}
    spec = _FakeMT5Executor(
        login=login,
        password="",
        server="FTMO-Demo",
        account_info=_acct(login),
        positions=positions or [],
        symbol_infos=sym_map,
    )
    _FAKE_REGISTRY[login] = spec
    return spec


def _account_block(*, label: str, login: int, magic: int = 90420) -> Dict[str, Any]:
    return {
        "label": label,
        "enabled": True,
        "username_env": f"MT5_{label.upper()}_USERNAME",
        "password_env": f"MT5_{label.upper()}_PASSWORD",
        "server_env": f"MT5_{label.upper()}_SERVER",
        "magic_number": magic,
        "terminal_path": None,
        "tradeable_tickers": ["ES", "NQ"],
        "symbol_overrides": {},
    }


def _make_config(
    *,
    accounts: List[Dict[str, Any]],
    audit_dir: Optional[Path] = None,
    default_on_timeout: str = "cancel",
) -> Dict[str, Any]:
    return {
        "tradeable_tickers": ["ES", "NQ", "GC", "SI"],
        "instruments": {
            "ES": {"mt5_symbol": "US500.cash"},
            "NQ": {"mt5_symbol": "US100.cash"},
            "GC": {"mt5_symbol": "XAUUSD"},
            "SI": {"mt5_symbol": "XAGUSD"},
        },
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


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    _FAKE_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_execute_flag_lists_accounts_only(monkeypatch, capsys):
    _install_fake_executor(monkeypatch)
    config = _make_config(accounts=[_account_block(label="ftmo_a", login=1)])
    wc.run_cfd_prop_weekend_close(
        args=_make_args(execute=False),
        config=config,
    )
    out = capsys.readouterr().out
    assert "WOULD be processed" in out
    assert "ftmo_a" in out


def test_no_enabled_accounts_skips_quietly(monkeypatch, capsys):
    _install_fake_executor(monkeypatch)
    config = _make_config(
        accounts=[{**_account_block(label="a", login=1), "enabled": False}],
    )
    wc.run_cfd_prop_weekend_close(
        args=_make_args(),
        config=config,
    )
    out = capsys.readouterr().out
    assert "no enabled accounts" in out


def test_account_with_no_positions_is_noop(monkeypatch, capsys, tmp_path):
    _install_fake_executor(monkeypatch)
    spec = _setup_account(login=12345, positions=[])
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "12345")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    fake_notifier = _FakeNotifier()
    monkeypatch.setattr(
        wc.TelegramNotifier, "for_cfd_prop", classmethod(lambda cls: fake_notifier)
    )
    monkeypatch.setattr(
        wc, "request_batch_approval",
        lambda *a, **kw: pytest.fail("approval should NOT be requested when flat"),
    )
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=12345)],
        audit_dir=tmp_path / "audit",
    )
    wc.run_cfd_prop_weekend_close(args=_make_args(), config=config)
    assert spec.close_calls == []
    out = capsys.readouterr().out
    assert "nothing to approve" in out.lower() or "nothing to close" in out.lower()
    assert any("already flat" in m["text"] for m in fake_notifier.sent)


def test_filters_to_our_magic_only(monkeypatch, capsys, tmp_path):
    """Positions tagged with someone else's magic must not be touched."""
    _install_fake_executor(monkeypatch)
    positions = [
        _pos(101, "US100.cash", OrderSide.BUY, 0.10, magic=90420),
        _pos(102, "US100.cash", OrderSide.SELL, 0.20, magic=12345),  # not ours
        _pos(103, "US500.cash", OrderSide.BUY, 0.30, magic=90420),
    ]
    spec = _setup_account(login=22222, positions=positions, symbols=("US100.cash", "US500.cash"))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "22222")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    fake_notifier = _FakeNotifier()
    monkeypatch.setattr(
        wc.TelegramNotifier, "for_cfd_prop", classmethod(lambda cls: fake_notifier)
    )
    monkeypatch.setattr(
        wc, "request_batch_approval",
        lambda *a, **kw: BatchApprovalOutcome(
            global_decision=ApprovalDecision.APPROVED,
            per_account={kw["account_labels"][0]: ApprovalDecision.APPROVED},
        ),
    )
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=22222, magic=90420)],
        audit_dir=tmp_path / "audit",
    )
    wc.run_cfd_prop_weekend_close(args=_make_args(), config=config)
    # Only our two tickets must be closed; #102 is invisible to us.
    closed_tickets = sorted(c["ticket"] for c in spec.close_calls)
    assert closed_tickets == [101, 103]


def test_happy_path_closes_and_summarises(monkeypatch, capsys, tmp_path):
    _install_fake_executor(monkeypatch)
    positions = [
        _pos(50, "US100.cash", OrderSide.BUY, 0.50, magic=90421, when=T0),
        _pos(51, "XAUUSD", OrderSide.SELL, 0.03, magic=90421, when=T0),
    ]
    spec = _setup_account(login=33333, positions=positions, symbols=("US100.cash", "XAUUSD"))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "33333")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    fake_notifier = _FakeNotifier()
    monkeypatch.setattr(
        wc.TelegramNotifier, "for_cfd_prop", classmethod(lambda cls: fake_notifier)
    )
    monkeypatch.setattr(
        wc, "request_batch_approval",
        lambda *a, **kw: BatchApprovalOutcome(
            global_decision=ApprovalDecision.APPROVED,
            per_account={lbl: ApprovalDecision.APPROVED for lbl in kw["account_labels"]},
        ),
    )
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=33333, magic=90421)],
        audit_dir=tmp_path / "audit",
    )
    wc.run_cfd_prop_weekend_close(args=_make_args(), config=config)

    # Both tickets closed, with original side preserved so executor sends opposite.
    assert {c["ticket"] for c in spec.close_calls} == {50, 51}
    sides_by_ticket = {c["ticket"]: c["side"] for c in spec.close_calls}
    assert sides_by_ticket[50] is OrderSide.BUY
    assert sides_by_ticket[51] is OrderSide.SELL

    # Telegram summary was sent.
    assert any("WEEKEND CLOSE" in m["text"].upper() or "EXECUTION SUMMARY" in m["text"]
               for m in fake_notifier.sent)

    # Audit log uses the weekend prefix.
    audit_files = list((tmp_path / "audit").iterdir())
    assert len(audit_files) == 1
    assert audit_files[0].name.startswith("cfd_prop_weekend_")
    # Audit payload deserialises and references our magic-tagged tickets.
    payload = json.loads(audit_files[0].read_text())
    audit_tickets = {
        act["ticket"]
        for r in payload["reports"]
        for res in r.get("results", [])
        for act in [res["action"]]
        if act.get("ticket")
    }
    assert audit_tickets == {50, 51}


def test_cancel_all_skips_execution(monkeypatch, capsys, tmp_path):
    _install_fake_executor(monkeypatch)
    positions = [_pos(70, "US100.cash", OrderSide.BUY, 0.10, magic=90420)]
    spec = _setup_account(login=44444, positions=positions, symbols=("US100.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "44444")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    fake_notifier = _FakeNotifier()
    monkeypatch.setattr(
        wc.TelegramNotifier, "for_cfd_prop", classmethod(lambda cls: fake_notifier)
    )
    monkeypatch.setattr(
        wc, "request_batch_approval",
        lambda *a, **kw: BatchApprovalOutcome(
            global_decision=ApprovalDecision.CANCELLED,
            per_account={lbl: ApprovalDecision.CANCELLED for lbl in kw["account_labels"]},
        ),
    )
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=44444)],
        audit_dir=tmp_path / "audit",
    )
    wc.run_cfd_prop_weekend_close(args=_make_args(), config=config)
    # Approval cancelled → no close_ticket calls.
    assert spec.close_calls == []


def test_dry_run_execute_builds_plans_but_does_not_close(monkeypatch, capsys, tmp_path):
    _install_fake_executor(monkeypatch)
    positions = [_pos(80, "US100.cash", OrderSide.BUY, 0.10, magic=90420)]
    spec = _setup_account(login=55555, positions=positions, symbols=("US100.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "55555")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    monkeypatch.setattr(
        wc, "request_batch_approval",
        lambda *a, **kw: pytest.fail("approval should NOT be requested in dry-run-execute"),
    )
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=55555)],
        audit_dir=tmp_path / "audit",
    )
    wc.run_cfd_prop_weekend_close(
        args=_make_args(dry_run_execute=True),
        config=config,
    )
    # Plan was built (we connected), but no actual close was sent.
    assert spec.close_calls == []
    out = capsys.readouterr().out
    assert "SKIPPING approval and execution" in out


def test_missing_creds_yields_preflight_failure(monkeypatch, capsys, tmp_path):
    _install_fake_executor(monkeypatch)
    _setup_account(login=66666, symbols=("US100.cash",))
    # Note: do NOT set env vars.
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=66666)],
        audit_dir=tmp_path / "audit",
    )
    monkeypatch.setattr(
        wc, "request_batch_approval",
        lambda *a, **kw: pytest.fail("approval should NOT be requested when ineligible"),
    )
    wc.run_cfd_prop_weekend_close(
        args=_make_args(approve_via_telegram=False),
        config=config,
    )
    out = capsys.readouterr().out
    assert "Missing env var(s)" in out
    assert "MT5_FTMO_A_USERNAME" in out


def test_no_approve_flag_auto_approves_and_executes(monkeypatch, capsys, tmp_path):
    """--execute without --approve-via-telegram auto-approves (matches the
    daily-rebalance script's behaviour after the gate-stripping refactor)."""
    _install_fake_executor(monkeypatch)
    positions = [_pos(90, "US100.cash", OrderSide.BUY, 0.10, magic=90420)]
    spec = _setup_account(login=77777, positions=positions, symbols=("US100.cash",))
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "77777")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    monkeypatch.setattr(
        wc, "request_batch_approval",
        lambda *a, **kw: pytest.fail("approval should NOT be requested"),
    )
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=77777)],
        audit_dir=tmp_path / "audit",
    )
    wc.run_cfd_prop_weekend_close(
        args=_make_args(approve_via_telegram=False),
        config=config,
    )
    # Auto-approved → close was actually sent.
    assert len(spec.close_calls) == 1
    assert spec.close_calls[0]["ticket"] == 90


def test_render_approval_text_lists_tickets_and_labels() -> None:
    from execution.mt5_models import AccountPlan

    plan = AccountPlan(
        label="ftmo_a",
        login=12345,
        server="FTMO-Demo",
        currency="USD",
        balance=100_000.0,
        equity=100_500.0,
        margin_free=99_000.0,
        actions_by_symbol={
            "US100.cash": [
                RebalanceAction(
                    kind=RebalanceActionKind.CLOSE_TICKET,
                    symbol="US100.cash",
                    side=OrderSide.BUY,
                    volume=0.50,
                    ticket=999,
                    reason="weekend close",
                ),
            ],
        },
        preflight_ok=True,
        preflight_messages=[],
    )
    text = wc.render_weekend_close_approval_text(plans=[plan], timeout_seconds=300)
    assert "WEEKEND CLOSE APPROVAL" in text
    assert "ftmo_a" in text
    assert "ticket #999" in text
    assert "BUY" in text
    assert "auto-decision in 5 min" in text


def test_render_approval_text_handles_flat_account_with_no_actions() -> None:
    from execution.mt5_models import AccountPlan

    plan = AccountPlan(
        label="ftmo_flat",
        login=98765,
        server="FTMO-Demo",
        currency="USD",
        balance=100_000.0,
        equity=100_000.0,
        margin_free=100_000.0,
        actions_by_symbol={},
        preflight_ok=True,
        preflight_messages=[],
    )
    text = wc.render_weekend_close_approval_text(plans=[plan], timeout_seconds=300)
    assert "ftmo_flat" in text
    assert "nothing to close" in text


def test_preflight_failure_blocks_close_for_that_account(monkeypatch, capsys, tmp_path):
    """Login mismatch in fetched account info must be surfaced and the
    account skipped (but other accounts in the batch still proceed)."""
    _install_fake_executor(monkeypatch)
    positions = [_pos(11, "US100.cash", OrderSide.BUY, 0.10, magic=90420)]
    spec = _setup_account(login=88888, positions=positions, symbols=("US100.cash",))
    # Patch the account_info so login doesn't match what env var says.
    spec.account_info = MT5AccountInfo(
        login=99999,  # mismatched!
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
    monkeypatch.setenv("MT5_FTMO_A_USERNAME", "88888")
    monkeypatch.setenv("MT5_FTMO_A_PASSWORD", "pw")
    monkeypatch.setenv("MT5_FTMO_A_SERVER", "FTMO-Demo")
    monkeypatch.setattr(
        wc, "request_batch_approval",
        lambda *a, **kw: pytest.fail("approval should NOT be requested"),
    )
    config = _make_config(
        accounts=[_account_block(label="ftmo_a", login=88888)],
        audit_dir=tmp_path / "audit",
    )
    wc.run_cfd_prop_weekend_close(args=_make_args(), config=config)
    # Preflight failed → no close calls.
    assert spec.close_calls == []
    out = capsys.readouterr().out
    assert "no accounts have positions to close" in out.lower() or \
           "nothing to approve" in out.lower()
