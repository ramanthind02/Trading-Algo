"""
Weekend-close orchestrator for the CFD prop-firm mode.

Called by ``scripts/enigma_cfd_prop_weekend_close.py``. Closes all
positions matching our magic number across every enabled MT5 account in
``configs/live_forecast_config_cfd_prop.json``, with the same Telegram
batch-approval flow as the daily rebalance.

Why this is a separate script (not a flag on the rebalance script):

- Different schedule (cron Friday 15:50 ET vs daily 15:30 ET).
- Different *semantics* — never opens, never sizes, never reads vault
  forecasts; just walks our magic-tagged positions and closes them.
- Operators may opt-in to weekend-close without changing the rebalance
  cadence (e.g. weekday rebalance on, weekend close off, then on later).

Reuses 90% of the daily-rebalance orchestrator's plumbing:

- Same JSON config + :func:`execution.run_mt5_execution._account_from_config`.
- Same :class:`AccountPlan` / :class:`AccountExecutionReport` models.
- Same :class:`TelegramNotifier.for_cfd_prop` channel.
- Same :func:`execution.approval_flow.request_batch_approval` UX.
- Same audit-log directory layout (filename prefix differentiates run type).

The forecast pipeline is **not** invoked — this is a pure
positions-fetch → close-all flow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from deployment.telegram_notifier import TelegramNotifier
from execution.approval_flow import (
    APPROVED_DECISIONS,
    ApprovalDecision,
    BatchApprovalOutcome,
    new_run_id,
    request_batch_approval,
)
from execution.mt5_models import (
    AccountExecutionReport,
    AccountPlan,
    OrderResult,
    RebalanceAction,
    RebalanceActionKind,
)
from execution.mt5_order_safety import (
    check_account_login_matches,
    check_action_compatible_with_trade_mode,
    check_symbol_tradeable,
    check_trade_allowed,
)
from execution.run_mt5_execution import (
    AccountConfig,
    _account_from_config,
    _write_audit_log,
    render_post_execution_summary,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan-build phase
# ---------------------------------------------------------------------------


def _build_close_plan(account: AccountConfig) -> AccountPlan:
    """Connect to ``account``, list our-magic positions, build CLOSE_TICKET
    actions for each. No vault forecast, no sizing — just close everything.

    Returns an :class:`AccountPlan` with ``preflight_ok=False`` on any
    connectivity / credential failure so the user sees it in the batch
    approval message rather than the script silently skipping the account.
    """
    username_raw = os.environ.get(account.username_env)
    password_raw = os.environ.get(account.password_env)
    server_raw = os.environ.get(account.server_env)
    missing: List[str] = []
    if not username_raw:
        missing.append(account.username_env)
    if not password_raw:
        missing.append(account.password_env)
    if not server_raw:
        missing.append(account.server_env)
    if missing:
        return AccountPlan(
            label=account.label,
            login=0,
            server="",
            currency="",
            balance=0.0,
            equity=0.0,
            margin_free=0.0,
            actions_by_symbol={},
            preflight_ok=False,
            preflight_messages=[f"Missing env var(s): {', '.join(missing)}"],
        )
    try:
        login = int(username_raw)
    except ValueError:
        return AccountPlan(
            label=account.label,
            login=0,
            server=server_raw,
            currency="",
            balance=0.0,
            equity=0.0,
            margin_free=0.0,
            actions_by_symbol={},
            preflight_ok=False,
            preflight_messages=[
                f"{account.username_env}={username_raw!r} is not an integer login"
            ],
        )

    from execution.mt5_trade_executor import MT5TradeExecutor

    try:
        with MT5TradeExecutor.connect_and_verify(
            login=login,
            password=password_raw,
            server=server_raw,
            terminal_path=account.terminal_path,
        ) as mt5x:
            account_info = mt5x.fetch_account_info()
            all_positions = mt5x.fetch_all_positions()

            # Filter to only positions tagged with our magic — never touch
            # other strategies' trades on the same account.
            our_positions = [p for p in all_positions if p.magic == account.magic_number]

            # Group by symbol → sorted CLOSE_TICKET actions (oldest first
            # so the on-screen list reads chronologically).
            actions_by_symbol: Dict[str, List[RebalanceAction]] = {}
            for p in sorted(our_positions, key=lambda x: x.open_time_utc):
                actions_by_symbol.setdefault(p.symbol, []).append(
                    RebalanceAction(
                        kind=RebalanceActionKind.CLOSE_TICKET,
                        symbol=p.symbol,
                        side=p.side,
                        volume=p.volume,
                        ticket=p.ticket,
                        reason="weekend close",
                    )
                )

            # Preflight: only the gates that make sense for a pure close.
            # Skip margin / DLL gates — closes never increase exposure.
            # Skip rebalance dead-bands — we close every position fully.
            failures: List[str] = []
            msg = check_account_login_matches(
                account_info=account_info,
                expected_login=login,
                expected_server=server_raw,
            )
            if msg:
                failures.append(msg)
            else:
                msg = check_trade_allowed(account_info)
                if msg:
                    failures.append(msg)

            # Per-symbol gates: even for closes we need the symbol to be
            # tradeable (broker may have flagged it) and the close action
            # compatible with trade_mode (e.g. DISABLED blocks everything).
            for symbol_name, actions in actions_by_symbol.items():
                try:
                    sym_info = mt5x.fetch_symbol_info(symbol_name)
                except Exception as e:
                    failures.append(f"{symbol_name}: could not fetch symbol info: {e}")
                    continue
                msg = check_symbol_tradeable(sym_info)
                if msg:
                    failures.append(f"{symbol_name}: {msg}")
                    continue
                for a in actions:
                    msg = check_action_compatible_with_trade_mode(a, sym_info)
                    if msg:
                        failures.append(f"{symbol_name} ticket #{a.ticket}: {msg}")

            preflight_ok = not failures
            return AccountPlan(
                label=account.label,
                login=account_info.login,
                server=account_info.server,
                currency=account_info.currency,
                balance=account_info.balance,
                equity=account_info.equity,
                margin_free=account_info.margin_free,
                actions_by_symbol=actions_by_symbol,
                preflight_ok=preflight_ok,
                preflight_messages=failures,
            )

    except Exception as e:
        logger.exception("Account %s weekend-close plan build failed", account.label)
        return AccountPlan(
            label=account.label,
            login=login,
            server=server_raw or "",
            currency="",
            balance=0.0,
            equity=0.0,
            margin_free=0.0,
            actions_by_symbol={},
            preflight_ok=False,
            preflight_messages=[f"plan build failed: {type(e).__name__}: {e}"],
        )


# ---------------------------------------------------------------------------
# Telegram message rendering
# ---------------------------------------------------------------------------


def render_weekend_close_approval_text(
    *,
    plans: Sequence[AccountPlan],
    timeout_seconds: int,
) -> str:
    lines: List[str] = []
    lines.append("*ENIGMA CFD PROP — WEEKEND CLOSE APPROVAL*")
    lines.append(f"_{datetime.now().strftime('%Y-%m-%d %H:%M PST')}_")
    lines.append("")
    lines.append("Closing all magic-tagged positions across the listed accounts.")
    lines.append("")
    for plan in plans:
        flag = "✅" if plan.preflight_ok else "🚫"
        lines.append(f"{flag} *{plan.label}* — login {plan.login} @ {plan.server}")
        if plan.preflight_ok:
            lines.append(
                f"   equity ${plan.equity:,.2f} {plan.currency}  "
                f"free ${plan.margin_free:,.2f}"
            )
        for msg in plan.preflight_messages:
            lines.append(f"   ⚠ {msg}")
        symbols = sorted(plan.actions_by_symbol.keys())
        for sym in symbols:
            for a in plan.actions_by_symbol[sym]:
                side = (a.side.value if a.side else "?").upper()
                lines.append(
                    f"  - CLOSE ticket #{a.ticket} ({side} {a.volume} lots {sym})"
                )
        if not symbols:
            lines.append("   (no open positions; nothing to close)")
        lines.append("")
    lines.append(
        f"_Approve all / Cancel all / Cancel <label> — auto-decision in {timeout_seconds // 60} min._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _execute_close_plan(
    *,
    account: AccountConfig,
    plan: AccountPlan,
) -> AccountExecutionReport:
    """Reconnect to the account and submit every CLOSE_TICKET action."""
    if not plan.actions_by_symbol:
        return AccountExecutionReport(
            label=plan.label,
            login=plan.login,
            decision=ApprovalDecision.APPROVED.value,
            plan=plan,
            results=[],
            error=None,
        )

    from execution.mt5_trade_executor import MT5TradeExecutor

    username_raw = os.environ.get(account.username_env) or ""
    password_raw = os.environ.get(account.password_env) or ""
    server_raw = os.environ.get(account.server_env) or ""
    try:
        login = int(username_raw)
    except ValueError:
        return AccountExecutionReport(
            label=plan.label,
            login=plan.login,
            decision=ApprovalDecision.APPROVED.value,
            plan=plan,
            results=[],
            error=f"invalid login in env: {username_raw!r}",
        )

    results: List[OrderResult] = []
    try:
        with MT5TradeExecutor.connect_and_verify(
            login=login,
            password=password_raw,
            server=server_raw,
            terminal_path=account.terminal_path,
        ) as mt5x:
            for symbol_name in sorted(plan.actions_by_symbol.keys()):
                for action in plan.actions_by_symbol[symbol_name]:
                    if (
                        action.kind is RebalanceActionKind.CLOSE_TICKET
                        and action.ticket
                        and action.volume
                        and action.side
                    ):
                        results.append(
                            mt5x.close_ticket(
                                ticket=int(action.ticket),
                                symbol_name=symbol_name,
                                original_side=action.side,
                                volume=float(action.volume),
                                magic=account.magic_number,
                            )
                        )
        return AccountExecutionReport(
            label=plan.label,
            login=plan.login,
            decision=ApprovalDecision.APPROVED.value,
            plan=plan,
            results=results,
            error=None,
        )
    except Exception as e:
        logger.exception("Weekend-close execution failed for %s", account.label)
        return AccountExecutionReport(
            label=plan.label,
            login=plan.login,
            decision=ApprovalDecision.APPROVED.value,
            plan=plan,
            results=results,
            error=f"execution exception: {type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_cfd_prop_weekend_close(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> None:
    """Top-level entry. Called from ``scripts/enigma_cfd_prop_weekend_close.py``.

    Args:
        args: argparse namespace with ``execute``, ``dry_run``,
            ``dry_run_execute``, ``approve_via_telegram`` flags.
        config: parsed ``configs/live_forecast_config_cfd_prop.json``.
    """
    is_dry_run = bool(getattr(args, "dry_run", False))
    dry_run_execute = bool(getattr(args, "dry_run_execute", False))
    require_approval = bool(getattr(args, "approve_via_telegram", True))

    accounts_raw = config.get("accounts") or []
    global_mt5 = config.get("mt5") or {}
    execution_cfg = config.get("execution") or {}
    top_level_tradeable: Sequence[str] = tuple(config.get("tradeable_tickers") or [])
    top_level_instruments: Dict[str, Dict[str, Any]] = dict(config.get("instruments") or {})

    accounts = [
        _account_from_config(
            account_cfg=acct,
            global_mt5=global_mt5,
            execution_cfg=execution_cfg,
            top_level_tradeable=top_level_tradeable,
            top_level_instruments=top_level_instruments,
        )
        for acct in accounts_raw
    ]
    enabled = [a for a in accounts if a.enabled]
    if not enabled:
        print("\nWeekend close: no enabled accounts in config; skipping.")
        return

    if not getattr(args, "execute", False):
        print(
            "\nWeekend close: --execute not set; printing accounts that WOULD be "
            "processed and exiting. Pass --execute --dry-run-execute to build "
            "plans without sending orders, or --execute --approve-via-telegram "
            "for the full flow."
        )
        for a in enabled:
            print(f"   - {a.label} (login env: {a.username_env})")
        return

    if require_approval and not is_dry_run:
        notifier = TelegramNotifier.for_cfd_prop()
    else:
        notifier = None

    print(f"\nWeekend close: building close plans for {len(enabled)} account(s)...")
    plans: List[AccountPlan] = []
    for account in enabled:
        print(f"   - {account.label}: connecting MT5...")
        plan = _build_close_plan(account)
        n_tickets = sum(len(a) for a in plan.actions_by_symbol.values())
        if plan.preflight_ok:
            print(
                f"     ✓ equity ${plan.equity:,.2f} {plan.currency}, "
                f"{n_tickets} ticket(s) to close"
            )
        else:
            for msg in plan.preflight_messages:
                print(f"     ✗ {msg}")
        plans.append(plan)

    timeout_seconds = int(execution_cfg.get("approval_timeout_seconds", 300))
    run_id = new_run_id()
    approval_text = render_weekend_close_approval_text(
        plans=plans,
        timeout_seconds=timeout_seconds,
    )
    print("\nBatch approval message:")
    print("-" * 60)
    print(approval_text)
    print("-" * 60)

    default_on_timeout_str = str(execution_cfg.get("default_on_timeout", "cancel")).lower()
    default_on_timeout = (
        ApprovalDecision.APPROVED if default_on_timeout_str == "approve"
        else ApprovalDecision.CANCELLED
    )

    if dry_run_execute or is_dry_run:
        print("\ndry-run-execute / dry-run set; SKIPPING approval and execution.")
        return

    # Accounts with nothing to close are NOT eligible for approval (no-op).
    eligible_labels = [
        p.label
        for p in plans
        if p.preflight_ok and p.actions_by_symbol
    ]
    if not eligible_labels:
        print("\nNo accounts have positions to close; nothing to approve. Done.")
        if notifier and notifier.is_configured():
            notifier.send_message(
                "*ENIGMA CFD PROP — WEEKEND CLOSE*\n"
                "All accounts already flat; nothing to close."
            )
        return

    if require_approval:
        if notifier is None or not notifier.is_configured():
            print(
                "\n⚠ Telegram notifier not configured; refusing to execute "
                "weekend close without approval. Set TELEGRAM_CFD_PROP_BOT_TOKEN "
                "/ TELEGRAM_CFD_PROP_CHAT_ID, or pass --dry-run-execute."
            )
            return
        authorized_user_ids = [
            int(uid) for uid in execution_cfg.get("authorized_telegram_user_ids") or []
        ]
        allow_any_approver = bool(execution_cfg.get("allow_any_approver", False))
        outcome = request_batch_approval(
            notifier,
            message_text=approval_text,
            run_id=run_id,
            account_labels=eligible_labels,
            authorized_user_ids=authorized_user_ids,
            timeout_seconds=timeout_seconds,
            poll_chunk_seconds=min(25, max(5, timeout_seconds // 12)),
            allow_any_approver=allow_any_approver,
            default_on_timeout=default_on_timeout,
        )
    else:
        # --execute without --approve-via-telegram → auto-approve.
        outcome = BatchApprovalOutcome(
            global_decision=ApprovalDecision.APPROVED,
            per_account={label: ApprovalDecision.APPROVED for label in eligible_labels},
        )

    print(f"\nGlobal decision: {outcome.global_decision.value}")
    for label, dec in outcome.per_account.items():
        print(f"     {label}: {dec.value}")

    reports: List[AccountExecutionReport] = []
    plan_by_label = {p.label: p for p in plans}
    account_by_label = {a.label: a for a in enabled}

    # Surface plans that weren't approved (preflight-failed or no positions
    # to close) so the audit log + Telegram summary cover every account.
    for plan in plans:
        if plan.label not in outcome.per_account:
            reports.append(
                AccountExecutionReport(
                    label=plan.label,
                    login=plan.login,
                    decision=(
                        ApprovalDecision.CANCELLED.value
                        if not plan.preflight_ok
                        else ApprovalDecision.NOT_REQUESTED.value
                    ),
                    plan=plan,
                    results=[],
                    error=(
                        "; ".join(plan.preflight_messages)
                        if not plan.preflight_ok
                        else "no positions to close"
                    ),
                )
            )

    for label, decision in outcome.per_account.items():
        plan = plan_by_label.get(label)
        account = account_by_label.get(label)
        if plan is None or account is None:
            continue
        if decision not in APPROVED_DECISIONS or not plan.preflight_ok:
            reports.append(
                AccountExecutionReport(
                    label=label,
                    login=plan.login,
                    decision=decision.value,
                    plan=plan,
                    results=[],
                )
            )
            continue
        n_tickets = sum(len(a) for a in plan.actions_by_symbol.values())
        print(f"\nExecuting close for {label} ({n_tickets} ticket(s))...")
        report = _execute_close_plan(account=account, plan=plan)
        reports.append(
            AccountExecutionReport(
                label=report.label,
                login=report.login,
                decision=decision.value,
                plan=report.plan,
                results=report.results,
                error=report.error,
            )
        )
        for r in report.results:
            mark = "✓" if r.success else "✗"
            print(
                f"     {mark} CLOSE #{r.action.ticket} {r.action.symbol} "
                f"vol={r.action.volume} retcode={r.retcode}"
            )

    summary_text = render_post_execution_summary(reports)
    print("\nSummary:")
    print("-" * 60)
    print(summary_text)
    print("-" * 60)
    if notifier and notifier.is_configured():
        notifier.send_message(summary_text)

    # Audit log — same dir as daily rebalance but filename prefix differentiates.
    audit_dir = Path(execution_cfg.get("audit_dir") or "logs/cfd_prop_audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"cfd_prop_weekend_{run_id}_{int(time.time())}.json"
    # Reuse _write_audit_log's internal serialisation, then rename for clarity.
    written = _write_audit_log(
        audit_dir=audit_dir,
        run_id=run_id,
        plans=plans,
        reports=reports,
        decision_outcome=outcome,
    )
    try:
        written.rename(audit_path)
        final_audit = audit_path
    except OSError:
        final_audit = written
    print(f"\nAudit log written to {final_audit}")
