"""
End-to-end orchestration of auto-execution for the personal forecast.

Call :func:`run_auto_execution` from the forecast script *after* the Telegram
signal has been sent and ``shares_df`` has been computed. If ``--execute``
is absent this is a no-op.

Responsibilities here (and nowhere else):
- Build :class:`ExecutionConfig` from the profile JSON.
- Connect a dedicated IB trade client.
- Build :class:`OrderIntent` list via the pure rebalancer.
- Run the preflight safety gates.
- Request human approval via Telegram inline buttons (when requested).
- Place orders sequentially, cancelling the rest on first unexpected failure.
- Write audit JSONL + closing-positions snapshot for tomorrow's reconcile.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from deployment.telegram_notifier import TelegramNotifier
from execution.approval_flow import ApprovalDecision, new_run_id, request_approval
from execution.audit_log import AuditLog, write_closing_positions
from execution.ib_trade_executor import IBTradeClient
from execution.models import ExecutionConfig, OrderIntent, OrderResult, OrderStatus
from execution.order_safety import (
    PreflightContext,
    SafetyViolation,
    run_all_preflight_checks,
    write_lock_file,
)
from execution.rebalancer import compute_order_intents


logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


def _safe_print(text: str) -> None:
    """Print to stdout, falling back to ASCII if cp1252 can't encode (Windows console).

    On Windows, the default console codepage often can't encode characters like
    U+2248 ('almost equal'). Rather than crash the entire execute pipeline on
    a cosmetic glyph, strip non-encodable chars and print a transliterated form.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = (sys.stdout.encoding or "ascii")
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def run_auto_execution(
    *,
    args,
    config: Dict,
    shares_df: pd.DataFrame,
    capital: float,
    profile: str,
) -> None:
    """Entry point from the main forecast script.

    Returns silently if auto-execution is not enabled for this run.
    Never raises on safety violations — prints a Telegram alert and exits the
    auto-execute stage cleanly so the forecast signal still stands.
    """
    if not getattr(args, "execute", False):
        return
    if profile != "personal":
        print("--execute is only supported for profile=personal. Skipping.")
        return
    if "execution" not in config:
        print("Config missing 'execution' block. Skipping auto-execute.")
        return

    exec_cfg = ExecutionConfig.from_dict(config["execution"])
    run_id = new_run_id()
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "logs" / "execution"
    lock_file = log_dir / f"executed_{datetime.now().strftime('%Y-%m-%d')}.lock"
    positions_snapshot_path = log_dir / "positions_latest.json"
    audit = AuditLog(run_id=run_id, log_dir=log_dir)

    try:
        _execute(
            args=args,
            config=config,
            shares_df=shares_df,
            capital=capital,
            exec_cfg=exec_cfg,
            audit=audit,
            lock_file=lock_file,
            positions_snapshot_path=positions_snapshot_path,
            run_id=run_id,
        )
    except Exception as e:
        logger.exception("Auto-execution crashed")
        audit.log_error("run_auto_execution", str(e))
        _alert(f"🚨 Auto-execution crashed: {e}")
    finally:
        audit.log_end(status="done")


def _execute(
    *,
    args,
    config: Dict,
    shares_df: pd.DataFrame,
    capital: float,
    exec_cfg: ExecutionConfig,
    audit: AuditLog,
    lock_file: Path,
    positions_snapshot_path: Path,
    run_id: str,
) -> None:
    port = args.port or config["connection"]["port"]
    host = config["connection"]["host"]
    ib_client_id = config["execution"].get("ib_client_id", 3)
    live_flag = getattr(args, "live", False)
    dry = getattr(args, "dry_run_execute", False)

    audit.log_start(profile="personal", port=port, account=exec_cfg.ib_account_id, live=live_flag)

    # --- Build target_shares + prices from the already-computed shares_df ---
    target_shares, prices = _extract_targets_and_prices(shares_df)
    if not target_shares:
        print("\n12. Auto-execute: nothing to rebalance (empty target). Skipping.")
        audit.log_end(status="empty_target")
        return

    # --- Connect trade client ---
    print(f"\n12. Auto-execute: connecting trade client (client_id={ib_client_id})...")
    trade = IBTradeClient(host=host, port=port, client_id=ib_client_id)
    try:
        trade.connect_and_start(timeout=5.0)
        managed_accounts = trade.fetch_managed_accounts(timeout=5.0)
        print(f"    Managed accounts: {managed_accounts}")
        current_positions = {k: v for k, v in trade.fetch_positions(timeout=10.0).items() if v != 0}
        print(f"    Current positions: {current_positions or '(none)'}")

        intents = compute_order_intents(
            target_shares=target_shares,
            current_positions=current_positions,
            prices=prices,
            config=exec_cfg,
        )
        audit.log_intents(intents)
        print(f"    Order intents: {len(intents)}")
        for i in intents:
            _safe_print(f"      {i.side.value} {i.shares} {i.etf} @~${i.est_price:.2f} ~= ${i.est_notional:.2f}")

        if not intents:
            print("    Already at target. No orders to place.")
            audit.log_end(status="no_op")
            return

        # --- Preflight ---
        ctx = PreflightContext(
            ib_port=port,
            live_flag=live_flag,
            managed_accounts=managed_accounts,
            intents=intents,
            now_et=datetime.now(_ET),
            lock_file=lock_file,
            allow_rerun=getattr(args, "allow_rerun", False),
        )
        try:
            run_all_preflight_checks(ctx, exec_cfg)
            audit.log_preflight(True)
            print("    Preflight: all checks passed.")
        except SafetyViolation as e:
            audit.log_preflight(False, reason=str(e))
            _alert(f"🚨 Preflight failed: {e}")
            print(f"    SAFETY VIOLATION: {e}")
            return

        # --- Dry-run: stop here ---
        if dry:
            print("    --dry-run-execute set: not sending approval request, not placing orders.")
            audit.log_end(status="dry_run")
            return

        # --- Approval ---
        notifier = TelegramNotifier.for_personal_account()
        if getattr(args, "approve_via_telegram", False):
            approval_text = _format_approval_message(
                run_id=run_id, intents=intents, account=exec_cfg.ib_account_id,
                live=live_flag, timeout_seconds=exec_cfg.approval_timeout_seconds,
            )
            outcome = request_approval(
                notifier,
                message_text=approval_text,
                run_id=run_id,
                authorized_user_ids=exec_cfg.authorized_telegram_user_ids,
                timeout_seconds=exec_cfg.approval_timeout_seconds,
                allow_any_approver=exec_cfg.allow_any_approver,
            )
            audit.log_approval(approver_id=outcome.approver_telegram_id, decision=outcome.decision.value)
            print(f"    Approval: {outcome.decision.value}")
            if outcome.decision != ApprovalDecision.APPROVED:
                return
        else:
            print("    --approve-via-telegram not set: proceeding without human approval.")
            audit.log_approval(approver_id=None, decision=ApprovalDecision.NOT_REQUESTED.value)

        # --- Place orders sequentially ---
        results = _place_orders(
            trade=trade,
            intents=intents,
            account=exec_cfg.ib_account_id,
            fill_timeout_s=exec_cfg.order_fill_timeout_seconds,
            audit=audit,
        )

        # --- Record outcome ---
        write_lock_file(lock_file, {"run_id": run_id, "results": [r.status.value for r in results]})
        closing = _compute_closing_positions(current_positions, results)
        write_closing_positions(closing, positions_snapshot_path)

        summary = _format_execution_summary(results)
        print("\n    Execution summary:\n" + summary)
        notifier.send_message(f"*Execution complete* (run {run_id})\n\n{summary}")
        audit.log_end(status="done")

    finally:
        trade.disconnect_and_stop()


def _extract_targets_and_prices(shares_df: pd.DataFrame) -> tuple[Dict[str, Decimal], Dict[str, float]]:
    """Pull per-ETF target share count and price from the shares table.

    shares_df is the DataFrame produced by ``calculate_etf_shares`` in the
    main script; it carries ``etf``, ``etf_price``, and ``shares_fractional``
    columns keyed on the *futures* ticker. We index outputs by ETF symbol.
    """
    target_shares: Dict[str, Decimal] = {}
    prices: Dict[str, float] = {}
    if shares_df.empty:
        return target_shares, prices
    for _, row in shares_df.iterrows():
        etf = row["etf"]
        target_shares[etf] = Decimal(str(row.get("shares_fractional", 0))).quantize(Decimal("0.0001"))
        prices[etf] = float(row["etf_price"])
    return target_shares, prices


def _format_approval_message(
    *, run_id: str, intents: List[OrderIntent], account: str, live: bool, timeout_seconds: int,
) -> str:
    mode = "LIVE" if live else "PAPER"
    lines = [
        f"🔔 *EXECUTION APPROVAL* (run `{run_id}`)",
        f"Account: `{account}` ({mode})",
        "",
        "Orders to place:",
    ]
    total = 0.0
    for i in intents:
        lines.append(f"• {i.side.value} {i.shares} {i.etf} @~${i.est_price:.2f} ≈ ${i.est_notional:.2f}")
        total += i.est_notional
    lines += [
        "",
        f"Batch: {len(intents)} orders, ${total:.2f} notional",
        f"Expires in {timeout_seconds // 60} min. Tap below.",
    ]
    return "\n".join(lines)


def _place_orders(
    *,
    trade: IBTradeClient,
    intents: List[OrderIntent],
    account: str,
    fill_timeout_s: int,
    audit: AuditLog,
) -> List[OrderResult]:
    results: List[OrderResult] = []
    for idx, intent in enumerate(intents):
        order_id = trade.submit_order(intent, account=account)
        audit.log_order_placed(intent=intent, ib_order_id=order_id)
        result = trade.wait_for_terminal(order_id, timeout_seconds=fill_timeout_s)
        audit.log_order_result(result)
        results.append(result)
        print(f"      [{idx+1}/{len(intents)}] {intent.etf}: {result.status.value} "
              f"filled={result.filled_shares} @ {result.avg_fill_price}")

        if result.status in {OrderStatus.REJECTED, OrderStatus.ERROR, OrderStatus.TIMED_OUT}:
            remaining = intents[idx + 1:]
            if remaining:
                msg = f"⚠ {intent.etf} ended {result.status.value}; cancelling remaining {len(remaining)} orders."
                logger.warning(msg)
                print(f"      {msg}")
                break
    return results


def _compute_closing_positions(
    start: Dict[str, Decimal],
    results: List[OrderResult],
) -> Dict[str, Decimal]:
    closing = dict(start)
    for r in results:
        if r.status != OrderStatus.FILLED:
            continue
        signed = r.filled_shares if r.intent.side.value == "BUY" else -r.filled_shares
        closing[r.intent.etf] = closing.get(r.intent.etf, Decimal("0")) + signed
    # Drop zeroed positions for cleanliness.
    return {k: v for k, v in closing.items() if v != 0}


def _format_execution_summary(results: List[OrderResult]) -> str:
    lines = []
    for r in results:
        line = (
            f"{r.intent.side.value} {r.intent.shares} {r.intent.etf}: "
            f"{r.status.value} | filled {r.filled_shares} @ {r.avg_fill_price}"
        )
        if r.error_text:
            line += f" | err {r.error_code}: {r.error_text}"
        lines.append(line)
    return "\n".join(lines) or "(no results)"


def _alert(text: str) -> None:
    """Best-effort Telegram alert on serious errors."""
    try:
        TelegramNotifier.for_personal_account().send_message(text)
    except Exception:
        logger.exception("Failed to send alert")
