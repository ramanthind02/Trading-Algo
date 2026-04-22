"""
Pre-flight safety gates for ETF auto-execution.

Every gate is a pure function: inputs in, ``None`` on pass,
:class:`SafetyViolation` on fail. Gates never sleep, call the network, or
touch IB. Callers wire them together via :func:`run_all_preflight_checks`.

Design intent: each rule is obvious on inspection. If a rule depends on
state outside its inputs, pass that state in. No hidden globals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from execution.models import ExecutionConfig, OrderIntent


class SafetyViolation(RuntimeError):
    """Raised when any pre-flight check fails. Aborts the batch."""


@dataclass(frozen=True)
class PreflightContext:
    """Inputs the pre-flight gates need. Built once at the top of execute()."""

    ib_port: int
    live_flag: bool
    managed_accounts: List[str]
    vault_root: str
    current_positions: Dict[str, Decimal]
    intents: List[OrderIntent]
    capital_usd: float
    max_position_pct: float
    now_et: datetime
    lock_file: Path
    allow_rerun: bool = False
    skip_reconciliation: bool = False
    expected_prior_positions: Optional[Dict[str, Decimal]] = None


_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------

def check_lock_file(ctx: PreflightContext) -> None:
    if ctx.lock_file.exists() and not ctx.allow_rerun:
        raise SafetyViolation(
            f"Lock file exists: {ctx.lock_file}. Today's run has already executed. "
            f"Pass --allow-rerun to override."
        )


def check_live_port_requires_flag(ctx: PreflightContext) -> None:
    if ctx.ib_port == 7496 and not ctx.live_flag:
        raise SafetyViolation(
            "Connected to IB port 7496 (LIVE) without --live. Refusing to place orders."
        )


def check_account_match(ctx: PreflightContext, config: ExecutionConfig) -> None:
    expected = config.ib_account_id
    if not expected or expected.startswith("REPLACE"):
        raise SafetyViolation(
            "execution.ib_account_id is not set. Refusing to place orders."
        )
    if expected not in ctx.managed_accounts:
        raise SafetyViolation(
            f"Expected account {expected} not in IB managedAccounts {ctx.managed_accounts}. "
            f"Refusing to place orders."
        )


def check_vault_root_is_personal(ctx: PreflightContext) -> None:
    if "personal" not in ctx.vault_root:
        raise SafetyViolation(
            f"vault_root is '{ctx.vault_root}', not vault_personal. "
            f"Auto-execution is only supported on the personal profile."
        )


def check_market_hours(ctx: PreflightContext, config: ExecutionConfig) -> None:
    open_t = _parse_hhmm(config.market_open_et)
    close_t = _parse_hhmm(config.market_close_et)
    now_t = ctx.now_et.astimezone(_ET).time()
    if not (open_t <= now_t <= close_t):
        raise SafetyViolation(
            f"Now {now_t} ET is outside trading window "
            f"{config.market_open_et}-{config.market_close_et} ET."
        )


def check_reconciliation(ctx: PreflightContext) -> None:
    if ctx.skip_reconciliation or ctx.expected_prior_positions is None:
        return
    eps = Decimal("0.0001")
    for etf, expected in ctx.expected_prior_positions.items():
        actual = ctx.current_positions.get(etf, Decimal("0"))
        if abs(actual - expected) > eps:
            raise SafetyViolation(
                f"Reconciliation failed: {etf} expected {expected}, actual {actual}. "
                f"Pass --skip-reconciliation after manual verification."
            )


def check_per_order_notional(ctx: PreflightContext, config: ExecutionConfig) -> None:
    for intent in ctx.intents:
        if intent.est_notional > config.max_order_notional_usd:
            raise SafetyViolation(
                f"Order for {intent.etf} notional ${intent.est_notional:.2f} "
                f"exceeds max_order_notional_usd ${config.max_order_notional_usd:.2f}."
            )


def check_batch_notional(ctx: PreflightContext, config: ExecutionConfig) -> None:
    total = sum(i.est_notional for i in ctx.intents)
    if total > config.max_batch_notional_usd:
        raise SafetyViolation(
            f"Batch notional ${total:.2f} exceeds "
            f"max_batch_notional_usd ${config.max_batch_notional_usd:.2f}."
        )


def check_position_cap(ctx: PreflightContext) -> None:
    """Resulting position notional must not exceed max_position_pct of capital."""
    cap_usd = ctx.capital_usd * (ctx.max_position_pct / 100.0)
    for intent in ctx.intents:
        current = ctx.current_positions.get(intent.etf, Decimal("0"))
        signed_delta = intent.shares if intent.side.value == "BUY" else -intent.shares
        resulting = current + signed_delta
        resulting_notional = abs(float(resulting)) * intent.est_price
        if resulting_notional > cap_usd:
            raise SafetyViolation(
                f"Resulting {intent.etf} position ${resulting_notional:.2f} "
                f"exceeds position cap ${cap_usd:.2f} ({ctx.max_position_pct}% of capital)."
            )


def check_max_orders_per_run(ctx: PreflightContext, config: ExecutionConfig) -> None:
    if len(ctx.intents) > config.max_orders_per_run:
        raise SafetyViolation(
            f"Intent count {len(ctx.intents)} exceeds max_orders_per_run "
            f"{config.max_orders_per_run}."
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_preflight_checks(ctx: PreflightContext, config: ExecutionConfig) -> None:
    """Run every gate in order. Raises on first failure."""
    check_lock_file(ctx)
    check_live_port_requires_flag(ctx)
    check_account_match(ctx, config)
    check_vault_root_is_personal(ctx)
    check_market_hours(ctx, config)
    check_reconciliation(ctx)
    check_per_order_notional(ctx, config)
    check_batch_notional(ctx, config)
    check_position_cap(ctx)
    check_max_orders_per_run(ctx, config)


def write_lock_file(lock_file: Path, payload: dict) -> None:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(json.dumps(payload, indent=2, default=str))


def _parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))
