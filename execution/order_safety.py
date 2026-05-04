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
from pathlib import Path
from typing import List
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
    intents: List[OrderIntent]
    now_et: datetime
    lock_file: Path
    allow_rerun: bool = False


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


def check_market_hours(ctx: PreflightContext, config: ExecutionConfig) -> None:
    open_t = _parse_hhmm(config.market_open_et)
    close_t = _parse_hhmm(config.market_close_et)
    now_t = ctx.now_et.astimezone(_ET).time()
    if not (open_t <= now_t <= close_t):
        raise SafetyViolation(
            f"Now {now_t} ET is outside trading window "
            f"{config.market_open_et}-{config.market_close_et} ET."
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
    check_market_hours(ctx, config)
    check_max_orders_per_run(ctx, config)


def write_lock_file(lock_file: Path, payload: dict) -> None:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(json.dumps(payload, indent=2, default=str))


def _parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))
