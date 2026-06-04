#!/usr/bin/env python3
"""
Enigma CFD Prop Weekend-Close Entrypoint
=========================================

Closes ALL magic-tagged positions across every enabled MT5 account in
``configs/live_forecast_config_cfd_prop.json`` and posts a Telegram
batch-approval message before sending close orders.

This is a SEPARATE script from the daily rebalance (different schedule,
different semantics — pure close-only, never reads the vault). Wire it
to your scheduler (Windows Task Scheduler / cron) to run on Friday
afternoon before the broker's swap cut-off, e.g.

    Friday 15:30 ET (US Eastern)

That leaves a 30-minute buffer before the US cash session close
(16:00 ET) — comfortable cushion in case the script needs a retry —
and ~90 minutes before FTMO's broker midnight swap charge
(~17:00 ET year-round).

**Recommended default: close before weekend.** Gap risk over the
non-trading window is uncompensated, weekend swap fees stack up, and
the daily rebalance script natively handles re-entering positions on
flat accounts (via ``OPEN_NEW`` actions in ``mt5_rebalancer``), so
Monday's run will re-establish whatever positions the forecast still
prefers. Users who want to A/B with hold-through-weekend simply
don't schedule this script.

**Friday scheduling**: run THIS script INSTEAD of the daily rebalance
on Fridays (don't run both — they'd race). Daily rebalance Mon-Thu;
this on Fri.

The forecast pipeline is **not** invoked — this is a pure
read-positions → close-tagged-positions flow. The same Telegram
channel + approver whitelist as the daily rebalance is used.

Usage::

    # No-op preview (lists accounts that would be processed):
    python scripts/enigma_cfd_prop_weekend_close.py

    # Connect to MT5, build close plans, NO orders:
    python scripts/enigma_cfd_prop_weekend_close.py --execute --dry-run-execute

    # Full flow with Telegram approval (default):
    python scripts/enigma_cfd_prop_weekend_close.py --execute --approve-via-telegram

    # Full flow auto-approve (skip Telegram) — only if you really mean it:
    python scripts/enigma_cfd_prop_weekend_close.py --execute --no-approve-via-telegram
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from execution.run_mt5_weekend_close import run_cfd_prop_weekend_close


DEFAULT_CONFIG = Path("configs/live_forecast_config_cfd_prop.json")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Close all magic-tagged positions across enabled MT5 accounts "
            "in the CFD prop config. Designed for Friday-afternoon execution "
            "before the FTMO swap cut-off."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to CFD prop config JSON (default: {DEFAULT_CONFIG}).",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Connect to MT5 and build close plans. Without --execute the "
            "script just lists accounts that would be processed."
        ),
    )
    p.add_argument(
        "--dry-run-execute",
        action="store_true",
        help=(
            "With --execute: connect, build plans, render approval message, "
            "but do NOT send any orders. Pairs with --execute, not --dry-run."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip MT5 connection entirely (just print the account list).",
    )
    approve_group = p.add_mutually_exclusive_group()
    approve_group.add_argument(
        "--approve-via-telegram",
        dest="approve_via_telegram",
        action="store_true",
        help="Require Telegram batch approval before closing (default).",
    )
    approve_group.add_argument(
        "--no-approve-via-telegram",
        dest="approve_via_telegram",
        action="store_false",
        help="Skip Telegram approval and auto-execute (use with caution).",
    )
    p.set_defaults(approve_via_telegram=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)
    with open(cfg_path) as f:
        config = json.load(f)

    print("=" * 60)
    print("Enigma CFD Prop -- WEEKEND CLOSE")
    print("=" * 60)
    print(f"Config: {cfg_path}")
    print(f"Execute: {args.execute}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Dry-run-execute: {args.dry_run_execute}")
    print(f"Approve via Telegram: {args.approve_via_telegram}")
    print()

    run_cfd_prop_weekend_close(args=args, config=config)


if __name__ == "__main__":
    main()
