"""
End-to-end orchestrator for the CFD prop-firm execution flow.

Called by ``scripts/enigma_live_forecast.py`` when ``--profile cfd_prop``.

High-level flow (see ``plan.md §9``):

1.  For each enabled account in ``config["accounts"]``:
    - connect to MT5, verify login/server, fetch account info, positions,
      symbol info, ticks
    - compute per-symbol ``target_signed_lots`` and rebalance actions
    - run preflight safety gates
    - disconnect (free MT5 session — package supports only one at a time)
    - collect the per-account :class:`AccountPlan`
2.  Build a single batch approval message describing all account plans.
3.  Wait up to ``approval_timeout_seconds`` for user approval / cancel.
4.  For each approved account:
    - reconnect, execute the plan sequentially with retries
    - disconnect
5.  Post a final execution summary to Telegram.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

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
    MT5AccountInfo,
    MT5SymbolInfo,
    MT5Tick,
    OrderResult,
    OrderSide,
    RebalanceAction,
    RebalanceActionKind,
)
from execution.mt5_order_safety import run_preflight
from execution.mt5_rebalancer import (
    SizingConfig,
    compute_target_signed_lots,
    plan_account_actions,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountConfig:
    """Per-account view computed from the JSON config + global defaults.

    The on-disk schema uses ``username_env`` / ``password_env`` / ``server_env``
    (env-var NAMES, not raw values) so secrets stay out of the config; the
    actual login number / password / server name are read from ``os.environ``
    at plan-build time.
    """

    label: str
    enabled: bool
    server_env: str
    username_env: str
    password_env: str
    magic_number: int
    terminal_path: Optional[str]
    tradeable_tickers: Tuple[str, ...]
    symbol_map: Dict[str, str]                # ticker -> MT5 symbol
    sizing_basis: str                          # "equity" | "balance"
    lot_size_ceiling: float
    min_rebalance_lots: float
    min_rebalance_notional_usd: float
    max_margin_usage_pct: float
    max_daily_loss_pct: Optional[float]
    sod_balance_env_var: Optional[str]


def _account_from_config(
    *,
    account_cfg: Dict[str, Any],
    global_mt5: Dict[str, Any],
    execution_cfg: Dict[str, Any],
    top_level_tradeable: Sequence[str],
    top_level_instruments: Dict[str, Dict[str, Any]],
) -> AccountConfig:
    """Merge a single account block with the global ``mt5`` + ``execution`` defaults.

    The schema (matching ``configs/live_forecast_config_cfd_prop.json``):

    - ``label``                 : str, required, unique
    - ``enabled``               : bool, defaults False
    - ``username_env`` / ``password_env`` / ``server_env`` : env var NAMES (required)
    - ``magic_number``          : int (optional → ``mt5.default_magic_number``)
    - ``terminal_path``         : optional path to ``terminal64.exe``
    - ``symbol_overrides``      : optional ``{ticker: {mt5_symbol: "..."}}`` patching
                                  the top-level ``instruments`` map for this account
    - ``max_daily_loss_pct``    : optional per-account override; defaults to the
                                  global ``execution.max_daily_loss_pct`` (5% by default)
    - ``sod_balance_env_var``   : optional env var holding start-of-day balance
    """
    label = str(account_cfg["label"])
    # Required env-var names. Fail loudly if missing rather than failing later.
    if "username_env" not in account_cfg:
        raise KeyError(f"account {label!r}: missing 'username_env' (env var NAME)")
    if "password_env" not in account_cfg:
        raise KeyError(f"account {label!r}: missing 'password_env' (env var NAME)")
    if "server_env" not in account_cfg:
        raise KeyError(f"account {label!r}: missing 'server_env' (env var NAME)")

    overrides = dict(account_cfg.get("symbol_overrides") or {})
    # Per-account tradeable_tickers override falls back to the top-level list.
    per_account_tradeable = account_cfg.get("tradeable_tickers")
    effective_tradeable = (
        list(per_account_tradeable) if per_account_tradeable else list(top_level_tradeable)
    )
    symbol_map: Dict[str, str] = {}
    for ticker in effective_tradeable:
        if ticker in overrides and "mt5_symbol" in overrides[ticker]:
            symbol_map[ticker] = str(overrides[ticker]["mt5_symbol"])
        elif ticker in top_level_instruments and "mt5_symbol" in top_level_instruments[ticker]:
            symbol_map[ticker] = str(top_level_instruments[ticker]["mt5_symbol"])

    def gm(key: str, default: Any = None) -> Any:
        """Per-account override → global mt5 default → hard default."""
        if key in account_cfg:
            return account_cfg[key]
        return global_mt5.get(key, default)

    def ge(key: str, default: Any = None) -> Any:
        """Per-account override → global execution default → hard default."""
        if key in account_cfg:
            return account_cfg[key]
        return execution_cfg.get(key, default)

    return AccountConfig(
        label=label,
        enabled=bool(account_cfg.get("enabled", False)),
        server_env=str(account_cfg["server_env"]),
        username_env=str(account_cfg["username_env"]),
        password_env=str(account_cfg["password_env"]),
        magic_number=int(
            account_cfg.get("magic_number")
            or global_mt5.get("default_magic_number")
            or 90420
        ),
        terminal_path=account_cfg.get("terminal_path") or global_mt5.get("terminal_path"),
        tradeable_tickers=tuple(effective_tradeable),
        symbol_map=symbol_map,
        sizing_basis=str(gm("sizing_basis", "equity")),
        lot_size_ceiling=float(gm("lot_size_ceiling", 100.0)),
        min_rebalance_lots=float(ge("min_rebalance_lots", 0.0)),
        min_rebalance_notional_usd=float(ge("min_rebalance_notional_usd", 0.0)),
        max_margin_usage_pct=float(ge("max_margin_usage_pct", 0.95)),
        max_daily_loss_pct=(
            float(account_cfg["max_daily_loss_pct"])
            if account_cfg.get("max_daily_loss_pct") is not None
            else (
                float(execution_cfg["max_daily_loss_pct"])
                if execution_cfg.get("max_daily_loss_pct") is not None
                else 0.05  # default 5% (FTMO DLL); matches docs
            )
        ),
        sod_balance_env_var=account_cfg.get("sod_balance_env_var"),
    )


# ---------------------------------------------------------------------------
# Plan building (per-account)
# ---------------------------------------------------------------------------


def _build_account_plan(
    *,
    account: AccountConfig,
    forecasts: Dict[str, float],   # ticker -> position_fraction
) -> AccountPlan:
    """Connect to MT5, fetch state, compute actions, run preflight, disconnect.

    Returns an :class:`AccountPlan` whose ``preflight_ok`` flag tells the
    orchestrator whether to surface this account for approval.

    On any unrecoverable error (bad creds, missing env var, MT5 connection
    failure), returns a plan with ``preflight_ok=False`` and the error
    captured in ``preflight_messages`` so the user sees it in the batch
    approval message rather than the script silently skipping the account.
    """
    # Read creds + server name from env first; fail-closed if any missing.
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
            preflight_messages=[
                f"Missing env var(s): {', '.join(missing)}"
            ],
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

    # Late import — keep the MT5 dep out of macOS/Linux test paths.
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

            # Fetch symbol info + ticks for each tradable symbol on this account.
            symbol_infos: Dict[str, MT5SymbolInfo] = {}
            ticks: Dict[str, MT5Tick] = {}
            for ticker in account.tradeable_tickers:
                mt5_symbol = account.symbol_map.get(ticker)
                if not mt5_symbol:
                    continue
                try:
                    symbol_infos[mt5_symbol] = mt5x.fetch_symbol_info(mt5_symbol)
                    ticks[mt5_symbol] = mt5x.fetch_tick(mt5_symbol)
                except Exception as e:
                    logger.warning(
                        "Failed to fetch info/tick for %s on %s: %s",
                        mt5_symbol, account.label, e,
                    )

            sizing = SizingConfig(
                sizing_basis_usd=account_info.sizing_basis_value(account.sizing_basis),
                lot_size_ceiling=account.lot_size_ceiling,
                min_rebalance_lots=account.min_rebalance_lots,
                min_rebalance_notional_usd=account.min_rebalance_notional_usd,
            )

            targets: Dict[str, Optional[float]] = {}
            for ticker in account.tradeable_tickers:
                if ticker not in forecasts:
                    continue
                mt5_symbol = account.symbol_map.get(ticker)
                if not mt5_symbol or mt5_symbol not in symbol_infos or mt5_symbol not in ticks:
                    continue
                target = compute_target_signed_lots(
                    position_fraction=float(forecasts[ticker]),
                    price=ticks[mt5_symbol].mid(),
                    symbol=symbol_infos[mt5_symbol],
                    sizing=sizing,
                )
                targets[mt5_symbol] = target

            actions_by_symbol = plan_account_actions(
                targets_signed_lots_by_symbol=targets,
                all_positions=all_positions,
                symbol_infos=symbol_infos,
                ticks=ticks,
                magic_number=account.magic_number,
                min_rebalance_lots=account.min_rebalance_lots,
                min_rebalance_notional_usd=account.min_rebalance_notional_usd,
            )

            # Sum margin requested across OPEN_NEW actions for the budget gate.
            requested_margin_usd = 0.0
            for sym_name, actions in actions_by_symbol.items():
                for a in actions:
                    if a.kind is RebalanceActionKind.OPEN_NEW and a.side is not None and a.volume:
                        try:
                            requested_margin_usd += mt5x.order_calc_margin_usd(
                                symbol_name=sym_name,
                                side=a.side,
                                volume=a.volume,
                                price=ticks[sym_name].for_side(a.side),
                            )
                        except Exception as e:
                            logger.warning("order_calc_margin failed for %s: %s", sym_name, e)

            sod_balance_usd: Optional[float] = None
            if account.sod_balance_env_var:
                raw = os.environ.get(account.sod_balance_env_var)
                if raw:
                    try:
                        sod_balance_usd = float(raw)
                    except ValueError:
                        logger.warning(
                            "%s=%r not parseable as float; ignoring daily-loss gate",
                            account.sod_balance_env_var, raw,
                        )

            failures = run_preflight(
                account_info=account_info,
                expected_login=login,
                expected_server=server_raw,
                actions_by_symbol=actions_by_symbol,
                symbol_infos=symbol_infos,
                ticks=ticks,
                requested_margin_usd=requested_margin_usd,
                max_margin_usage_pct=account.max_margin_usage_pct,
                sod_balance_usd=sod_balance_usd,
                max_daily_loss_pct=account.max_daily_loss_pct,
            )
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
        logger.exception("Account %s plan build failed", account.label)
        return AccountPlan(
            label=account.label,
            login=login,
            server=server_raw,
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


def _render_action_line(action: RebalanceAction) -> str:
    if action.kind is RebalanceActionKind.OPEN_NEW:
        side = (action.side.value if action.side else "?").upper()
        return f"  - OPEN {side} {action.volume} lots {action.symbol}"
    if action.kind is RebalanceActionKind.CLOSE_TICKET:
        return (
            f"  - CLOSE ticket #{action.ticket} ({action.volume} lots {action.symbol})"
        )
    if action.kind is RebalanceActionKind.ABORT_UNMANAGED:
        return f"  - ABORT {action.symbol} (unmanaged position): {action.reason}"
    return f"  - no-op {action.symbol} ({action.reason})"


def render_batch_approval_text(
    *,
    plans: Sequence[AccountPlan],
    forecasts: Dict[str, float],
    timeout_seconds: int,
) -> str:
    lines: List[str] = []
    lines.append("*ENIGMA CFD PROP — EXECUTION APPROVAL*")
    lines.append(f"_{datetime.now().strftime('%Y-%m-%d %H:%M PST')}_")
    lines.append("")
    lines.append("*Forecast (position % of sizing basis)*")
    lines.append("```")
    for ticker, frac in sorted(forecasts.items()):
        lines.append(f"  {ticker:<6} {frac * 100:+7.1f}%")
    lines.append("```")
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
                lines.append(_render_action_line(a))
        if not symbols:
            lines.append("   (no symbols planned)")
        lines.append("")
    lines.append(
        f"_Approve all / Cancel all / Cancel <label> — auto-decision in {timeout_seconds // 60} min._"
    )
    return "\n".join(lines)


def render_post_execution_summary(reports: Sequence[AccountExecutionReport]) -> str:
    lines: List[str] = []
    lines.append("*ENIGMA CFD PROP — EXECUTION SUMMARY*")
    lines.append(f"_{datetime.now().strftime('%Y-%m-%d %H:%M PST')}_")
    lines.append("")
    for r in reports:
        flag = {
            ApprovalDecision.APPROVED.value: "✅",
            ApprovalDecision.APPROVED_BY_TIMEOUT.value: "✅(timeout)",
            ApprovalDecision.CANCELLED.value: "❌",
            ApprovalDecision.TIMED_OUT.value: "⏱",
            ApprovalDecision.POLL_UNHEALTHY.value: "🚫",
        }.get(r.decision, "?")
        lines.append(f"{flag} *{r.label}* (login {r.login}): {r.decision}")
        if r.error:
            lines.append(f"   ⚠ {r.error}")
        for res in r.results:
            ok = "✓" if res.success else "✗"
            a = res.action
            sym = a.symbol
            if a.kind is RebalanceActionKind.OPEN_NEW and a.side and a.volume:
                desc = f"OPEN {a.side.value.upper()} {a.volume} {sym}"
            elif a.kind is RebalanceActionKind.CLOSE_TICKET and a.ticket:
                desc = f"CLOSE #{a.ticket} {a.volume} {sym}"
            else:
                desc = f"{a.kind.value} {sym}"
            extra = f" retcode={res.retcode}" if not res.success else ""
            deal_id = f" deal={res.deal_ticket}" if res.deal_ticket else ""
            lines.append(f"   {ok} {desc}{deal_id}{extra}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution (post-approval)
# ---------------------------------------------------------------------------


def _execute_account(
    *,
    account: AccountConfig,
    plan: AccountPlan,
    dry_run: bool,
) -> AccountExecutionReport:
    if not plan.has_real_actions():
        return AccountExecutionReport(
            label=plan.label,
            login=plan.login,
            decision=ApprovalDecision.APPROVED.value,
            plan=plan,
            results=[],
            error=None,
        )
    if dry_run:
        # Surface what would have happened without touching the broker.
        synthetic = [
            OrderResult(action=a, success=True, deal_ticket=None, retcode=0, comment="dry-run")
            for actions in plan.actions_by_symbol.values()
            for a in actions
            if a.kind in (RebalanceActionKind.OPEN_NEW, RebalanceActionKind.CLOSE_TICKET)
        ]
        return AccountExecutionReport(
            label=plan.label,
            login=plan.login,
            decision=ApprovalDecision.APPROVED.value,
            plan=plan,
            results=synthetic,
            error=None,
        )

    # Real execution: reconnect (we shut down after plan-build).
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
                    if action.kind is RebalanceActionKind.OPEN_NEW and action.side and action.volume:
                        results.append(
                            mt5x.place_market_order(
                                symbol_name=symbol_name,
                                side=action.side,
                                volume=float(action.volume),
                                magic=account.magic_number,
                            )
                        )
                    elif action.kind is RebalanceActionKind.CLOSE_TICKET and action.ticket and action.volume and action.side:
                        results.append(
                            mt5x.close_ticket(
                                ticket=int(action.ticket),
                                symbol_name=symbol_name,
                                original_side=action.side,
                                volume=float(action.volume),
                                magic=account.magic_number,
                            )
                        )
                    # NO_OP / ABORT_UNMANAGED: skip silently.
        return AccountExecutionReport(
            label=plan.label,
            login=plan.login,
            decision=ApprovalDecision.APPROVED.value,
            plan=plan,
            results=results,
            error=None,
        )
    except Exception as e:
        logger.exception("Execution failed for %s", account.label)
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


def _forecasts_from_df(forecasts_df: pd.DataFrame) -> Dict[str, float]:
    """Convert the orchestrator's input frame into a ticker→position_fraction dict.

    Accepts either ``position_fraction`` (preferred) or ``position_pct``
    (legacy). Tickers may be Enum members or plain strings.
    """
    out: Dict[str, float] = {}
    for _, row in forecasts_df.iterrows():
        ticker = row["ticker"]
        ticker_str = ticker.name if hasattr(ticker, "name") else str(ticker)
        if "position_fraction" in forecasts_df.columns:
            value = float(row["position_fraction"])
        elif "position_pct" in forecasts_df.columns:
            value = float(row["position_pct"]) / 100.0
        else:
            continue
        out[ticker_str] = value
    return out


def _write_audit_log(
    *,
    audit_dir: Path,
    run_id: str,
    plans: Sequence[AccountPlan],
    reports: Sequence[AccountExecutionReport],
    decision_outcome: BatchApprovalOutcome,
) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"cfd_prop_{run_id}_{int(time.time())}.json"

    def _serialize_action(a: RebalanceAction) -> Dict[str, Any]:
        return {
            "kind": a.kind.value,
            "symbol": a.symbol,
            "side": a.side.value if a.side else None,
            "volume": a.volume,
            "ticket": a.ticket,
            "reason": a.reason,
        }

    def _serialize_plan(p: AccountPlan) -> Dict[str, Any]:
        return {
            "label": p.label,
            "login": p.login,
            "server": p.server,
            "currency": p.currency,
            "balance": p.balance,
            "equity": p.equity,
            "margin_free": p.margin_free,
            "preflight_ok": p.preflight_ok,
            "preflight_messages": list(p.preflight_messages),
            "actions_by_symbol": {
                sym: [_serialize_action(a) for a in actions]
                for sym, actions in p.actions_by_symbol.items()
            },
        }

    def _serialize_report(r: AccountExecutionReport) -> Dict[str, Any]:
        return {
            "label": r.label,
            "login": r.login,
            "decision": r.decision,
            "error": r.error,
            "results": [
                {
                    "action": _serialize_action(res.action),
                    "success": res.success,
                    "deal_ticket": res.deal_ticket,
                    "retcode": res.retcode,
                    "comment": res.comment,
                }
                for res in r.results
            ],
        }

    payload = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "global_decision": decision_outcome.global_decision.value,
        "per_account_decision": {
            label: dec.value for label, dec in decision_outcome.per_account.items()
        },
        "approver_telegram_id": decision_outcome.approver_telegram_id,
        "plans": [_serialize_plan(p) for p in plans],
        "reports": [_serialize_report(r) for r in reports],
    }
    with open(audit_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return audit_path


def run_cfd_prop_execution(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    forecasts_df: pd.DataFrame,
) -> None:
    """Top-level entry called from ``scripts/enigma_live_forecast.py``."""

    # Match run_auto_execution semantics: only attempt MT5 work when
    # --execute is passed. --dry-run alone produces the forecast but
    # never touches MT5 / Telegram.
    if not getattr(args, "execute", False):
        print(
            "\n12. CFD prop execution: --execute not set; skipping MT5 connection. "
            "Pass --execute --dry-run-execute to build plans without sending orders, "
            "or --execute --approve-via-telegram for the full flow."
        )
        return

    forecasts = _forecasts_from_df(forecasts_df)
    if not forecasts:
        print("\n12. CFD prop execution: forecast frame is empty; nothing to do.")
        return

    accounts_raw = config.get("accounts") or []
    global_mt5 = config.get("mt5") or {}
    execution_cfg = config.get("execution") or {}
    top_level_tradeable: Sequence[str] = tuple(config.get("tradeable_tickers") or [])
    top_level_instruments: Dict[str, Dict[str, Any]] = dict(config.get("instruments") or {})

    if not top_level_tradeable:
        print("\n12. CFD prop execution: 'tradeable_tickers' is empty at top level; nothing to do.")
        return

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
        print("\n12. CFD prop execution: no enabled accounts in config; skipping.")
        return

    # Phase 1 stub flags re-used. Phase 3 honours them properly.
    dry_run_execute = bool(getattr(args, "dry_run_execute", False))
    require_approval = bool(getattr(args, "approve_via_telegram", True))
    is_dry_run = bool(getattr(args, "dry_run", False))

    if require_approval and not is_dry_run:
        notifier = TelegramNotifier.for_cfd_prop()
    else:
        notifier = None

    # --- Phase: build plans (one MT5 connect per account, sequential) ---
    print(f"\n12. CFD prop execution: building plans for {len(enabled)} account(s)...")
    plans: List[AccountPlan] = []
    for account in enabled:
        print(f"   - {account.label}: connecting MT5...")
        plan = _build_account_plan(account=account, forecasts=forecasts)
        if plan.preflight_ok:
            print(
                f"     ✓ equity ${plan.equity:,.2f} {plan.currency}, "
                f"{sum(len(a) for a in plan.actions_by_symbol.values())} action(s) planned"
            )
        else:
            for msg in plan.preflight_messages:
                print(f"     ✗ {msg}")
        plans.append(plan)

    timeout_seconds = int(execution_cfg.get("approval_timeout_seconds", 300))
    run_id = new_run_id()
    approval_text = render_batch_approval_text(
        plans=plans,
        forecasts=forecasts,
        timeout_seconds=timeout_seconds,
    )

    print("\n13. Batch approval message:")
    print("-" * 60)
    print(approval_text)
    print("-" * 60)

    # --- Phase: approval ---
    default_on_timeout_str = str(execution_cfg.get("default_on_timeout", "cancel")).lower()
    default_on_timeout = (
        ApprovalDecision.APPROVED if default_on_timeout_str == "approve"
        else ApprovalDecision.CANCELLED
    )

    if dry_run_execute or is_dry_run:
        print("\n14. dry-run-execute / dry-run set; SKIPPING approval and execution.")
        return

    eligible_labels = [p.label for p in plans if p.preflight_ok and p.has_real_actions()]
    if not eligible_labels:
        print("\n14. No accounts have actionable plans; nothing to approve. Done.")
        return

    if require_approval:
        if notifier is None or not notifier.is_configured():
            print(
                "\n14. ⚠ Telegram notifier not configured; refusing to execute without approval. "
                "Set TELEGRAM_CFD_PROP_BOT_TOKEN / TELEGRAM_CFD_PROP_CHAT_ID, or pass --dry-run-execute."
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
        # No-approval mode: --execute without --approve-via-telegram → auto-approve
        # all preflight-ok accounts. The user explicitly opted out of Telegram
        # by not passing --approve-via-telegram; trust their intent.
        outcome = BatchApprovalOutcome(
            global_decision=ApprovalDecision.APPROVED,
            per_account={label: ApprovalDecision.APPROVED for label in eligible_labels},
        )

    print(f"\n15. Global decision: {outcome.global_decision.value}")
    for label, dec in outcome.per_account.items():
        print(f"     {label}: {dec.value}")

    # --- Phase: execution ---
    reports: List[AccountExecutionReport] = []
    plan_by_label = {p.label: p for p in plans}
    account_by_label = {a.label: a for a in enabled}

    # Plans absent from the approval (preflight-failed or no real actions)
    # still need a report row so the user sees they were considered.
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
                        else "no actionable orders"
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
        print(f"\n16. Executing {label} ({len(plan.actions_by_symbol)} symbol(s))...")
        report = _execute_account(
            account=account,
            plan=plan,
            dry_run=False,
        )
        # Carry the actual approval decision into the report (executor uses
        # APPROVED as a placeholder; the orchestrator owns the real decision).
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
                f"     {mark} {r.action.kind.value} {r.action.symbol} "
                f"vol={r.action.volume} retcode={r.retcode}"
            )

    summary_text = render_post_execution_summary(reports)
    print("\n17. Summary:")
    print("-" * 60)
    print(summary_text)
    print("-" * 60)
    if notifier and notifier.is_configured():
        notifier.send_message(summary_text)

    # --- Phase: audit log ---
    audit_dir = Path(execution_cfg.get("audit_dir") or "logs/cfd_prop_audit")
    audit_path = _write_audit_log(
        audit_dir=audit_dir,
        run_id=run_id,
        plans=plans,
        reports=reports,
        decision_outcome=outcome,
    )
    print(f"\n18. Audit log written to {audit_path}")
