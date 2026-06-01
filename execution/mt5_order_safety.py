"""
Per-account preflight safety gates for MT5 execution.

Each gate is a pure function returning either ``None`` (gate passed) or a
short reason string (gate failed). The orchestrator collects failures
and refuses to execute an account whose preflight is not clean unless
``allow_unsafe_preflight`` is set in config (which we never do for live).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from execution.mt5_models import (
    MT5AccountInfo,
    MT5Position,
    MT5SymbolInfo,
    MT5Tick,
    OrderSide,
    RebalanceAction,
    RebalanceActionKind,
    TradeMode,
)


def check_account_login_matches(
    account_info: MT5AccountInfo, *, expected_login: int, expected_server: str
) -> Optional[str]:
    """Refuse to trade if we connected to the wrong account by accident.

    A copy-paste error in env vars could otherwise route a $100k FTMO plan
    to a $5k personal demo. Fail closed.
    """
    if account_info.login != expected_login:
        return (
            f"login mismatch: connected as {account_info.login}, "
            f"expected {expected_login}"
        )
    if account_info.server != expected_server:
        return (
            f"server mismatch: connected to {account_info.server!r}, "
            f"expected {expected_server!r}"
        )
    return None


def check_trade_allowed(account_info: MT5AccountInfo) -> Optional[str]:
    """``trade_allowed=False`` means MT5/server is in read-only mode."""
    if not account_info.trade_allowed:
        return "broker reports trade_allowed=False"
    if not account_info.trade_expert:
        return "broker reports trade_expert=False (algo trading disabled)"
    return None


def check_symbol_tradeable(symbol_info: MT5SymbolInfo) -> Optional[str]:
    """Symbol must be visible AND in a trade mode that supports market orders."""
    if not symbol_info.visible:
        return f"{symbol_info.name}: symbol not visible (run symbol_select)"
    if symbol_info.trade_mode is TradeMode.DISABLED:
        return f"{symbol_info.name}: trade_mode=DISABLED"
    return None


def check_action_compatible_with_trade_mode(
    action: RebalanceAction, symbol_info: MT5SymbolInfo
) -> Optional[str]:
    """LONG_ONLY symbols can only OPEN_NEW buys; SHORT_ONLY only sells."""
    if action.kind is not RebalanceActionKind.OPEN_NEW or action.side is None:
        return None  # closes are allowed in any mode
    if symbol_info.trade_mode is TradeMode.LONG_ONLY and action.side is OrderSide.SELL:
        return f"{symbol_info.name}: trade_mode=LONG_ONLY; cannot open SELL"
    if symbol_info.trade_mode is TradeMode.SHORT_ONLY and action.side is OrderSide.BUY:
        return f"{symbol_info.name}: trade_mode=SHORT_ONLY; cannot open BUY"
    if symbol_info.trade_mode is TradeMode.CLOSE_ONLY:
        return f"{symbol_info.name}: trade_mode=CLOSE_ONLY; cannot open"
    if symbol_info.trade_mode is TradeMode.DISABLED:
        return f"{symbol_info.name}: trade_mode=DISABLED"
    return None


def check_spread(
    symbol_info: MT5SymbolInfo, *, max_spread_points: int
) -> Optional[str]:
    """Refuse if the broker spread blew out beyond an absolute points cap."""
    if max_spread_points <= 0:
        return None
    if symbol_info.spread > max_spread_points:
        return (
            f"{symbol_info.name}: spread={symbol_info.spread} pts > "
            f"max_spread_points={max_spread_points}"
        )
    return None


def check_tick_freshness(
    tick: MT5Tick,
    *,
    now_utc: datetime,
    max_staleness_seconds: int,
) -> Optional[str]:
    """Refuse if the latest tick is older than ``max_staleness_seconds``.

    Common cause: a closed market on a weekend / holiday, or a server
    feeding stale ticks. ``now_utc`` is injected so the gate is pure
    (testable without ``datetime.now`` mocking).
    """
    if max_staleness_seconds <= 0:
        return None
    age_seconds = (now_utc - tick.time_utc).total_seconds()
    if age_seconds > max_staleness_seconds:
        return (
            f"{tick.symbol}: last tick is {age_seconds:.0f}s old "
            f"(> {max_staleness_seconds}s); market likely closed"
        )
    return None


def check_margin_budget(
    *,
    requested_margin_usd: float,
    account_info: MT5AccountInfo,
    max_margin_usage_pct: float = 0.95,
) -> Optional[str]:
    """Refuse if the planned actions would push margin usage past the cap."""
    if requested_margin_usd <= 0:
        return None
    cap = account_info.margin_free * max_margin_usage_pct
    if requested_margin_usd > cap:
        return (
            f"requested margin ${requested_margin_usd:,.2f} > "
            f"{int(max_margin_usage_pct * 100)}% of margin_free "
            f"(${account_info.margin_free:,.2f}); skipping account"
        )
    return None


def check_daily_loss_floor(
    *,
    account_info: MT5AccountInfo,
    sod_balance_usd: Optional[float],
    max_daily_loss_pct: Optional[float],
) -> Optional[str]:
    """If daily drawdown exceeds ``max_daily_loss_pct`` of start-of-day
    balance, halt this account so we don't blow the prop firm's daily
    loss rule. Disabled when either knob is ``None``."""
    if sod_balance_usd is None or max_daily_loss_pct is None:
        return None
    if max_daily_loss_pct <= 0 or sod_balance_usd <= 0:
        return None
    drawdown_usd = sod_balance_usd - account_info.equity
    drawdown_pct = drawdown_usd / sod_balance_usd
    if drawdown_pct >= max_daily_loss_pct:
        return (
            f"daily drawdown {drawdown_pct * 100:.2f}% "
            f"(${drawdown_usd:,.2f}) >= cap {max_daily_loss_pct * 100:.2f}%; "
            f"halting account to protect prop-firm daily loss rule"
        )
    return None


def run_preflight(
    *,
    account_info: MT5AccountInfo,
    expected_login: int,
    expected_server: str,
    actions_by_symbol: Dict[str, List[RebalanceAction]],
    symbol_infos: Dict[str, MT5SymbolInfo],
    ticks: Dict[str, MT5Tick],
    requested_margin_usd: float,
    now_utc: datetime,
    max_spread_points: int = 0,
    max_tick_staleness_seconds: int = 0,
    max_margin_usage_pct: float = 0.95,
    sod_balance_usd: Optional[float] = None,
    max_daily_loss_pct: Optional[float] = None,
) -> List[str]:
    """Run every gate; return a list of failure reasons (empty = pass)."""
    failures: List[str] = []

    for check in (
        check_account_login_matches(
            account_info,
            expected_login=expected_login,
            expected_server=expected_server,
        ),
        check_trade_allowed(account_info),
        check_margin_budget(
            requested_margin_usd=requested_margin_usd,
            account_info=account_info,
            max_margin_usage_pct=max_margin_usage_pct,
        ),
        check_daily_loss_floor(
            account_info=account_info,
            sod_balance_usd=sod_balance_usd,
            max_daily_loss_pct=max_daily_loss_pct,
        ),
    ):
        if check is not None:
            failures.append(check)

    touched_symbols = sorted(actions_by_symbol.keys())
    for symbol_name in touched_symbols:
        sym = symbol_infos.get(symbol_name)
        tick = ticks.get(symbol_name)
        if sym is None or tick is None:
            # Already surfaced by the rebalancer as NO_OP; skip silently here.
            continue
        actions = actions_by_symbol[symbol_name]
        # Only check symbol-level gates if there's at least one real action.
        if not any(
            a.kind in (RebalanceActionKind.OPEN_NEW, RebalanceActionKind.CLOSE_TICKET)
            for a in actions
        ):
            continue
        if (msg := check_symbol_tradeable(sym)) is not None:
            failures.append(msg)
        if (msg := check_spread(sym, max_spread_points=max_spread_points)) is not None:
            failures.append(msg)
        if (
            msg := check_tick_freshness(
                tick,
                now_utc=now_utc,
                max_staleness_seconds=max_tick_staleness_seconds,
            )
        ) is not None:
            failures.append(msg)
        for action in actions:
            msg = check_action_compatible_with_trade_mode(action, sym)
            if msg is not None:
                failures.append(msg)

    return failures
