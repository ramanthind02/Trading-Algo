"""Live-monitoring API: read the node-published snapshot + issue control commands.

The live ``TradingNode`` runs as a SEPARATE process that owns its broker's MT5
terminal and *publishes* a JSON snapshot (see
:mod:`deployment.live.monitoring.live_state`). This module never opens an MT5
connection — it only READS the snapshot/equity files and WRITES a command file
the node consumes. That makes monitoring + the ``flatten`` kill switch safe
against the running node: no terminal contention (one MT5 owner — the node) and
no double-trade race (the node flattens itself and HALTS; an out-of-process
flatten would have raced the node's next rebalance).

Prop-firm utilisation gauges are DERIVED here (pure arithmetic) from the
snapshot's equity + persisted risk baseline + the broker's static
``risk_rules`` — no code computes these live anywhere else.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from data_platform.providers.mt5 import brokers
from deployment.live.monitoring import live_state

#: A node publishes every ~5s; treat a snapshot older than this as "node offline".
STALE_AFTER_SECS = 30.0


# ── time helpers ───────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def _age_seconds(ts_iso: Any, now: datetime) -> float | None:
    ts = _parse_iso(ts_iso)
    return None if ts is None else (now - ts).total_seconds()


def _is_online(age: float | None) -> bool:
    return age is not None and age <= STALE_AFTER_SECS


def _require_known(broker: str) -> None:
    if broker not in brokers.known_brokers():
        raise ValueError(f"Unknown broker: {broker!r}")


def _snapshot_path(broker: str):
    return live_state.snapshot_path(live_state.live_state_dir(broker))


# ── prop-firm gauges (pure) ────────────────────────────────────────────────────

def compute_risk_gauges(
    *,
    equity: float | None,
    account_start: float | None,
    day_start_equity: float | None,
    day_start_balance: float | None = None,
    initial_balance: float | None = None,
    gross_notional: float | None,
    rules: Any,
) -> dict[str, Any]:
    """Derive prop-firm utilisation from live equity + baselines + static limits.

    Faithful to FTMO-style rules: limits are a fixed dollar amount = ``pct`` of the
    INITIAL balance (not of the day-start equity). The daily anchor is
    ``max(day_start_balance, day_start_equity)`` — FTMO recalculates the daily limit
    off the day-start balance, and using the max is firm-conservative for a strategy
    that holds positions overnight. ``rules`` is a ``brokers.RiskRules`` or ``None``
    (live-retail). Drawdowns are positive percentages of the base; headroom is the
    remaining distance to the limit (negative == breached).
    """
    g: dict[str, Any] = {
        "daily_pnl_pct": None,
        "daily_drawdown_pct": None,
        "daily_limit_pct": None,
        "daily_headroom_pct": None,
        "daily_used_pct": None,
        "total_pnl_pct": None,
        "total_drawdown_pct": None,
        "total_limit_pct": None,
        "total_headroom_pct": None,
        "total_used_pct": None,
        "profit_target_pct": None,
        "profit_progress_pct": None,
        "leverage_in_use": None,
        "max_leverage": None,
    }
    # Dollar basis for the % limits: the configured initial balance, else the
    # persisted account-start anchor.
    base = initial_balance if (initial_balance and initial_balance > 0) else account_start
    anchors = [v for v in (day_start_equity, day_start_balance) if v]
    day_anchor = max(anchors) if anchors else None

    if equity is not None and day_anchor and base:
        g["daily_pnl_pct"] = (equity - day_anchor) / base * 100.0
        g["daily_drawdown_pct"] = max(0.0, (day_anchor - equity) / base * 100.0)
    if equity is not None and base:
        g["total_pnl_pct"] = (equity - base) / base * 100.0
        g["total_drawdown_pct"] = max(0.0, (base - equity) / base * 100.0)
    if gross_notional is not None and equity:
        g["leverage_in_use"] = gross_notional / equity

    if rules is None:
        return g

    if rules.max_daily_loss_pct:
        g["daily_limit_pct"] = rules.max_daily_loss_pct
        if g["daily_drawdown_pct"] is not None:
            g["daily_headroom_pct"] = rules.max_daily_loss_pct - g["daily_drawdown_pct"]
            g["daily_used_pct"] = g["daily_drawdown_pct"] / rules.max_daily_loss_pct * 100.0
    if rules.max_total_loss_pct:
        g["total_limit_pct"] = rules.max_total_loss_pct
        if g["total_drawdown_pct"] is not None:
            g["total_headroom_pct"] = rules.max_total_loss_pct - g["total_drawdown_pct"]
            g["total_used_pct"] = g["total_drawdown_pct"] / rules.max_total_loss_pct * 100.0
    if rules.profit_target_pct:
        g["profit_target_pct"] = rules.profit_target_pct
        if g["total_pnl_pct"] is not None:
            g["profit_progress_pct"] = g["total_pnl_pct"] / rules.profit_target_pct * 100.0
    g["max_leverage"] = rules.max_leverage
    return g


def _rules_to_dict(rules: Any) -> dict[str, Any] | None:
    if rules is None:
        return None
    return {
        "max_daily_loss_pct": rules.max_daily_loss_pct,
        "max_total_loss_pct": rules.max_total_loss_pct,
        "profit_target_pct": rules.profit_target_pct,
        "min_trading_days": rules.min_trading_days,
        "max_leverage": rules.max_leverage,
        "news_trading_restricted": rules.news_trading_restricted,
    }


# ── read endpoints ──────────────────────────────────────────────────────────────

def list_brokers() -> dict[str, Any]:
    """Every known broker + whether a live node is currently publishing for it."""
    now = _now()
    out: list[dict[str, Any]] = []
    for b in brokers.known_brokers():
        snap = live_state.read_snapshot(_snapshot_path(b))
        age = _age_seconds(snap.get("ts"), now) if snap else None
        out.append({
            "broker": b,
            "confirmed": brokers.is_confirmed(b),
            "has_snapshot": snap is not None,
            "online": _is_online(age),
            "age_seconds": age,
            "exec_tier": snap.get("exec_tier") if snap else None,
            "halted": snap.get("halted") if snap else None,
            "has_risk_limits": brokers.risk_rules(b) is not None,
        })
    return {"brokers": out, "stale_after_secs": STALE_AFTER_SECS}


def get_snapshot(broker: str) -> dict[str, Any]:
    """The full live snapshot for ``broker`` (account, positions, targets, warmup)."""
    _require_known(broker)
    snap = live_state.read_snapshot(_snapshot_path(broker))
    if snap is None:
        raise FileNotFoundError(
            f"No live snapshot for broker {broker!r} (the live node is not running)."
        )
    age = _age_seconds(snap.get("ts"), _now())
    return {**snap, "age_seconds": age, "online": _is_online(age)}


def get_risk(broker: str) -> dict[str, Any]:
    """Prop-firm limits + DERIVED utilisation gauges for ``broker``."""
    _require_known(broker)
    snap = live_state.read_snapshot(_snapshot_path(broker))
    rules = brokers.risk_rules(broker)
    account = (snap or {}).get("account") or {}
    baseline = (snap or {}).get("risk_baseline") or {
        "account_start_equity": None,
        "account_start_at": None,
        "day_start_equity": None,
        "day_start_balance": None,
        "day_start_date": None,
        "day_start_estimated": False,
    }
    equity = account.get("equity")
    age = _age_seconds(snap.get("ts"), _now()) if snap else None
    gauges = compute_risk_gauges(
        equity=equity,
        account_start=baseline.get("account_start_equity"),
        day_start_equity=baseline.get("day_start_equity"),
        day_start_balance=baseline.get("day_start_balance"),
        initial_balance=(snap or {}).get("initial_balance"),
        gross_notional=(snap or {}).get("gross_notional"),
        rules=rules,
    )
    return {
        "broker": broker,
        "has_limits": rules is not None,
        "limits": _rules_to_dict(rules),
        "gauges": gauges,
        "equity": equity,
        "baseline": baseline,
        "day_start_estimated": bool(baseline.get("day_start_estimated")),
        "online": _is_online(age),
    }


def get_equity(broker: str, *, limit: int = 2000) -> dict[str, Any]:
    """The persisted live equity series (oldest first) for the curve / drawdown."""
    _require_known(broker)
    series = live_state.read_equity_series(
        live_state.equity_path(live_state.live_state_dir(broker)), limit=limit
    )
    return {"broker": broker, "series": series}


# ── control endpoint (dashboard → node) ─────────────────────────────────────────

def flatten(broker: str, *, confirm: str) -> dict[str, Any]:
    """Queue a node-mediated FLATTEN-ALL kill switch for ``broker``.

    Writes a command file the running node consumes on its next snapshot tick:
    the node closes every position through its OWN execution client and HALTS
    (so it never re-opens). Demo-only — refused on a live/funded account. Requires
    ``confirm == broker`` (the UI makes the operator type the broker name).
    """
    _require_known(broker)
    if confirm != broker:
        raise ValueError(
            f"Confirmation mismatch: type the broker name {broker!r} to confirm flatten."
        )
    state_dir = live_state.live_state_dir(broker)
    snap = live_state.read_snapshot(live_state.snapshot_path(state_dir))
    if snap is None:
        raise FileNotFoundError(
            f"No live node for {broker!r}; cannot flatten from the dashboard (node offline). "
            "Use `python -m deployment.live.manual_trade flatten` on a free terminal instead."
        )
    age = _age_seconds(snap.get("ts"), _now())
    if not _is_online(age):
        age_txt = f"{age:.0f}s old" if age is not None else "no valid timestamp"
        raise RuntimeError(
            f"Live node for {broker!r} appears offline (snapshot {age_txt}); "
            "refusing to queue a flatten it may never consume."
        )
    if str(snap.get("exec_tier", "")).lower() == "live":
        raise PermissionError("Demo-only: flatten is disabled on a live/funded account.")

    cmd = live_state.Command(
        id=uuid.uuid4().hex, action="flatten", issued_at=_now().isoformat(), confirm=confirm
    )
    live_state.write_command(live_state.command_path(state_dir), cmd)
    return {
        "status": "submitted",
        "command_id": cmd.id,
        "broker": broker,
        "issued_at": cmd.issued_at,
        "note": "Flatten queued. The node will close all positions and halt on its next tick; "
        "watch last_command_result in the snapshot for the ack.",
    }


__all__ = [
    "STALE_AFTER_SECS",
    "compute_risk_gauges",
    "list_brokers",
    "get_snapshot",
    "get_risk",
    "get_equity",
    "flatten",
]
