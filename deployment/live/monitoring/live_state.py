"""Live-state publication for the vault node (node-mediated dashboard feed).

The running ``TradingNode`` is the SOLE owner of its broker's MT5 terminal, so the
monitoring dashboard must never open its own MT5 connection (MT5's Python API binds
one terminal per OS process; a second connection would contend, and an
out-of-process "flatten" would race the node's next rebalance). Instead the live
strategy *publishes* a JSON snapshot (account, positions, target-vs-actual,
warmup, risk baseline) on a timer and *consumes* a command file (flatten). The
dashboard API only reads the snapshot and writes commands — one writer (the node),
many readers (the API). No terminal contention, no double-trade race.

This module is the FUNCTIONAL CORE: file-path resolution, the risk-baseline
math, command parsing, equity-history IO, halt persistence, and atomic snapshot
writes. It imports NOTHING from NautilusTrader or MetaTrader5. The strategy (the
imperative shell) extracts the live numbers from the Nautilus cache/portfolio and
calls these helpers; the API (the reader) imports the same paths + parsers so both
sides agree on the contract.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from lib.core.atomic_io import atomic_write_text, read_or_quarantine

SCHEMA_VERSION = 1

#: Bound the equity history: keep ~this many of the most recent samples on disk and
#: compact when the file grows past the byte budget (append is ~1 line/60s).
MAX_EQUITY_SAMPLES = 5000
_EQUITY_COMPACT_BYTES = 1_000_000


# ── file layout (one live_state dir per broker) ────────────────────────────────

def live_state_dir(broker: str) -> Path:
    """``data/broker_cache/<broker>/live_state`` — sibling of ``decision_state.json``."""
    from deployment.live.broker_data import broker_cache_root

    return broker_cache_root(broker).parent / "live_state"


def snapshot_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / "snapshot.json"


def command_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / "command.json"


def baseline_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / "baseline.json"


def equity_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / "equity_history.jsonl"


def equity_durable_path(state_dir: str | Path) -> Path:
    """Uncompacted durable equity log — every sample kept forever (migration plan §7.5)."""
    return Path(state_dir) / "equity.jsonl"


def halt_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / "halt.json"


# ── risk baseline (day-start + account-start anchors for prop-firm gauges) ──────

@dataclass(frozen=True)
class RiskBaseline:
    """Persisted equity/balance anchors the prop-firm gauges are measured against.

    ``account_start_equity`` anchors the total/max-loss gauge (static, FTMO-style
    "from initial balance"). ``day_start_equity`` / ``day_start_balance`` anchor the
    daily-loss gauge and reset when the broker calendar date rolls over.
    ``day_start_estimated`` is True when the day anchor was set on a COLD START into a
    new broker day (node was down across the boundary) rather than a witnessed
    rollover — the true midnight equity is then unrecoverable, so the daily gauge
    should be shown as approximate.
    """

    account_start_equity: float | None = None
    account_start_at: str | None = None
    day_start_equity: float | None = None
    day_start_balance: float | None = None
    day_start_date: str | None = None
    day_start_estimated: bool = False


def update_baseline(
    prev: RiskBaseline,
    *,
    equity: float,
    broker_date: str,
    now_iso: str,
    balance: float | None = None,
    initial_balance: float = 0.0,
    prev_observed_date: str | None = None,
) -> RiskBaseline:
    """Pure baseline transition for one observed (equity, balance) at ``broker_date``.

    The account anchor is set once (to ``initial_balance`` if given, else the first
    observed equity). The day anchor resets to the current equity/balance whenever
    the broker date changes. ``prev_observed_date`` is the broker date seen on the
    PREVIOUS tick of the current run (``None`` on the first tick after start); it is
    used to flag a cold-start day reset (node was down across midnight) as estimated.
    """
    account_start_equity = prev.account_start_equity
    account_start_at = prev.account_start_at
    if account_start_equity is None:
        account_start_equity = float(initial_balance) if initial_balance and initial_balance > 0 else float(equity)
        account_start_at = now_iso

    day_start_equity = prev.day_start_equity
    day_start_balance = prev.day_start_balance
    day_start_date = prev.day_start_date
    day_start_estimated = prev.day_start_estimated

    if day_start_date != broker_date or day_start_equity is None:
        day_start_equity = float(equity)
        day_start_balance = float(balance) if balance is not None else float(equity)
        # Witnessed rollover = the node was running on the prior anchor day. A cold
        # start into a NEW day (prev_observed_date is None or != the old anchor) can't
        # reconstruct the true midnight equity → flag estimated. The very first
        # baseline (prev.day_start_date is None) is a genuine start, not estimated.
        witnessed = prev_observed_date is not None and prev_observed_date == prev.day_start_date
        day_start_estimated = (prev.day_start_date is not None) and not witnessed
        day_start_date = broker_date

    return RiskBaseline(
        account_start_equity=account_start_equity,
        account_start_at=account_start_at,
        day_start_equity=day_start_equity,
        day_start_balance=day_start_balance,
        day_start_date=day_start_date,
        day_start_estimated=day_start_estimated,
    )


def load_baseline(path: str | Path) -> RiskBaseline:
    """Read the persisted baseline; empty (and self-healing) on missing/corrupt."""

    def _read(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    raw = read_or_quarantine(path, _read)
    if not raw:
        return RiskBaseline()
    return RiskBaseline(
        account_start_equity=raw.get("account_start_equity"),
        account_start_at=raw.get("account_start_at"),
        day_start_equity=raw.get("day_start_equity"),
        day_start_balance=raw.get("day_start_balance"),
        day_start_date=raw.get("day_start_date"),
        day_start_estimated=bool(raw.get("day_start_estimated", False)),
    )


def save_baseline(path: str | Path, baseline: RiskBaseline) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(asdict(baseline), indent=2))


# ── halt persistence (durable kill-switch state across restarts) ────────────────

@dataclass(frozen=True)
class HaltState:
    """Durable record that the node was halted by the kill switch.

    Persisted so a crash/auto-restart does NOT silently resume trading: a halted
    node reloads this on start and stays halted until an operator clears it
    (delete ``halt.json`` and restart, or re-flatten to clean up).
    """

    halted: bool = False
    by_command_id: str | None = None
    at: str | None = None


def load_halt(path: str | Path) -> HaltState:
    def _read(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    raw = read_or_quarantine(path, _read)
    if not raw:
        return HaltState()
    return HaltState(
        halted=bool(raw.get("halted", False)),
        by_command_id=raw.get("by_command_id"),
        at=raw.get("at"),
    )


def save_halt(path: str | Path, state: HaltState) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(asdict(state), indent=2))


def clear_halt(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


# ── command channel (dashboard → node) ─────────────────────────────────────────

@dataclass(frozen=True)
class Command:
    """A control command the dashboard wrote for the node to execute."""

    id: str
    action: str
    issued_at: str  # ISO-8601 UTC
    confirm: str | None = None


def read_command(path: str | Path) -> Command | None:
    """Parse the command file; ``None`` if missing, corrupt, or malformed."""

    def _read(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    raw = read_or_quarantine(path, _read)
    if not raw or not isinstance(raw, dict):
        return None
    cid, action, issued = raw.get("id"), raw.get("action"), raw.get("issued_at")
    if not (isinstance(cid, str) and isinstance(action, str) and isinstance(issued, str)):
        return None
    confirm = raw.get("confirm")
    return Command(id=cid, action=action, issued_at=issued, confirm=confirm if isinstance(confirm, str) else None)


def write_command(path: str | Path, command: Command) -> None:
    """Atomically write a command (dashboard side)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(asdict(command), indent=2))


def clear_command(path: str | Path) -> None:
    """Delete a consumed command so it can never be re-read after a restart."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


# ── equity history (append-only samples, bounded on disk + on read) ────────────

def append_equity_sample(path: str | Path, *, ts_iso: str, equity: float) -> None:
    """Append one ``{ts, equity}`` JSON line; compact when the file grows too large."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": ts_iso, "equity": float(equity)})
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    try:
        if p.stat().st_size > _EQUITY_COMPACT_BYTES:
            _compact_equity(p)
    except OSError:  # pragma: no cover - best-effort
        pass


def _compact_equity(p: Path) -> None:
    with p.open("r", encoding="utf-8") as fh:
        tail = deque(fh, maxlen=MAX_EQUITY_SAMPLES)
    atomic_write_text(p, "".join(tail), fsync=False)


def append_equity_sample_durable(
    path: str | Path,
    *,
    ts_iso: str,
    equity: float,
    balance: float | None = None,
    floating_pnl: float | None = None,
    gross_notional: float | None = None,
    marks_fresh: bool | None = None,
) -> None:
    """Append one record to the permanent durable equity log (migration plan §7.5).

    Parallel to ``append_equity_sample`` (the capped dashboard series) but retains
    every sample forever — never compacted. Optional ``balance``, ``floating_pnl``,
    ``gross_notional``, and ``marks_fresh`` are included when provided; absent fields
    are omitted so the base ``{ts, equity}`` shape is always valid.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec: dict = {"ts": ts_iso, "equity": float(equity)}
    if balance is not None:
        rec["balance"] = float(balance)
    if floating_pnl is not None:
        rec["floating_pnl"] = float(floating_pnl)
    if gross_notional is not None:
        rec["gross_notional"] = float(gross_notional)
    if marks_fresh is not None:
        rec["marks_fresh"] = bool(marks_fresh)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def read_equity_series(path: str | Path, *, limit: int = 2000) -> list[dict]:
    """Return up to the last ``limit`` valid ``{ts, equity}`` samples (oldest first).

    Tail-reads (bounded memory) and skips any malformed/torn line — a JSON-invalid
    line OR a JSON-valid line with a non-numeric equity — so one bad record never
    poisons the whole series.
    """
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8") as fh:
        tail = deque(fh, maxlen=limit)
    out: list[dict] = []
    for raw in tail:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
            if isinstance(rec, dict) and "ts" in rec and "equity" in rec:
                out.append({"ts": rec["ts"], "equity": float(rec["equity"])})
        except (ValueError, TypeError, KeyError):
            continue  # torn line or non-numeric value — skip, don't fail the series
    return out


# ── snapshot IO ────────────────────────────────────────────────────────────────

def write_snapshot(path: str | Path, snapshot: dict) -> None:
    """Atomically write the snapshot dict (node side).

    ``fsync=False``: the snapshot is a disposable status file regenerated every few
    seconds, so we keep the atomic rename but skip the forced disk flush to avoid
    blocking the trading event-loop thread on every tick.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(snapshot, indent=2, default=str), fsync=False)


def read_snapshot(path: str | Path) -> dict | None:
    """Read + parse the snapshot dict; ``None`` if missing or corrupt (reader side)."""

    def _read(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    raw = read_or_quarantine(path, _read)
    return raw if isinstance(raw, dict) else None


__all__ = [
    "SCHEMA_VERSION",
    "MAX_EQUITY_SAMPLES",
    "RiskBaseline",
    "HaltState",
    "Command",
    "live_state_dir",
    "snapshot_path",
    "command_path",
    "baseline_path",
    "equity_path",
    "equity_durable_path",
    "halt_path",
    "update_baseline",
    "load_baseline",
    "save_baseline",
    "load_halt",
    "save_halt",
    "clear_halt",
    "read_command",
    "write_command",
    "clear_command",
    "append_equity_sample",
    "append_equity_sample_durable",
    "read_equity_series",
    "write_snapshot",
    "read_snapshot",
]
