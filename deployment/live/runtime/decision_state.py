"""Durable daily-decision state so a crash + restart never re-fires a done session.

The live strategy decides once per US session (16:05 ET). That "already decided
today" marker AND the set of symbols already executed must survive a process crash
— otherwise a restart at, say, 16:30 ET re-runs the day's decision (and, before
broker reconciliation populates positions, could double a position). We persist a
tiny JSON per EXECUTION broker and reload it in ``on_start``.

Writes are atomic (:mod:`lib.core.atomic_io`); a corrupt/missing file reads back as
empty state, so the day simply re-decides — which is safe because the netting-delta
planner + broker reconciliation make a re-decide a no-op once positions are known.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from lib.core.atomic_io import atomic_write_text, read_or_quarantine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionState:
    """Last decided session + the canonicals already executed that session."""

    session_date: date | None = None
    executed: frozenset[str] = field(default_factory=frozenset)

    def executed_for(self, session: date | None) -> frozenset[str]:
        """Executed tickers IF the stored state is for ``session``, else empty."""
        if session is None or self.session_date != session:
            return frozenset()
        return self.executed


def state_path_for_broker(broker: str) -> Path:
    """``data/broker_cache/<broker>/decision_state.json`` (per execution broker)."""
    from deployment.live.broker_data import broker_cache_root

    return broker_cache_root(broker).parent / "decision_state.json"


def load(path: str | Path) -> DecisionState:
    """Read decision state; empty (and self-healing) on missing/corrupt file."""

    def _read(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    raw = read_or_quarantine(path, _read)
    if not raw:
        return DecisionState()
    sd = raw.get("session_date")
    session = date.fromisoformat(sd) if isinstance(sd, str) and sd else None
    executed = frozenset(str(t) for t in raw.get("executed", ()))
    return DecisionState(session_date=session, executed=executed)


def save(path: str | Path, state: DecisionState) -> None:
    """Atomically persist decision state."""
    payload = {
        "session_date": state.session_date.isoformat() if state.session_date else None,
        "executed": sorted(state.executed),
    }
    atomic_write_text(path, json.dumps(payload, indent=2))


__all__ = ["DecisionState", "load", "save", "state_path_for_broker"]
