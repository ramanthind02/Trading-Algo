"""
Append-only JSONL audit log for execution runs.

One file per run (``logs/execution/{date}_{run_id}.jsonl``) plus a flat
``logs/execution/positions_latest.json`` that records closing positions so
the next run can do reconciliation.

Every record is a flat dict with a ``kind`` field. No updates, no deletes;
appending only. If the script crashes, the file on disk reflects exactly
what was known up to that point.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from execution.models import OrderIntent, OrderResult


class AuditLog:
    """Per-run append-only JSONL writer."""

    def __init__(self, run_id: str, log_dir: Path) -> None:
        self.run_id = run_id
        date_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{date_stamp}_{run_id}.jsonl"

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "kind": kind,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def log_start(self, *, profile: str, port: int, account: str, live: bool) -> None:
        self._emit("run_start", {"profile": profile, "port": port, "account": account, "live": live})

    def log_intents(self, intents: List[OrderIntent]) -> None:
        self._emit("intents", {"orders": [_to_dict(i) for i in intents]})

    def log_preflight(self, passed: bool, reason: Optional[str] = None) -> None:
        self._emit("preflight", {"passed": passed, "reason": reason})

    def log_approval(self, *, approver_id: Optional[int], decision: str) -> None:
        self._emit("approval", {"approver_id": approver_id, "decision": decision})

    def log_order_placed(self, *, intent: OrderIntent, ib_order_id: int) -> None:
        self._emit("order_placed", {"ib_order_id": ib_order_id, "intent": _to_dict(intent)})

    def log_order_status(
        self,
        *,
        ib_order_id: int,
        status: str,
        filled: Optional[Decimal] = None,
        avg_price: Optional[float] = None,
    ) -> None:
        self._emit("order_status", {
            "ib_order_id": ib_order_id,
            "status": status,
            "filled": filled,
            "avg_price": avg_price,
        })

    def log_order_result(self, result: OrderResult) -> None:
        self._emit("order_result", _to_dict(result))

    def log_error(self, component: str, message: str) -> None:
        self._emit("error", {"component": component, "message": message})

    def log_end(self, *, status: str) -> None:
        self._emit("run_end", {"status": status})


def write_closing_positions(
    positions: Dict[str, Decimal],
    out_path: Path,
) -> None:
    """Persist the set of positions we believe we hold after execution.

    Next run's reconciliation gate reads this file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "positions": {etf: str(shares) for etf, shares in positions.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2))


def read_closing_positions(path: Path) -> Optional[Dict[str, Decimal]]:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {etf: Decimal(s) for etf, s in raw["positions"].items()}


def _to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    raise TypeError(f"Unserializable: {type(obj).__name__}")
