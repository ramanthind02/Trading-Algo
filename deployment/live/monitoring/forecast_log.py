"""Durable per-decision forecast capture (migration plan §7.4).

The live node's daily targets previously existed only in the ~5s-overwritten
``snapshot.json`` — there was no historical record of what was forecast on any
past day. This appends one JSONL row per (decision, canonical) at the ENTRY
decision point; ``registry ingest live`` loads them into ``forecast_history``.

Append-only JSONL, same idiom as ``slippage.record_submit``: the caller wraps
every call so a capture failure never breaks trading. Retained permanently
(ADR-3 — archived on account rotation, never pruned).
"""
from __future__ import annotations

import json
from pathlib import Path

from deployment.live.monitoring.live_state import live_state_dir

SCHEMA_VERSION = 1


def forecasts_jsonl_path(broker: str) -> Path:
    return Path(live_state_dir(broker)) / "forecasts.jsonl"


def record_forecasts(broker: str, rows: list[dict]) -> None:
    """Append one row per (decision, canonical). Caller wraps exceptions."""
    if not rows:
        return
    path = forecasts_jsonl_path(broker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.writelines(
            json.dumps({"schema_version": SCHEMA_VERSION, "broker": broker, **row}) + "\n"
            for row in rows
        )
