"""Canonical bar record — the silver-layer schema all adapters emit.

A source-tagged OHLCV row keyed by canonical ``InstrumentId`` value. The
reconciler consumes these from each adapter and produces the canonical winner
with provenance. Superset of the current parquet schema (adds source, adjusted,
adjustment_type, provenance_ratio).

See docs/library/Data/multi_source_update_architecture.md §7.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CanonicalBarRecord:
    instrument_id: str        # InstrumentId.value, e.g. "ES.XCME"
    resolution: str           # "D" | "W" | "M" | "M1"
    ts_event: datetime        # bar close timestamp (tz-naive UTC)
    ts_ingest: datetime       # wall-clock ingest time
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str               # "norgate" | "ib" | "mt5"
    adjusted: bool            # True = back-adjusted (CCB or ratio)
    adjustment_type: str      # "BACK_ADJUSTED" | "NONE" | "RATIO" | "TOTAL_RETURN" | "CAPITAL"
    provenance_ratio: float | None = None  # junction ratio applied at splice, if any
