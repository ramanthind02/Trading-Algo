"""Provenance + conflict persistence for reconciled bars (Phase 3).

Every canonical bar batch the reconciler writes records which source produced it,
when, and whether a junction ratio was applied. Stored separately from the OHLC
parquet (no schema change to existing consumers):

    data/provenance/{INSTRUMENT}/D_provenance.parquet

Conflicts (overlapping dates with close deviation above threshold) are appended
to a per-instrument conflict log and emitted via the logger.

See docs/library/Data/multi_source_update_architecture.md §8.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .reconciler import ConflictRecord, ProvenanceRecord

logger = logging.getLogger(__name__)

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )


def provenance_dir() -> Path:
    return _repo_root() / "data" / "provenance"


def _safe_instrument(instrument_id: str) -> str:
    return instrument_id.replace(".", "_").replace("/", "_")


def _append_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], axis=0, ignore_index=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


def write_provenance(records: tuple[ProvenanceRecord, ...], *, resolution: str = "D") -> None:
    """Append provenance rows to data/provenance/{INSTRUMENT}/{RES}_provenance.parquet."""
    if not records:
        return
    by_inst: dict[str, list[dict]] = {}
    for r in records:
        by_inst.setdefault(r.instrument_id, []).append(asdict(r))
    for instrument_id, rows in by_inst.items():
        path = provenance_dir() / _safe_instrument(instrument_id) / f"{resolution}_provenance.parquet"
        _append_parquet(pd.DataFrame(rows), path)


def log_conflicts(conflicts: tuple[ConflictRecord, ...], *, fail_fast: bool = False) -> None:
    """Emit conflicts via the logger; optionally raise (post-handover integrity).

    During the Norgate+IB overlap, conflicts on recent dates are expected and
    informational. Post-handover, an overlap with the frozen Norgate series is a
    re-ingestion error — pass ``fail_fast=True`` to raise instead of warn.
    """
    for c in conflicts:
        msg = (
            f"SOURCE CONFLICT {c.instrument_id} {c.date.date()}: "
            f"{c.source_a}={c.close_a:.4f} {c.source_b}={c.close_b:.4f} "
            f"dev={c.deviation * 100:.2f}%"
        )
        if fail_fast:
            raise ValueError(msg)
        logger.warning(msg)


def write_conflicts(conflicts: tuple[ConflictRecord, ...], *, resolution: str = "D") -> None:
    """Persist conflicts to data/provenance/{INSTRUMENT}/{RES}_conflicts.parquet."""
    if not conflicts:
        return
    by_inst: dict[str, list[dict]] = {}
    for c in conflicts:
        row = asdict(c)
        row["date"] = pd.Timestamp(c.date).isoformat()
        by_inst.setdefault(c.instrument_id, []).append(row)
    for instrument_id, rows in by_inst.items():
        path = provenance_dir() / _safe_instrument(instrument_id) / f"{resolution}_conflicts.parquet"
        _append_parquet(pd.DataFrame(rows), path)
