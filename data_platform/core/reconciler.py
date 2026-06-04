"""Source-priority reconciliation — single call site for merging source batches.

Given the canonical existing daily frame for an instrument and incoming batches
keyed by source, the reconciler:
  1. asks SourcePriorityConfig which source wins for (instrument_class, resolution),
  2. for the IB case, applies the existing append-only + junction-ratio splice
     (``prepare_ib_rows_for_central_cache_append`` in utils/cache/runtime — the
     runtime/engine layer, which this layer calls but does not own),
  3. for the Norgate/MT5 case, passes rows through append-only,
  4. flags overlap conflicts (close deviation above threshold),
  5. returns the merged frame + provenance records.

This generalises the ad-hoc ``upsert_tws_candles`` direct-call pattern. It does
not modify the existing live path; ``upsert_tws_candles`` can be refactored to
call ``reconcile_daily_batch`` with no behaviour change.

See docs/library/Data/multi_source_update_architecture.md §7.3.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from utils.cache.runtime.ib_candle_ratio_align import (
    prepare_ib_rows_for_central_cache_append,
)

from .catalog import InstrumentCatalog
from .enums import BarAggregation, InstrumentClass
from .identifiers import InstrumentId
from .source_priority import SourcePriorityConfig

DEFAULT_CONFLICT_THRESHOLD = 0.005  # 0.5% close deviation

_RESOLUTION_TO_AGG = {
    "D": BarAggregation.DAY,
    "W": BarAggregation.WEEK,
    "M": BarAggregation.MONTH,
}


@dataclass(frozen=True)
class ProvenanceRecord:
    instrument_id: str
    resolution: str
    source: str
    ts_ingest: datetime
    ratio_applied: bool
    ratio_value: float | None
    rows_written: int


@dataclass(frozen=True)
class ConflictRecord:
    instrument_id: str
    date: pd.Timestamp
    source_a: str
    source_b: str
    close_a: float
    close_b: float
    deviation: float


@dataclass(frozen=True)
class ReconcileResult:
    merged: pd.DataFrame
    provenance: tuple[ProvenanceRecord, ...]
    conflicts: tuple[ConflictRecord, ...]
    active_source: str | None
    skip_reason: str | None


class SourcePriorityReconciler:
    def __init__(
        self,
        config: SourcePriorityConfig,
        catalog: InstrumentCatalog,
        *,
        conflict_threshold: float = DEFAULT_CONFLICT_THRESHOLD,
    ) -> None:
        self._config = config
        self._catalog = catalog
        self._conflict_threshold = conflict_threshold

    def reconcile_daily_batch(
        self,
        instrument_id: InstrumentId | str,
        existing_df: pd.DataFrame,
        incoming_batches: dict[str, pd.DataFrame],
        *,
        resolution: str = "D",
    ) -> ReconcileResult:
        inst = self._catalog.find(instrument_id)
        if inst is None:
            raise ValueError(f"Instrument {instrument_id} not in catalog")

        agg = _RESOLUTION_TO_AGG.get(resolution, BarAggregation.DAY)
        active = self._config.active_source(inst.instrument_class, agg)
        ts_ingest = datetime.now(timezone.utc)

        if active is None or active not in incoming_batches:
            return ReconcileResult(
                merged=existing_df, provenance=(), conflicts=(),
                active_source=active, skip_reason="no_active_source_batch",
            )

        incoming = incoming_batches[active]
        conflicts = self._detect_conflicts(str(inst.id), existing_df, incoming)

        if active == "ib":
            result = prepare_ib_rows_for_central_cache_append(
                existing_df, incoming, apply_junction_ratio=True,
            )
            kept = result.candles_df
            ratio_applied, ratio_value = result.applied_ratio, result.ratio
            skip_reason = result.skip_reason
        else:
            # Norgate / MT5: append-only (no junction ratio), no scaling.
            result = prepare_ib_rows_for_central_cache_append(
                existing_df, incoming, apply_junction_ratio=False,
            )
            kept = result.candles_df
            ratio_applied, ratio_value = False, None
            skip_reason = result.skip_reason

        merged = _append(existing_df, kept)
        prov = ProvenanceRecord(
            instrument_id=str(inst.id), resolution=resolution, source=active,
            ts_ingest=ts_ingest, ratio_applied=ratio_applied,
            ratio_value=ratio_value, rows_written=len(kept),
        )
        return ReconcileResult(
            merged=merged, provenance=(prov,), conflicts=conflicts,
            active_source=active, skip_reason=skip_reason,
        )

    def _detect_conflicts(
        self, instrument_id: str, existing_df: pd.DataFrame, incoming: pd.DataFrame,
    ) -> tuple[ConflictRecord, ...]:
        if existing_df.empty or incoming.empty:
            return ()
        ex = _ensure_dt_index(existing_df)
        inc = _ensure_dt_index(incoming)
        overlap = ex.index.intersection(inc.index)
        out: list[ConflictRecord] = []
        for dt in overlap:
            ca, cb = float(ex.loc[dt, "close"]), float(inc.loc[dt, "close"])
            if ca == 0:
                continue
            dev = abs(ca - cb) / abs(ca)
            if dev > self._conflict_threshold:
                out.append(ConflictRecord(instrument_id, dt, "existing", "incoming", ca, cb, dev))
        return tuple(out)


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    col = "datetime" if "datetime" in out.columns else ("date" if "date" in out.columns else None)
    if col is not None:
        out = out.set_index(pd.to_datetime(out[col]))
    return out


def _append(existing_df: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    """Append kept rows to existing, normalising both to a DatetimeIndex.

    ``prepare_ib_rows_for_central_cache_append`` returns kept rows with a
    ``datetime`` *column* (not index); align it to the existing frame's index
    convention so the merged frame is comparable/sortable by date.
    """
    if kept.empty:
        return existing_df
    kept_idx = _ensure_dt_index(kept)
    if "datetime" in kept_idx.columns:
        kept_idx = kept_idx.drop(columns=["datetime"])
    if existing_df.empty:
        return kept_idx.sort_index()
    existing_idx = _ensure_dt_index(existing_df)
    merged = pd.concat([existing_idx, kept_idx], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged
