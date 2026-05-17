from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping
from uuid import UUID

import pandas as pd

from quantfoundry_core.zone_manager.errors import (
    MaterializeError,
    MaterializeValidationFailed,
    NonDatetimeIndex,
    UnsortedDatetimeIndex,
)
from quantfoundry_core.zone_manager.models import ZoneSnapshot, ZoneSpec
from quantfoundry_core.zone_manager.result import Failure, Result, Success
from quantfoundry_core.zone_manager.validate import validate_zones


class BoundsCheckMode(Enum):
    """Reserved for future index coverage vs catalog checks."""

    STRUCTURAL = auto()


@dataclass(frozen=True)
class ZoneSliceStats:
    zone_id: UUID
    bar_count: int
    index_min: pd.Timestamp | None
    index_max: pd.Timestamp | None


@dataclass(frozen=True)
class ProjectTestSliceStats:
    bar_count: int
    index_min: pd.Timestamp | None
    index_max: pd.Timestamp | None


@dataclass(frozen=True)
class ZoneMaterialization:
    """Strategy-zone slices (Tier 2) plus project test window slice (Tier 1)."""

    strategy_ordered_slices: tuple[tuple[ZoneSpec, pd.DataFrame], ...]
    per_zone_stats: tuple[ZoneSliceStats, ...]
    project_test_frame: pd.DataFrame
    project_test_stats: ProjectTestSliceStats

    @property
    def ordered_slices(self) -> tuple[tuple[ZoneSpec, pd.DataFrame], ...]:
        """Alias matching historical name; strategy tiers only."""

        return self.strategy_ordered_slices

    @property
    def slices_by_id(self) -> Mapping[UUID, pd.DataFrame]:
        return dict((spec.zone_id, frame) for spec, frame in self.strategy_ordered_slices)


def _index_as_utcComparable(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _slice_range(
    idx_utc: pd.DatetimeIndex,
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, int, pd.Timestamp | None, pd.Timestamp | None]:
    mask = (idx_utc >= start) & (idx_utc <= end)
    sliced = df.loc[mask]
    bar_count = int(len(sliced))
    index_min = None if bar_count == 0 else sliced.index.min()
    index_max = None if bar_count == 0 else sliced.index.max()
    return sliced, bar_count, index_min, index_max


def materialize_zones(
    snapshot: ZoneSnapshot,
    df: pd.DataFrame,
    *,
    skip_validation: bool = False,
) -> Result[ZoneMaterialization, MaterializeError]:
    if not skip_validation:
        validation = validate_zones(snapshot)
        match validation:
            case Failure(err):
                return Failure(MaterializeValidationFailed(err))
            case Success(_):
                pass

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return Failure(NonDatetimeIndex(index_type_name=type(idx).__name__))
    if not idx.is_monotonic_increasing:
        return Failure(UnsortedDatetimeIndex())

    idx_utc = _index_as_utcComparable(idx)

    ordered: list[tuple[ZoneSpec, pd.DataFrame]] = []
    stats: list[ZoneSliceStats] = []
    for spec in snapshot.strategy_zones:
        start = pd.Timestamp(spec.start_at_utc)
        end = pd.Timestamp(spec.end_at_utc)
        sliced, bar_count, index_min, index_max = _slice_range(idx_utc, df, start, end)
        ordered.append((spec, sliced))
        stats.append(
            ZoneSliceStats(
                zone_id=spec.zone_id,
                bar_count=bar_count,
                index_min=index_min,
                index_max=index_max,
            )
        )

    pt_start = pd.Timestamp(snapshot.project_test_start)
    pt_end = pd.Timestamp(snapshot.project_test_end)
    pt_frame, pt_count, pt_min, pt_max = _slice_range(idx_utc, df, pt_start, pt_end)

    return Success(
        ZoneMaterialization(
            strategy_ordered_slices=tuple(ordered),
            per_zone_stats=tuple(stats),
            project_test_frame=pt_frame,
            project_test_stats=ProjectTestSliceStats(
                bar_count=pt_count,
                index_min=pt_min,
                index_max=pt_max,
            ),
        )
    )
