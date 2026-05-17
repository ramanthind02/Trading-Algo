from __future__ import annotations

from collections import Counter
from datetime import datetime
from itertools import combinations

from quantfoundry_core.zone_manager.errors import (
    DuplicateZoneIds,
    OverlappingZones,
    StrategyZoneViolatesProjectTestBoundary,
    ZoneValidationError,
)
from quantfoundry_core.zone_manager.models import UTC, ZoneSnapshot, ZoneSpec
from quantfoundry_core.zone_manager.result import Failure, Result, Success


def _intervals_overlap_inclusive(a: ZoneSpec, b: ZoneSpec) -> bool:
    a0, a1 = a.start_at_utc.astimezone(UTC), a.end_at_utc.astimezone(UTC)
    b0, b1 = b.start_at_utc.astimezone(UTC), b.end_at_utc.astimezone(UTC)
    return a0 <= b1 and b0 <= a1


def _strategy_zones_pre_project_test(spec: ZoneSpec, project_test_start: datetime) -> bool:
    return spec.end_at_utc.astimezone(UTC) < project_test_start.astimezone(UTC)


def validate_zones(snapshot: ZoneSnapshot) -> Result[None, ZoneValidationError]:
    p_start = snapshot.project_test_start
    zones = list(snapshot.strategy_zones)

    duplicate_ids = frozenset(
        zid for zid, count in Counter(z.zone_id for z in zones).items() if count > 1
    )
    if duplicate_ids:
        return Failure(DuplicateZoneIds(duplicate_ids))

    for spec in zones:
        if not _strategy_zones_pre_project_test(spec, p_start):
            return Failure(
                StrategyZoneViolatesProjectTestBoundary(zone=spec, project_test_start=p_start)
            )

    for x, y in combinations(zones, 2):
        if _intervals_overlap_inclusive(x, y):
            return Failure(OverlappingZones(zone_a=x, zone_b=y))
    return Success(None)


def validate_strategy_zones(
    specs: list[ZoneSpec],
    *,
    project_test_start: datetime,
) -> Result[None, ZoneValidationError]:
    """Validate strategy tiers only (overlaps, duplicates, pre-test boundary)."""

    duplicate_ids = frozenset(
        zid for zid, count in Counter(z.zone_id for z in specs).items() if count > 1
    )
    if duplicate_ids:
        return Failure(DuplicateZoneIds(duplicate_ids))

    for spec in specs:
        if not _strategy_zones_pre_project_test(spec, project_test_start):
            return Failure(
                StrategyZoneViolatesProjectTestBoundary(
                    zone=spec,
                    project_test_start=project_test_start,
                )
            )

    for x, y in combinations(specs, 2):
        if _intervals_overlap_inclusive(x, y):
            return Failure(OverlappingZones(zone_a=x, zone_b=y))
    return Success(None)
