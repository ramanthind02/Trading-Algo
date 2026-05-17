from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from quantfoundry_core.zone_manager.models import ZoneSpec


@dataclass(frozen=True)
class DuplicateZoneIds:
    zone_ids: frozenset[UUID]


@dataclass(frozen=True)
class OverlappingZones:
    """Inclusive intervals; touching at one instant counts as overlap."""

    zone_a: ZoneSpec
    zone_b: ZoneSpec


@dataclass(frozen=True)
class StrategyZoneViolatesProjectTestBoundary:
    """Strategy zone must satisfy ``end_at_utc < project_test_start`` (strict)."""

    zone: ZoneSpec
    project_test_start: datetime


ZoneValidationError = (
    DuplicateZoneIds | OverlappingZones | StrategyZoneViolatesProjectTestBoundary
)


@dataclass(frozen=True)
class NonDatetimeIndex:
    index_type_name: str


@dataclass(frozen=True)
class UnsortedDatetimeIndex:
    pass


@dataclass(frozen=True)
class MaterializeValidationFailed:
    """Structural validation failed before slicing."""

    error: ZoneValidationError


MaterializeError = NonDatetimeIndex | UnsortedDatetimeIndex | MaterializeValidationFailed
