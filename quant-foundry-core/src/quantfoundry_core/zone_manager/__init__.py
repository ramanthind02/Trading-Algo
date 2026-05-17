"""Zone configuration validation and DataFrame slicing."""

from quantfoundry_core.zone_manager.errors import (
    DuplicateZoneIds,
    MaterializeError,
    MaterializeValidationFailed,
    NonDatetimeIndex,
    OverlappingZones,
    StrategyZoneViolatesProjectTestBoundary,
    UnsortedDatetimeIndex,
    ZoneValidationError,
)
from quantfoundry_core.zone_manager.manager import ZoneManager
from quantfoundry_core.zone_manager.materialize import (
    BoundsCheckMode,
    ProjectTestSliceStats,
    ZoneMaterialization,
    ZoneSliceStats,
    materialize_zones,
)
from quantfoundry_core.zone_manager.models import ZoneSpec, ZoneSnapshot, ZoneType
from quantfoundry_core.zone_manager.result import Failure, Result, Success
from quantfoundry_core.zone_manager.robustness import TestId, allowed_robustness_tests
from quantfoundry_core.zone_manager.validate import validate_strategy_zones, validate_zones

__all__ = (
    "BoundsCheckMode",
    "DuplicateZoneIds",
    "Failure",
    "MaterializeError",
    "MaterializeValidationFailed",
    "NonDatetimeIndex",
    "OverlappingZones",
    "ProjectTestSliceStats",
    "Result",
    "StrategyZoneViolatesProjectTestBoundary",
    "Success",
    "TestId",
    "UnsortedDatetimeIndex",
    "ZoneManager",
    "ZoneMaterialization",
    "ZoneSliceStats",
    "ZoneSnapshot",
    "ZoneSpec",
    "ZoneType",
    "ZoneValidationError",
    "allowed_robustness_tests",
    "materialize_zones",
    "validate_strategy_zones",
    "validate_zones",
)
