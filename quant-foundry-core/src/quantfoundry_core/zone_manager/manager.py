from __future__ import annotations

import pandas as pd

from quantfoundry_core.zone_manager.errors import MaterializeError, ZoneValidationError
from quantfoundry_core.zone_manager.materialize import (
    BoundsCheckMode,
    ZoneMaterialization,
    materialize_zones,
)
from quantfoundry_core.zone_manager.models import ZoneSnapshot
from quantfoundry_core.zone_manager.result import Result
from quantfoundry_core.zone_manager.validate import validate_zones


class ZoneManager:
    """Facade over pure validation and materialization functions."""

    __slots__ = ("_bounds_check",)

    def __init__(
        self,
        *,
        bounds_check: BoundsCheckMode = BoundsCheckMode.STRUCTURAL,
    ) -> None:
        self._bounds_check = bounds_check

    def validate(self, snapshot: ZoneSnapshot) -> Result[None, ZoneValidationError]:
        return validate_zones(snapshot)

    def materialize(
        self,
        snapshot: ZoneSnapshot,
        df: pd.DataFrame,
        *,
        skip_validation: bool = False,
    ) -> Result[ZoneMaterialization, MaterializeError]:
        _ = self._bounds_check
        return materialize_zones(snapshot, df, skip_validation=skip_validation)
