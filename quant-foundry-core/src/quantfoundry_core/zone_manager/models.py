from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

UTC = ZoneInfo("UTC")


def _parse_utc_datetime(value: object) -> datetime:
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    if not isinstance(value, datetime):
        msg = f"expected datetime or str, got {type(value).__name__}"
        raise TypeError(msg)
    if value.tzinfo is None:
        msg = "datetime must be timezone-aware"
        raise ValueError(msg)
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, BeforeValidator(_parse_utc_datetime)]


class ZoneType(StrEnum):
    """Strategy-tier zone kind (Tier 2). Project holdout is not a ZoneType — see ZoneSnapshot."""

    TRAIN = "train"
    VALIDATION = "validation"


class ZoneSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    zone_id: UUID
    name: Annotated[str, Field(min_length=1)]
    zone_type: ZoneType
    start_at_utc: UtcDatetime
    end_at_utc: UtcDatetime

    @model_validator(mode="after")
    def _start_before_end(self) -> Self:
        if self.start_at_utc > self.end_at_utc:
            msg = "start_at_utc must be <= end_at_utc (after UTC normalization)"
            raise ValueError(msg)
        return self


class ZoneSnapshot(BaseModel):
    """Tier 1 (project test window) + Tier 2 (strategy train/validation only).

    See ``docs/SaaS/zone_manager.md`` §7.5.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, Field(ge=1)] = 1
    project_test_start: UtcDatetime
    project_test_end: UtcDatetime
    strategy_zones: list[ZoneSpec] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _project_test_window_nonempty(self) -> Self:
        if self.project_test_start > self.project_test_end:
            msg = "project_test_start must be <= project_test_end"
            raise ValueError(msg)
        return self
