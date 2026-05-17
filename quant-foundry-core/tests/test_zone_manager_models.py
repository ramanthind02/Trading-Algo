from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from quantfoundry_core.zone_manager import ZoneSnapshot, ZoneSpec, ZoneType

PT_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
PT_END = datetime(2021, 12, 31, tzinfo=timezone.utc)


def test_zone_snapshot_json_round_trip() -> None:
    zid = uuid4()
    spec = ZoneSpec(
        zone_id=zid,
        name="train",
        zone_type=ZoneType.TRAIN,
        start_at_utc=datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_at_utc=datetime(2020, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    snap = ZoneSnapshot(
        schema_version=1,
        project_test_start=PT_START,
        project_test_end=PT_END,
        strategy_zones=[spec],
        metadata={"k": 1},
    )
    raw = snap.model_dump_json()
    restored = ZoneSnapshot.model_validate_json(raw)
    assert restored == snap
    assert restored.strategy_zones[0].zone_id == zid


def test_zone_spec_rejects_start_after_end() -> None:
    with pytest.raises(ValidationError):
        ZoneSpec(
            zone_id=uuid4(),
            name="x",
            zone_type=ZoneType.TRAIN,
            start_at_utc=datetime(2021, 1, 2, tzinfo=timezone.utc),
            end_at_utc=datetime(2021, 1, 1, tzinfo=timezone.utc),
        )


def test_zone_spec_normalizes_to_utc() -> None:
    from zoneinfo import ZoneInfo

    spec = ZoneSpec(
        zone_id=uuid4(),
        name="t",
        zone_type=ZoneType.TRAIN,
        start_at_utc=datetime(2020, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        end_at_utc=datetime(2020, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        + timedelta(hours=1),
    )
    assert spec.start_at_utc.tzinfo == ZoneInfo("UTC")
    assert spec.end_at_utc.tzinfo == ZoneInfo("UTC")
    assert spec.start_at_utc == datetime(2020, 1, 1, 5, 0, 0, tzinfo=ZoneInfo("UTC"))
    assert spec.end_at_utc == datetime(2020, 1, 1, 6, 0, 0, tzinfo=ZoneInfo("UTC"))


def test_empty_strategy_zones_snapshot_allowed() -> None:
    snap = ZoneSnapshot(project_test_start=PT_START, project_test_end=PT_END, strategy_zones=[])
    assert snap.strategy_zones == []


def test_project_test_window_rejects_start_after_end() -> None:
    with pytest.raises(ValidationError):
        ZoneSnapshot(
            project_test_start=PT_END,
            project_test_end=PT_START,
            strategy_zones=[],
        )
