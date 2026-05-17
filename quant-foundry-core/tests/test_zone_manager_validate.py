from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from quantfoundry_core.zone_manager import (
    DuplicateZoneIds,
    Failure,
    OverlappingZones,
    StrategyZoneViolatesProjectTestBoundary,
    Success,
    ZoneSpec,
    ZoneSnapshot,
    ZoneType,
    validate_strategy_zones,
    validate_zones,
)

PROJECT_TEST_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
PROJECT_TEST_END = datetime(2021, 12, 31, tzinfo=timezone.utc)


def _snap(*specs: ZoneSpec) -> ZoneSnapshot:
    return ZoneSnapshot(
        project_test_start=PROJECT_TEST_START,
        project_test_end=PROJECT_TEST_END,
        strategy_zones=list(specs),
    )


def _spec(
    *,
    zid: UUID | None = None,
    start: datetime,
    end: datetime,
    zt: ZoneType = ZoneType.TRAIN,
    name: str = "z",
) -> ZoneSpec:
    return ZoneSpec(
        zone_id=zid or uuid4(),
        name=name,
        zone_type=zt,
        start_at_utc=start,
        end_at_utc=end,
    )


def test_disjoint_zones_ok() -> None:
    a = _spec(start=datetime(2020, 1, 1, tzinfo=timezone.utc), end=datetime(2020, 6, 30, tzinfo=timezone.utc))
    b = _spec(start=datetime(2020, 7, 1, tzinfo=timezone.utc), end=datetime(2020, 12, 31, tzinfo=timezone.utc))
    res = validate_zones(_snap(a, b))
    assert isinstance(res, Success)


def test_touching_inclusive_boundaries_overlap() -> None:
    t = datetime(2020, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    a = _spec(start=datetime(2020, 1, 1, tzinfo=timezone.utc), end=t)
    b = _spec(start=t, end=datetime(2020, 12, 31, tzinfo=timezone.utc))
    res = validate_zones(_snap(a, b))
    assert isinstance(res, Failure)
    assert isinstance(res.error, OverlappingZones)


def test_subset_overlap() -> None:
    a = _spec(start=datetime(2020, 1, 1, tzinfo=timezone.utc), end=datetime(2020, 12, 31, tzinfo=timezone.utc))
    b = _spec(start=datetime(2020, 6, 1, tzinfo=timezone.utc), end=datetime(2020, 8, 1, tzinfo=timezone.utc))
    res = validate_zones(_snap(a, b))
    assert isinstance(res, Failure)
    assert isinstance(res.error, OverlappingZones)


def test_duplicate_zone_ids() -> None:
    zid = uuid4()
    a = _spec(zid=zid, start=datetime(2020, 1, 1, tzinfo=timezone.utc), end=datetime(2020, 3, 1, tzinfo=timezone.utc))
    b = _spec(zid=zid, start=datetime(2020, 4, 1, tzinfo=timezone.utc), end=datetime(2020, 6, 1, tzinfo=timezone.utc))
    res = validate_zones(_snap(a, b))
    assert isinstance(res, Failure)
    assert isinstance(res.error, DuplicateZoneIds)
    assert res.error.zone_ids == frozenset({zid})


def test_validate_empty_strategy_zones_success() -> None:
    res = validate_zones(
        ZoneSnapshot(
            project_test_start=PROJECT_TEST_START,
            project_test_end=PROJECT_TEST_END,
            strategy_zones=[],
        )
    )
    assert isinstance(res, Success)


def test_validate_strategy_zones_empty_list_success() -> None:
    res = validate_strategy_zones([], project_test_start=PROJECT_TEST_START)
    assert isinstance(res, Success)


def test_strategy_zone_end_on_project_test_start_rejected() -> None:
    spec = _spec(
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=PROJECT_TEST_START,
    )
    res = validate_zones(_snap(spec))
    assert isinstance(res, Failure)
    assert isinstance(res.error, StrategyZoneViolatesProjectTestBoundary)


def test_strategy_zone_end_after_project_test_rejected() -> None:
    spec = _spec(
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2022, 1, 1, tzinfo=timezone.utc),
    )
    res = validate_zones(_snap(spec))
    assert isinstance(res, Failure)
    assert isinstance(res.error, StrategyZoneViolatesProjectTestBoundary)


def test_multiple_duplicate_ids_collected() -> None:
    z1 = uuid4()
    s1 = _spec(zid=z1, start=datetime(2020, 1, 1, tzinfo=timezone.utc), end=datetime(2020, 2, 1, tzinfo=timezone.utc))
    s2 = _spec(zid=z1, start=datetime(2020, 3, 1, tzinfo=timezone.utc), end=datetime(2020, 4, 1, tzinfo=timezone.utc))
    z2 = uuid4()
    s3 = _spec(zid=z2, start=datetime(2020, 5, 1, tzinfo=timezone.utc), end=datetime(2020, 6, 1, tzinfo=timezone.utc))
    s4 = _spec(zid=z2, start=datetime(2020, 7, 1, tzinfo=timezone.utc), end=datetime(2020, 8, 1, tzinfo=timezone.utc))
    res = validate_zones(_snap(s1, s2, s3, s4))
    assert isinstance(res, Failure)
    assert isinstance(res.error, DuplicateZoneIds)
    assert res.error.zone_ids == frozenset({z1, z2})
