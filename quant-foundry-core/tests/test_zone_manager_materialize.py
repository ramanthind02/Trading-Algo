from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from quantfoundry_core.zone_manager import (
    Failure,
    MaterializeValidationFailed,
    NonDatetimeIndex,
    Success,
    UnsortedDatetimeIndex,
    ZoneManager,
    ZoneSpec,
    ZoneSnapshot,
    ZoneType,
    materialize_zones,
)

PROJECT_TEST_START = datetime(2020, 1, 11, tzinfo=timezone.utc)
PROJECT_TEST_END = datetime(2020, 1, 20, tzinfo=timezone.utc)


def _daily_df(*, start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq="D", tz="UTC")
    return pd.DataFrame({"close": range(periods)}, index=idx)


def _snapshot(z_train: ZoneSpec, z_val: ZoneSpec) -> ZoneSnapshot:
    return ZoneSnapshot(
        project_test_start=PROJECT_TEST_START,
        project_test_end=PROJECT_TEST_END,
        strategy_zones=[z_train, z_val],
    )


def test_materialize_two_zones_and_project_test() -> None:
    df = _daily_df(start="2020-01-01", periods=20)
    z_train = ZoneSpec(
        zone_id=uuid4(),
        name="train",
        zone_type=ZoneType.TRAIN,
        start_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_at_utc=datetime(2020, 1, 5, tzinfo=timezone.utc),
    )
    z_val = ZoneSpec(
        zone_id=uuid4(),
        name="val",
        zone_type=ZoneType.VALIDATION,
        start_at_utc=datetime(2020, 1, 6, tzinfo=timezone.utc),
        end_at_utc=datetime(2020, 1, 10, tzinfo=timezone.utc),
    )
    snap = _snapshot(z_train, z_val)
    res = materialize_zones(snap, df)
    assert isinstance(res, Success)
    mat = res.value
    assert mat.per_zone_stats[0].bar_count == 5
    assert mat.per_zone_stats[1].bar_count == 5
    assert len(mat.slices_by_id[z_train.zone_id]) == 5
    assert mat.project_test_stats.bar_count == 10
    assert len(mat.project_test_frame) == 10


def test_materialize_unsorted_index_fails() -> None:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-02", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC")]
    )
    df = pd.DataFrame({"x": [1, 2]}, index=idx)
    z = ZoneSpec(
        zone_id=uuid4(),
        name="a",
        zone_type=ZoneType.TRAIN,
        start_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    snap = ZoneSnapshot(
        project_test_start=datetime(2020, 1, 2, tzinfo=timezone.utc),
        project_test_end=datetime(2020, 1, 2, tzinfo=timezone.utc),
        strategy_zones=[z],
    )
    res = materialize_zones(snap, df, skip_validation=True)
    assert isinstance(res, Failure)
    assert isinstance(res.error, UnsortedDatetimeIndex)


def test_materialize_non_datetime_index_fails() -> None:
    df = pd.DataFrame({"x": [1, 2]}, index=[0, 1])
    z = ZoneSpec(
        zone_id=uuid4(),
        name="a",
        zone_type=ZoneType.TRAIN,
        start_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    snap = ZoneSnapshot(
        project_test_start=datetime(2020, 1, 2, tzinfo=timezone.utc),
        project_test_end=datetime(2020, 1, 2, tzinfo=timezone.utc),
        strategy_zones=[z],
    )
    res = materialize_zones(snap, df, skip_validation=True)
    assert isinstance(res, Failure)
    assert isinstance(res.error, NonDatetimeIndex)


def test_materialize_empty_strategy_slice_allowed() -> None:
    df = _daily_df(start="2020-01-01", periods=20)
    z = ZoneSpec(
        zone_id=uuid4(),
        name="empty",
        zone_type=ZoneType.VALIDATION,
        start_at_utc=datetime(2019, 6, 1, tzinfo=timezone.utc),
        end_at_utc=datetime(2019, 12, 31, tzinfo=timezone.utc),
    )
    snap = ZoneSnapshot(
        project_test_start=PROJECT_TEST_START,
        project_test_end=PROJECT_TEST_END,
        strategy_zones=[z],
    )
    res = materialize_zones(snap, df)
    assert isinstance(res, Success)
    assert res.value.per_zone_stats[0].bar_count == 0
    assert res.value.per_zone_stats[0].index_min is None
    assert res.value.project_test_stats.bar_count == 10


def test_materialize_wraps_validation_failure() -> None:
    t = datetime(2020, 1, 5, tzinfo=timezone.utc)
    a = ZoneSpec(
        zone_id=uuid4(),
        name="a",
        zone_type=ZoneType.TRAIN,
        start_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_at_utc=t,
    )
    b = ZoneSpec(
        zone_id=uuid4(),
        name="b",
        zone_type=ZoneType.VALIDATION,
        start_at_utc=t,
        end_at_utc=datetime(2020, 1, 10, tzinfo=timezone.utc),
    )
    df = _daily_df(start="2020-01-01", periods=20)
    snap = ZoneSnapshot(
        project_test_start=PROJECT_TEST_START,
        project_test_end=PROJECT_TEST_END,
        strategy_zones=[a, b],
    )
    res = materialize_zones(snap, df)
    assert isinstance(res, Failure)
    assert isinstance(res.error, MaterializeValidationFailed)


def test_naive_index_treated_as_utc() -> None:
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.DataFrame({"x": range(20)}, index=idx)
    z = ZoneSpec(
        zone_id=uuid4(),
        name="n",
        zone_type=ZoneType.TRAIN,
        start_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_at_utc=datetime(2020, 1, 3, tzinfo=timezone.utc),
    )
    snap = ZoneSnapshot(
        project_test_start=PROJECT_TEST_START,
        project_test_end=PROJECT_TEST_END,
        strategy_zones=[z],
    )
    res = ZoneManager().materialize(snap, df)
    assert isinstance(res, Success)
    assert res.value.per_zone_stats[0].bar_count == 3


def test_inclusive_end_includes_last_bar() -> None:
    idx = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    df = pd.DataFrame({"x": range(20)}, index=idx)
    z = ZoneSpec(
        zone_id=uuid4(),
        name="e",
        zone_type=ZoneType.TRAIN,
        start_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_at_utc=datetime(2020, 1, 3, tzinfo=timezone.utc),
    )
    snap = ZoneSnapshot(
        project_test_start=PROJECT_TEST_START,
        project_test_end=PROJECT_TEST_END,
        strategy_zones=[z],
    )
    res = materialize_zones(snap, df)
    assert isinstance(res, Success)
    assert res.value.per_zone_stats[0].bar_count == 3
    assert list(res.value.strategy_ordered_slices[0][1]["x"]) == [0, 1, 2]


def test_materialize_empty_strategy_zones_only_project_test() -> None:
    df = _daily_df(start="2020-01-01", periods=15)
    snap = ZoneSnapshot(
        project_test_start=datetime(2020, 1, 11, tzinfo=timezone.utc),
        project_test_end=datetime(2020, 1, 15, tzinfo=timezone.utc),
        strategy_zones=[],
    )
    res = materialize_zones(snap, df)
    assert isinstance(res, Success)
    assert res.value.strategy_ordered_slices == ()
    assert res.value.per_zone_stats == ()
    assert res.value.project_test_stats.bar_count == 5
