"""Unit tests for calendar JSON loading."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from data_platform.events.calendar_loader import (
    HolidayAssetBucket,
    load_calendar_bundle,
    repo_calendar_dir,
)


def test_load_calendar_bundle_has_fomc_and_holidays() -> None:
    bundle = load_calendar_bundle()
    assert len(bundle.fomc_decision_dates) >= 160
    assert bundle.fomc_decision_dates[0] >= date(2005, 1, 1)
    assert len(bundle.holiday_events) >= 140


def test_holiday_events_include_equity_and_gold_buckets() -> None:
    bundle = load_calendar_bundle()
    buckets = {event.asset_bucket for event in bundle.holiday_events}
    assert HolidayAssetBucket.EQUITY in buckets
    assert HolidayAssetBucket.GOLD in buckets


def test_repo_calendar_dir_points_at_data_calendar() -> None:
    root = Path(__file__).resolve().parents[4]
    assert (repo_calendar_dir(root) / "fomc_decision_dates.json").is_file()


def test_fomc_json_missing_dates_raises(tmp_path: Path) -> None:
    bad_dir = tmp_path / "calendar"
    bad_dir.mkdir()
    (bad_dir / "fomc_decision_dates.json").write_text('{"dates": "not-a-list"}', encoding="utf-8")
    (bad_dir / "nyse_holiday_events.json").write_text('{"events": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="dates"):
        from data_platform.events import calendar_loader

        calendar_loader._load_fomc_dates(bad_dir / "fomc_decision_dates.json")
