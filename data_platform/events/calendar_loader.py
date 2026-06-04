"""Load committed calendar JSON artifacts for bias nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path


class HolidayAssetBucket(str, Enum):
    EQUITY = "equity"
    GOLD = "gold"


@dataclass(frozen=True)
class HolidayEvent:
    holiday_id: str
    asset_bucket: HolidayAssetBucket
    closure_date: date
    d0: date


@dataclass(frozen=True)
class CalendarBundle:
    fomc_decision_dates: tuple[date, ...]
    holiday_events: tuple[HolidayEvent, ...]


def repo_calendar_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or _default_repo_root()
    return root / "data" / "events" / "calendar"


def load_calendar_bundle(repo_root: Path | None = None) -> CalendarBundle:
    cal_dir = repo_calendar_dir(repo_root)
    fomc_path = cal_dir / "fomc_decision_dates.json"
    holiday_path = cal_dir / "nyse_holiday_events.json"
    fomc_dates = _load_fomc_dates(fomc_path)
    holidays = _load_holiday_events(holiday_path)
    return CalendarBundle(fomc_decision_dates=fomc_dates, holiday_events=holidays)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_fomc_dates(path: Path) -> tuple[date, ...]:
    payload = _read_json(path)
    raw_dates = payload.get("dates")
    if not isinstance(raw_dates, list):
        raise ValueError(f"{path}: missing 'dates' list")
    parsed = tuple(sorted({_parse_iso_date(str(d)) for d in raw_dates}))
    return parsed


def _load_holiday_events(path: Path) -> tuple[HolidayEvent, ...]:
    payload = _read_json(path)
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError(f"{path}: missing 'events' list")
    events = tuple(_parse_holiday_event(item, path) for item in raw_events)
    return events


def _parse_holiday_event(item: object, path: Path) -> HolidayEvent:
    if not isinstance(item, dict):
        raise ValueError(f"{path}: holiday event must be object")
    holiday_id = str(item.get("holiday_id", ""))
    bucket_raw = str(item.get("asset_bucket", ""))
    closure_raw = item.get("closure_date")
    d0_raw = item.get("d0")
    if not holiday_id or not bucket_raw:
        raise ValueError(f"{path}: holiday event missing id or asset_bucket")
    if closure_raw is None or d0_raw is None:
        raise ValueError(f"{path}: holiday {holiday_id!r} missing closure_date or d0")
    return HolidayEvent(
        holiday_id=holiday_id,
        asset_bucket=HolidayAssetBucket(bucket_raw),
        closure_date=_parse_iso_date(str(closure_raw)),
        d0=_parse_iso_date(str(d0_raw)),
    )


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _read_json(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected JSON object")
    return loaded
