"""Shared calendar-window signal logic for seasonal bias nodes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from data_platform.events.calendar_loader import CalendarBundle, HolidayAssetBucket, load_calendar_bundle
from data_platform.events.trading_day_index import TradingDayIndex, load_es_trading_sessions


def build_active_sessions(
    d0_dates: tuple[date, ...],
    entry_offset: int,
    exit_offset: int,
    trading_index: TradingDayIndex | None = None,
) -> frozenset[date]:
    index = trading_index or TradingDayIndex(load_es_trading_sessions())
    return index.active_sessions(d0_dates, entry_offset, exit_offset)


def load_bundle(repo_root: Path | None = None) -> CalendarBundle:
    return load_calendar_bundle(repo_root)


def holiday_d0_dates(
    bundle: CalendarBundle,
    bucket: HolidayAssetBucket,
) -> tuple[date, ...]:
    return tuple(
        sorted(
            event.d0
            for event in bundle.holiday_events
            if event.asset_bucket == bucket
        )
    )
