"""Unit tests for NYSE holiday closure rules."""

from __future__ import annotations

from datetime import date

from data_platform.events.nyse_holidays import build_holiday_events, closure_to_d0


def test_thanksgiving_2024_d0_is_prior_session() -> None:
    sessions = (
        date(2024, 11, 25),
        date(2024, 11, 26),
        date(2024, 11, 27),
        date(2024, 11, 29),
    )
    closure = date(2024, 11, 28)
    d0_map = closure_to_d0([closure], sessions)
    assert d0_map[closure] == date(2024, 11, 27)


def test_build_holiday_events_emits_records() -> None:
    sessions = [date(2024, 1, 2) + __import__("datetime").timedelta(days=i) for i in range(0, 300)]
    events = build_holiday_events(2024, 2024, sessions)
    assert events
    assert all("d0" in event and "holiday_id" in event for event in events)
