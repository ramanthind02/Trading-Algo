"""Tests for execution.us_market_session — pre-flight market-open check."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from execution.us_market_session import decide_session, _ET


def _et(year: int, month: int, day: int, hour: int = 15, minute: int = 30) -> datetime:
    """Helper: construct an ET-tz'd datetime then return UTC."""
    naive = datetime(year, month, day, hour, minute)
    return naive.replace(tzinfo=_ET).astimezone(timezone.utc)


# ─────────────────────────── Open-market cases ───────────────────────────


def test_normal_weekday_is_open() -> None:
    decision = decide_session(_et(2026, 6, 4))  # Thu
    assert decision.skip is False
    assert decision.reason is None


def test_monday_open_market_is_open() -> None:
    decision = decide_session(_et(2026, 6, 1))
    assert decision.skip is False


def test_april_1_weekday_no_special_close() -> None:
    decision = decide_session(_et(2026, 4, 1))  # Wed, no holiday
    assert decision.skip is False


# ─────────────────────────── Weekends ───────────────────────────


def test_saturday_skipped() -> None:
    decision = decide_session(_et(2026, 6, 6))
    assert decision.skip is True
    assert "Saturday" in (decision.reason or "")


def test_sunday_skipped() -> None:
    decision = decide_session(_et(2026, 6, 7))
    assert decision.skip is True
    assert "Sunday" in (decision.reason or "")


# ─────────────────────── Full-day NYSE closures ───────────────────────


def test_new_years_day_2026_is_thursday_full_close() -> None:
    decision = decide_session(_et(2026, 1, 1))
    assert decision.skip is True
    assert "New Year" in (decision.reason or "")


def test_new_years_day_2028_falls_saturday_observed_friday() -> None:
    decision = decide_session(_et(2027, 12, 31))  # 2028-01-01 is Sat, observed Fri 2027-12-31
    assert decision.skip is True
    assert "New Year" in (decision.reason or "")


def test_mlk_day_2026_third_monday_january() -> None:
    decision = decide_session(_et(2026, 1, 19))
    assert decision.skip is True
    assert "MLK" in (decision.reason or "")


def test_presidents_day_2026_third_monday_february() -> None:
    decision = decide_session(_et(2026, 2, 16))
    assert decision.skip is True
    assert "Presidents" in (decision.reason or "")


def test_good_friday_2026_is_april_3() -> None:
    decision = decide_session(_et(2026, 4, 3))
    assert decision.skip is True
    assert "Good Friday" in (decision.reason or "")


def test_memorial_day_2026_last_monday_may() -> None:
    decision = decide_session(_et(2026, 5, 25))
    assert decision.skip is True
    assert "Memorial" in (decision.reason or "")


def test_juneteenth_2026_falls_friday() -> None:
    decision = decide_session(_et(2026, 6, 19))
    assert decision.skip is True
    assert "Juneteenth" in (decision.reason or "")


def test_independence_day_2026_falls_saturday_observed_friday() -> None:
    decision = decide_session(_et(2026, 7, 3))
    assert decision.skip is True
    # July 3 2026 has TWO conditions: Independence Day observed (4th=Sat) AND
    # it would otherwise be the early-close day. Observed-holiday rule wins.
    assert "Independence" in (decision.reason or "")


def test_labor_day_2026_first_monday_september() -> None:
    decision = decide_session(_et(2026, 9, 7))
    assert decision.skip is True
    assert "Labor" in (decision.reason or "")


def test_thanksgiving_2026_fourth_thursday_november() -> None:
    decision = decide_session(_et(2026, 11, 26))
    assert decision.skip is True
    assert "Thanksgiving" in (decision.reason or "")


def test_christmas_2026_falls_friday_full_close() -> None:
    decision = decide_session(_et(2026, 12, 25))
    assert decision.skip is True
    assert "Christmas" in (decision.reason or "")


# ─────────────────────── Early-close (1pm ET) ───────────────────────


def test_black_friday_2026_is_early_close() -> None:
    decision = decide_session(_et(2026, 11, 27))  # day after Thanksgiving
    assert decision.skip is True
    assert "Day after Thanksgiving" in (decision.reason or "")


def test_christmas_eve_weekday_is_early_close_2027() -> None:
    decision = decide_session(_et(2027, 12, 24))  # Fri (Christmas 12/25 is Sat-observed Fri 24th in 2027? let's check)
    # 2027-12-25 is Sat → observed Fri 2027-12-24 → that's a full-day holiday, not early close
    assert decision.skip is True
    # Could match either Christmas (observed) or Christmas Eve — observed-holiday wins
    reason = decision.reason or ""
    assert "Christmas" in reason


def test_christmas_eve_2024_thursday_early_close() -> None:
    # 2024-12-24 is Tue, 2024-12-25 is Wed → CE is plain weekday early close
    decision = decide_session(_et(2024, 12, 24))
    assert decision.skip is True
    assert "Christmas Eve" in (decision.reason or "")


def test_july_3_2025_thursday_early_close() -> None:
    # 2025-07-04 is Fri (weekday), 2025-07-03 is Thu → early close
    decision = decide_session(_et(2025, 7, 3))
    assert decision.skip is True
    assert "July 3" in (decision.reason or "")


# ─────────────────────── now_utc handling ───────────────────────


def test_uses_system_clock_when_none() -> None:
    # Just verify the call doesn't crash with default None
    decision = decide_session()
    assert decision.skip in (True, False)


def test_naive_datetime_treated_as_utc() -> None:
    naive = datetime(2026, 6, 6, 19, 30)  # Sat 19:30 UTC = Sat 15:30 ET
    decision = decide_session(naive)
    assert decision.skip is True
    assert "Saturday" in (decision.reason or "")
