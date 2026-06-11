"""Unit tests for deployment.live.runtime.broker_clock (synthetic instants)."""
from __future__ import annotations

from datetime import date, datetime, time, timezone

from deployment.live.runtime import broker_clock as bc

# 2026-01-15 is a Thursday; 2026-01-17 is a Saturday (winter: ET=UTC-5, EET=UTC+2).
_THU_2110_UTC = datetime(2026, 1, 15, 21, 10, tzinfo=timezone.utc)  # 16:10 ET / 23:10 EET
_THU_1400ET_UTC = datetime(2026, 1, 15, 19, 0, tzinfo=timezone.utc)  # 14:00 ET (pre-close)
_SAT_2110_UTC = datetime(2026, 1, 17, 21, 10, tzinfo=timezone.utc)


def test_to_eastern_winter_offset() -> None:
    et = bc.to_eastern(datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc))
    assert (et.hour, et.minute) == (16, 0)  # 16:00 ET = US cash close


def test_to_broker_eet_matches_stored_2300() -> None:
    # The 16:00 ET close maps to broker stored time 23:00 (EET = UTC+2 in winter).
    eet = bc.to_broker(datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc), "ftmo")
    assert eet.hour == 23


def test_naive_datetime_assumed_utc() -> None:
    aware = bc.to_eastern(datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc))
    naive = bc.to_eastern(datetime(2026, 1, 15, 21, 0))
    assert (aware.hour, aware.minute) == (naive.hour, naive.minute)


def test_rollover_deadzone_is_broker_midnight_hour() -> None:
    # EET 00:30 (= 22:30 UTC winter) is inside the dead-zone; EET 23:00 is not.
    assert bc.in_rollover_deadzone(datetime(2026, 1, 15, 22, 30, tzinfo=timezone.utc), "ftmo")
    assert not bc.in_rollover_deadzone(datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc), "ftmo")


def test_parse_hhmm() -> None:
    assert bc.parse_hhmm("16:05") == time(16, 5)


def test_decision_clock_fires_once_per_day_after_close() -> None:
    clock = bc.DecisionClock(decision_time_et=time(16, 5))
    # First observation after the close on a weekday fires.
    assert clock.is_due(_THU_2110_UTC, last_session_date=None)
    # Same day, already decided -> does not re-fire.
    assert not clock.is_due(_THU_2110_UTC, last_session_date=date(2026, 1, 15))
    # Before the decision time -> not due.
    assert not clock.is_due(_THU_1400ET_UTC, last_session_date=None)


def test_decision_clock_skips_weekends() -> None:
    clock = bc.DecisionClock(decision_time_et=time(16, 5))
    assert not clock.is_due(_SAT_2110_UTC, last_session_date=None)
    assert not clock.is_trading_day(_SAT_2110_UTC)
    assert clock.is_trading_day(_THU_2110_UTC)


def test_decision_clock_next_session_after_prior_fires() -> None:
    clock = bc.DecisionClock(decision_time_et=time(16, 5))
    fri = datetime(2026, 1, 16, 21, 10, tzinfo=timezone.utc)  # Fri 16:10 ET
    assert clock.is_due(fri, last_session_date=date(2026, 1, 15))
    assert clock.session_date(fri) == date(2026, 1, 16)


def test_from_hhmm_constructor() -> None:
    clock = bc.DecisionClock.from_hhmm("18:00")
    assert clock.decision_time_et == time(18, 0)


def test_is_past_decision_is_latch_free() -> None:
    """``is_past_decision`` is the weekday/time gate WITHOUT a once-per-day latch."""
    clock = bc.DecisionClock(decision_time_et=time(16, 5))
    # Weekday at/after the decision time -> True (regardless of prior decisions).
    assert clock.is_past_decision(_THU_2110_UTC)
    # Before the decision time -> False.
    assert not clock.is_past_decision(_THU_1400ET_UTC)
    # Weekend -> False.
    assert not clock.is_past_decision(_SAT_2110_UTC)
    # No latch: repeated calls on the same day stay True (the strategy owns the
    # once-per-day latch via persisted decision-state, so a restart can resume).
    assert clock.is_past_decision(_THU_2110_UTC)
