"""Unit tests for the pure rollover-cycle schedule (broker wall-clock)."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from deployment.live.runtime.rollover_schedule import (
    RolloverParams,
    SymbolSession,
    exit_boundary_date,
    exit_time,
    in_entry_window,
    in_exit_window,
    is_triple_night,
    last_rollover,
    next_rollover,
    reopen_at,
    session_date,
)

P = RolloverParams()  # exit_lead=15, window=12, default reopen 01:00, entry band 180m
SESS_0100 = SymbolSession(open_h=1, open_m=0, close_h=0, close_m=0)   # 01:00 reopen, 00:00 close


def test_next_and_last_rollover_around_midnight() -> None:
    late = datetime(2026, 6, 8, 23, 45)          # Mon 23:45 broker
    assert next_rollover(late) == datetime(2026, 6, 9, 0, 0)
    assert last_rollover(late) == datetime(2026, 6, 8, 0, 0)

    deadzone = datetime(2026, 6, 9, 0, 30)        # just after the rollover
    assert next_rollover(deadzone) == datetime(2026, 6, 10, 0, 0)
    assert last_rollover(deadzone) == datetime(2026, 6, 9, 0, 0)


def test_exit_time_and_window() -> None:
    assert exit_time(datetime(2026, 6, 8, 23, 0), P) == datetime(2026, 6, 8, 23, 45)  # T-15 before 00:00
    # window is [exit_time, rollover) = [23:45, 00:00)
    assert in_exit_window(datetime(2026, 6, 8, 23, 50), P) is True
    assert in_exit_window(datetime(2026, 6, 8, 23, 45), P) is True   # opens exactly at T-15
    assert in_exit_window(datetime(2026, 6, 8, 23, 44), P) is False  # one minute too early
    assert in_exit_window(datetime(2026, 6, 8, 23, 30), P) is False
    # after the rollover (dead-zone) is not an exit window
    assert in_exit_window(datetime(2026, 6, 9, 0, 5), P) is False
    assert exit_boundary_date(datetime(2026, 6, 8, 23, 50)) == date(2026, 6, 9)


def test_exit_window_opens_at_t_minus_lead() -> None:
    # FTMO closes ~23:49; the window must already be open at 23:45 (market still up).
    assert in_exit_window(datetime(2026, 6, 8, 23, 45), P) is True
    assert in_exit_window(datetime(2026, 6, 8, 23, 48), P) is True


def test_session_date_and_reopen() -> None:
    deadzone = datetime(2026, 6, 9, 0, 30)
    assert session_date(deadzone) == date(2026, 6, 9)
    # default reopen 01:00, no settle
    assert reopen_at(deadzone, None, 0, P) == datetime(2026, 6, 9, 1, 0)
    # explicit session 01:00 + 5m settle (gold)
    assert reopen_at(deadzone, SESS_0100, 5, P) == datetime(2026, 6, 9, 1, 5)


def test_entry_window() -> None:
    # before reopen → not yet
    assert in_entry_window(datetime(2026, 6, 9, 0, 30), SESS_0100, 0, P) is False
    # just after reopen → enter
    assert in_entry_window(datetime(2026, 6, 9, 1, 1), SESS_0100, 0, P) is True
    # gold +5 settle: 01:01 is still before 01:05
    assert in_entry_window(datetime(2026, 6, 9, 1, 1), SESS_0100, 5, P) is False
    assert in_entry_window(datetime(2026, 6, 9, 1, 6), SESS_0100, 5, P) is True
    # past the 180m band → closed
    assert in_entry_window(datetime(2026, 6, 9, 5, 0), SESS_0100, 0, P) is False


@pytest.mark.parametrize(
    "d, triple_weekday, expected",
    [
        # indices: triple booked Friday (swap_rollover3days=5). Exit decided Friday.
        (datetime(2026, 6, 12, 23, 50), 5, True),    # 2026-06-12 is a Friday
        (datetime(2026, 6, 11, 23, 50), 5, False),   # Thursday → not triple for indices
        # metals/energy: triple booked Wednesday (=3). Exit decided Wednesday.
        (datetime(2026, 6, 10, 23, 50), 3, True),    # 2026-06-10 is a Wednesday
        (datetime(2026, 6, 12, 23, 50), 3, False),   # Friday → not triple for metals
    ],
)
def test_is_triple_night(d: datetime, triple_weekday: int, expected: bool) -> None:
    assert is_triple_night(d, triple_weekday) is expected


def test_triple_weekday_convention_matches_mt5() -> None:
    # MT5 swap_rollover3days: Sun=0 … Sat=6. Our broker_now Friday must map to 5.
    friday = datetime(2026, 6, 12, 23, 50)
    assert (friday.weekday() + 1) % 7 == 5
    wednesday = datetime(2026, 6, 10, 23, 50)
    assert (wednesday.weekday() + 1) % 7 == 3
