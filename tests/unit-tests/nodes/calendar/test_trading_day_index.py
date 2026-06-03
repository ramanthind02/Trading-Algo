"""Unit tests for trading-day offset indexing."""

from __future__ import annotations

from datetime import date

from utils.calendar.trading_day_index import TradingDayIndex


def _november_2024_sessions() -> tuple[date, ...]:
    return (
        date(2024, 11, 18),
        date(2024, 11, 19),
        date(2024, 11, 20),
        date(2024, 11, 21),
        date(2024, 11, 22),
        date(2024, 11, 25),
        date(2024, 11, 26),
        date(2024, 11, 27),
        date(2024, 11, 29),
    )


def test_offset_d0_is_zero() -> None:
    index = TradingDayIndex(_november_2024_sessions())
    d0 = date(2024, 11, 27)
    assert index.offset(d0, d0) == 0


def test_thanksgiving_equity_window_d_minus_4_to_d0() -> None:
    index = TradingDayIndex(_november_2024_sessions())
    d0 = date(2024, 11, 27)
    active = index.active_sessions((d0,), entry_offset=-4, exit_offset=0)
    assert active == frozenset(
        {
            date(2024, 11, 21),
            date(2024, 11, 22),
            date(2024, 11, 25),
            date(2024, 11, 26),
            date(2024, 11, 27),
        }
    )


def test_fomc_window_excludes_day_after_d0() -> None:
    index = TradingDayIndex(_november_2024_sessions())
    d0 = date(2024, 11, 27)
    active = index.active_sessions((d0,), entry_offset=-2, exit_offset=0)
    assert date(2024, 11, 25) in active
    assert date(2024, 11, 26) in active
    assert date(2024, 11, 27) in active
    assert date(2024, 11, 29) not in active
