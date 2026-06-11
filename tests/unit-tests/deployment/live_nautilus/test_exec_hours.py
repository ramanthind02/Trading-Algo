"""Unit tests for the execution-hours + quote-freshness gate.

Darwinex (signal) and FTMO (execution) keep different hours; orders must only go
out when the EXECUTION broker's market is open AND the quote is fresh.
"""
from __future__ import annotations

from datetime import datetime, timezone

from deployment.live.runtime import exec_hours

# June 2026 is EU summer time (EEST = UTC+3), so EET local = UTC + 3h.
# FTMO index session (US500.cash <- ES) is 01:00–23:15 EEST.
_OPEN = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)    # 15:00 EEST — open
_AFTER_CLOSE = datetime(2026, 6, 5, 21, 0, tzinfo=timezone.utc)  # 24:00 EEST — closed
_NEAR_CLOSE = datetime(2026, 6, 5, 20, 5, tzinfo=timezone.utc)   # 23:05 EEST — still open (<23:15)


def test_freshness() -> None:
    assert exec_hours.is_quote_fresh(10.0, max_age_secs=300.0)
    assert not exec_hours.is_quote_fresh(900.0, max_age_secs=300.0)
    assert not exec_hours.is_quote_fresh(None)
    assert not exec_hours.is_quote_fresh(-1.0)  # future/garbage age


def test_market_open_schedule() -> None:
    assert exec_hours.market_open("ftmo", "ES", _OPEN)
    assert exec_hours.market_open("ftmo", "ES", _NEAR_CLOSE)   # 23:05 < 23:15
    assert not exec_hours.market_open("ftmo", "ES", _AFTER_CLOSE)  # past 23:15


def test_weekend_closed() -> None:
    # Find a Saturday in EEST and assert closed.
    sat = datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)
    assert sat.weekday() == 5
    assert not exec_hours.market_open("ftmo", "ES", sat)


def test_tradeable_requires_open_and_fresh() -> None:
    # Open + fresh -> tradeable.
    assert exec_hours.tradeable("ftmo", "ES", _OPEN, quote_age_secs=30.0)
    # Open but stale -> defer.
    assert not exec_hours.tradeable("ftmo", "ES", _OPEN, quote_age_secs=10_000.0)
    # Closed even with a fresh quote -> defer.
    assert not exec_hours.tradeable("ftmo", "ES", _AFTER_CLOSE, quote_age_secs=1.0)
    # No quote -> defer.
    assert not exec_hours.tradeable("ftmo", "ES", _OPEN, quote_age_secs=None)


def test_darwinex_vs_ftmo_hours_differ_at_decision() -> None:
    """At 23:05 EEST the signal (Darwinex) is past its index close but FTMO is open."""
    # FTMO index open until 23:15; Darwinex index closes 23:00 -> closed at 23:05.
    assert exec_hours.market_open("ftmo", "ES", _NEAR_CLOSE)
    assert not exec_hours.market_open("darwinex", "ES", _NEAR_CLOSE)
