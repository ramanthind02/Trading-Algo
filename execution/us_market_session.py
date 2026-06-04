"""
US market session check for the CFD prop scheduler.

Purpose: tell ``scripts/enigma_cfd_prop_forecast.py`` and
``scripts/enigma_cfd_prop_weekend_close.py`` when to no-op because US
cash equity markets are closed (so US100.cash / US500.cash have wide
spreads and stale prices, and any forecast is feeding off a non-fresh
MT5 D1 bar).

The CFD prop scripts run at ~15:30 ET. For our purposes "skip" means:

- US equity full-day closure (NYSE/CBOE): 10 calendar holidays observed
- US equity early close (1pm ET) when the run time of 15:30 ET is
  AFTER the close:
    * Day after Thanksgiving ("Black Friday")
    * July 3 when July 4 is observed Mon-Thu (early-close day)
    * Christmas Eve when it falls on a weekday

(XAUUSD / XAGUSD trade nearly 24/5 even on US holidays, but the
spreads on US indices on a closed day are punitive and rebalancing
on stale data is exactly what we want to avoid. Skip the whole run.)

This module does NOT decide trade direction or sizing; that's
upstream's job. It's a pre-flight "is the market open enough for us
to rebalance" gate.

The Easter / Good Friday computation and the ``nth_weekday`` /
``observed_fixed`` helpers are duplicated rather than imported from
``utils.calendar.nyse_holidays`` to keep this runtime module
self-contained (the research module is for a different purpose and
has a narrower hardcoded holiday list).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

_ET = ZoneInfo("America/New_York")

_RUN_HOUR_ET = 15
_RUN_MINUTE_ET = 30


@dataclass(frozen=True)
class SessionDecision:
    skip: bool
    reason: Optional[str]


def decide_session(now_utc: Optional[datetime] = None) -> SessionDecision:
    """Returns ``(skip=True, reason=...)`` if the CFD prop scripts should
    NOT execute today; otherwise ``(skip=False, reason=None)``.

    ``now_utc``: optional UTC override for tests. Defaults to system clock.
    """
    if now_utc is None:
        now_utc = datetime.now(tz=timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    now_et = now_utc.astimezone(_ET)
    today_et = now_et.date()

    if today_et.weekday() >= 5:
        return SessionDecision(skip=True, reason=f"weekend ({_weekday_name(today_et)})")

    closure_label = _full_closure_label(today_et)
    if closure_label is not None:
        return SessionDecision(skip=True, reason=f"NYSE full close: {closure_label}")

    early_close_label = _early_close_label(today_et)
    if early_close_label is not None:
        return SessionDecision(skip=True, reason=f"NYSE early close (13:00 ET): {early_close_label}")

    return SessionDecision(skip=False, reason=None)


def _full_closure_label(d: date) -> Optional[str]:
    year = d.year
    if d == _observed_fixed(year, 1, 1):
        return "New Year's Day"
    if d == _observed_fixed(year + 1, 1, 1):
        return "New Year's Day"
    if d == _nth_weekday(year, 1, 0, 3):
        return "MLK Day"
    if d == _nth_weekday(year, 2, 0, 3):
        return "Presidents Day"
    if d == _good_friday(year):
        return "Good Friday"
    if d == _last_weekday(year, 5, 0):
        return "Memorial Day"
    if d == _observed_fixed(year, 6, 19):
        return "Juneteenth"
    if d == _observed_fixed(year, 7, 4):
        return "Independence Day"
    if d == _nth_weekday(year, 9, 0, 1):
        return "Labor Day"
    if d == _nth_weekday(year, 11, 3, 4):
        return "Thanksgiving"
    if d == _observed_fixed(year, 12, 25):
        return "Christmas"
    return None


def _early_close_label(d: date) -> Optional[str]:
    year = d.year
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    if d == thanksgiving + timedelta(days=1):
        return "Day after Thanksgiving"

    july_4_observed = _observed_fixed(year, 7, 4)
    july_3 = date(year, 7, 3)
    if (
        d == july_3
        and july_3.weekday() < 5
        and july_4_observed == date(year, 7, 4)
        and date(year, 7, 4).weekday() < 5
    ):
        return "July 3 (pre-Independence Day)"

    christmas_eve = date(year, 12, 24)
    if d == christmas_eve and christmas_eve.weekday() < 5 and date(year, 12, 25).weekday() < 5:
        return "Christmas Eve"

    return None


def _weekday_name(d: date) -> str:
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d.weekday()]


def _observed_fixed(year: int, month: int, day: int) -> date:
    """NYSE weekend observation: Sat->Fri, Sun->Mon."""
    d = date(year, month, day)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    count = 0
    while d.month == month:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)
    raise ValueError(f"No {n}th weekday={weekday} in {year}-{month:02d}")


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    while last_day.weekday() != weekday:
        last_day -= timedelta(days=1)
    return last_day


def _good_friday(year: int) -> date:
    easter = _easter_sunday(year)
    return easter - timedelta(days=2)


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d_ = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d_ - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)
