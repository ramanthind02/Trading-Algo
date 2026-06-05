"""NYSE closure dates and D0 (last session before closure) for calendar holidays."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Iterable, Sequence


class HolidayAssetBucket(str, Enum):
    EQUITY = "equity"
    GOLD = "gold"


@dataclass(frozen=True)
class HolidaySpec:
    holiday_id: str
    asset_bucket: HolidayAssetBucket


_HOLIDAY_SPECS: tuple[HolidaySpec, ...] = (
    HolidaySpec("good_friday", HolidayAssetBucket.EQUITY),
    HolidaySpec("independence_day", HolidayAssetBucket.EQUITY),
    HolidaySpec("thanksgiving", HolidayAssetBucket.EQUITY),
    HolidaySpec("christmas", HolidayAssetBucket.GOLD),
    HolidaySpec("new_year", HolidayAssetBucket.GOLD),
    HolidaySpec("mlk_day", HolidayAssetBucket.GOLD),
    HolidaySpec("presidents_day", HolidayAssetBucket.GOLD),
)


def holiday_specs() -> tuple[HolidaySpec, ...]:
    return _HOLIDAY_SPECS


def nyse_full_closure_dates(start_year: int, end_year: int) -> dict[str, list[date]]:
    """Map holiday_id -> NYSE full-closure calendar dates in range."""
    out: dict[str, list[date]] = {spec.holiday_id: [] for spec in _HOLIDAY_SPECS}
    for year in range(start_year, end_year + 1):
        out["good_friday"].append(_good_friday(year))
        out["independence_day"].append(_observed_fixed(year, 7, 4))
        out["thanksgiving"].append(_thanksgiving(year))
        out["christmas"].append(_observed_fixed(year, 12, 25))
        out["new_year"].append(_observed_fixed(year, 1, 1))
        out["mlk_day"].append(_nth_weekday(year, 1, 0, 3))
        out["presidents_day"].append(_nth_weekday(year, 2, 0, 3))
    return out


def closure_to_d0(
    closure_dates: Iterable[date],
    trading_sessions: Sequence[date],
) -> dict[date, date]:
    """Map each closure date to the prior NYSE session (D0)."""
    session_set = set(trading_sessions)
    ordered = sorted(trading_sessions)
    index = {d: i for i, d in enumerate(ordered)}
    d0_map: dict[date, date] = {}
    for closure in closure_dates:
        if closure in session_set:
            pos = index[closure]
            if pos == 0:
                continue
            d0_map[closure] = ordered[pos - 1]
            continue
        prior = [s for s in ordered if s < closure]
        if prior:
            d0_map[closure] = prior[-1]
    return d0_map


def build_holiday_events(
    start_year: int,
    end_year: int,
    trading_sessions: Sequence[date],
) -> list[dict[str, str]]:
    """Build JSON-serializable holiday event records with trading-adjusted D0."""
    closures_by_id = nyse_full_closure_dates(start_year, end_year)
    events: list[dict[str, str]] = []
    for spec in _HOLIDAY_SPECS:
        closures = closures_by_id[spec.holiday_id]
        d0_by_closure = closure_to_d0(closures, trading_sessions)
        for closure in sorted(closures):
            d0 = d0_by_closure.get(closure)
            if d0 is None:
                continue
            events.append(
                {
                    "holiday_id": spec.holiday_id,
                    "asset_bucket": spec.asset_bucket.value,
                    "closure_date": closure.isoformat(),
                    "d0": d0.isoformat(),
                }
            )
    return events


def _observed_fixed(year: int, month: int, day: int) -> date:
    """Weekend observation: Sat->Fri, Sun->Mon (NYSE convention)."""
    d = date(year, month, day)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _thanksgiving(year: int) -> date:
    """Fourth Thursday in November — NYSE closed."""
    return _nth_weekday(year, 11, 3, 4)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """``weekday``: Monday=0 … Sunday=6; ``n`` is 1-based occurrence."""
    d = date(year, month, 1)
    count = 0
    while d.month == month:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)
    raise ValueError(f"No {n}th weekday={weekday} in {year}-{month:02d}")


def _good_friday(year: int) -> date:
    easter = _easter_sunday(year)
    return easter - timedelta(days=2)


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)
