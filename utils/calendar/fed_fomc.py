"""Parse FOMC decision dates from Federal Reserve HTML pages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from bs4 import BeautifulSoup

_MEETING_PDF_RE = re.compile(r"FOMC(\d{8})meeting\.pdf", re.IGNORECASE)
_HISTORICAL_YEAR_HREF_RE = re.compile(r"fomchistorical(\d{4})\.htm", re.IGNORECASE)
_YEAR_PANEL_RE = re.compile(r"^(20\d{2})\s+FOMC\s+Meetings$", re.IGNORECASE)

_MONTH_NAMES: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


@dataclass(frozen=True)
class FomcScrapeResult:
    dates: tuple[date, ...]
    warnings: tuple[str, ...]


def parse_historical_year_html(html: str) -> tuple[date, ...]:
    """Extract decision days from ``FOMC{YYYYMMDD}meeting.pdf`` transcript links."""
    stamps = sorted({m.group(1) for m in _MEETING_PDF_RE.finditer(html)})
    return tuple(_stamp_to_date(s) for s in stamps)


def parse_historical_index_years(html: str) -> tuple[int, ...]:
    """Return sorted years linked from ``fomc_historical_year.htm``."""
    years = {int(m.group(1)) for m in _HISTORICAL_YEAR_HREF_RE.finditer(html)}
    return tuple(sorted(years))


def parse_forward_calendar_html(html: str, min_year: int, max_year: int) -> tuple[date, ...]:
    """Parse ``fomccalendars.htm`` year panels into decision (last meeting) days."""
    soup = BeautifulSoup(html, "html.parser")
    dates: list[date] = []
    for h4 in soup.find_all("h4"):
        title = h4.get_text(strip=True)
        match = _YEAR_PANEL_RE.match(title)
        if match is None:
            continue
        year = int(match.group(1))
        if year < min_year or year > max_year:
            continue
        panel = h4.find_parent("div", class_="panel")
        if panel is None:
            continue
        dates.extend(_parse_calendar_panel(panel, year))
    return tuple(sorted(set(dates)))


def merge_fomc_dates(*groups: Iterable[date]) -> tuple[date, ...]:
    merged = sorted({d for group in groups for d in group})
    return tuple(merged)


def validate_fomc_dates(
    dates: tuple[date, ...],
    *,
    expected_2024: tuple[str, ...] = (
        "2024-01-31",
        "2024-03-20",
        "2024-05-01",
        "2024-06-12",
        "2024-07-31",
        "2024-09-18",
        "2024-11-07",
        "2024-12-18",
    ),
) -> tuple[str, ...]:
    """Return validation warning messages (empty if all checks pass)."""
    warnings: list[str] = []
    by_year: dict[int, list[date]] = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)

    for year in range(2005, 2021):
        count = len(by_year.get(year, []))
        if count == 0:
            warnings.append(f"{year}: no FOMC dates parsed")
        elif year != 2020 and count != 8:
            warnings.append(f"{year}: expected 8 meetings, got {count}")
        elif year == 2020 and count < 8:
            warnings.append(f"2020: expected at least 8 meetings, got {count}")

    for year in range(2021, 2027):
        count = len(by_year.get(year, []))
        if count == 0:
            warnings.append(f"{year}: no FOMC dates parsed")
        elif count != 8:
            warnings.append(f"{year}: expected 8 scheduled meetings, got {count}")

    iso_set = {d.isoformat() for d in dates}
    missing_2024 = [d for d in expected_2024 if d not in iso_set]
    if missing_2024:
        warnings.append(f"2024 sanity check missing dates: {missing_2024}")

    return tuple(warnings)


def dates_to_iso_strings(dates: Iterable[date]) -> list[str]:
    return [d.isoformat() for d in sorted(set(dates))]


def _parse_calendar_panel(panel: object, year: int) -> list[date]:
    soup_panel = panel  # BeautifulSoup element
    rows = soup_panel.find_all("div", class_="row")  # type: ignore[union-attr]
    out: list[date] = []
    for row in rows:
        cols = row.find_all("div")  # type: ignore[union-attr]
        texts = [c.get_text(" ", strip=True) for c in cols if c.get_text(strip=True)]
        if len(texts) < 2:
            continue
        month_label, day_cell = texts[0], texts[1]
        if "notation vote" in day_cell.lower():
            continue
        parsed = _parse_meeting_end_date(year, month_label, day_cell)
        if parsed is not None:
            out.append(parsed)
    return out


def _parse_meeting_end_date(year: int, month_label: str, day_cell: str) -> date | None:
    day_token = day_cell.split()[0].replace("*", "").strip()
    if not re.match(r"^\d{1,2}-\d{1,2}$", day_token):
        return None
    start_day_s, end_day_s = day_token.split("-")
    end_day = int(end_day_s)
    end_month = _end_month_from_label(month_label)
    end_year = year
    start_month = _start_month_from_label(month_label)
    start_day = int(start_day_s)
    if end_month < start_month or (end_month == start_month and end_day < start_day):
        end_year += 1
    return date(end_year, end_month, end_day)


def _end_month_from_label(month_label: str) -> int:
    label = month_label.strip().lower()
    if "/" in label:
        part = label.split("/")[-1].strip()
        return _month_name_to_int(part)
    return _month_name_to_int(label)


def _start_month_from_label(month_label: str) -> int:
    label = month_label.strip().lower()
    if "/" in label:
        part = label.split("/")[0].strip()
        return _month_name_to_int(part)
    return _month_name_to_int(label)


def _month_name_to_int(name: str) -> int:
    key = name.strip().lower()
    if key not in _MONTH_NAMES:
        raise ValueError(f"Unknown month label: {name!r}")
    return _MONTH_NAMES[key]


def _stamp_to_date(stamp: str) -> date:
    return date(int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8]))
