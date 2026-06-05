"""Unit tests for Fed FOMC HTML parsers (fixture-backed)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from data_platform.events.fed_fomc import (
    parse_forward_calendar_html,
    parse_historical_year_html,
    validate_fomc_dates,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "calendar" / "fed"


@pytest.fixture(scope="module")
def historical_2005_html() -> str:
    return (_FIXTURE_DIR / "fomchistorical2005.htm").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def forward_calendar_html() -> str:
    return (_FIXTURE_DIR / "fomccalendars.htm").read_text(encoding="utf-8")


def test_parse_historical_year_2005(historical_2005_html: str) -> None:
    dates = parse_historical_year_html(historical_2005_html)
    assert dates == (
        date(2005, 2, 2),
        date(2005, 3, 22),
        date(2005, 5, 3),
        date(2005, 6, 30),
        date(2005, 8, 9),
        date(2005, 9, 20),
        date(2005, 11, 1),
        date(2005, 12, 13),
    )


def test_parse_forward_calendar_2024(forward_calendar_html: str) -> None:
    dates = parse_forward_calendar_html(forward_calendar_html, 2024, 2024)
    assert [d.isoformat() for d in dates] == [
        "2024-01-31",
        "2024-03-20",
        "2024-05-01",
        "2024-06-12",
        "2024-07-31",
        "2024-09-18",
        "2024-11-07",
        "2024-12-18",
    ]


def test_validate_fomc_dates_accepts_2024_subset(forward_calendar_html: str) -> None:
    dates = parse_forward_calendar_html(forward_calendar_html, 2024, 2024)
    warnings = validate_fomc_dates(dates)
    assert "2024 sanity check" not in " ".join(warnings)
