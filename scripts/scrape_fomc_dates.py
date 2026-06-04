#!/usr/bin/env python3
"""Scrape FOMC decision dates from federalreserve.gov and write JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from scripts._bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from data_platform.events.fed_fomc import (  # noqa: E402
    FomcScrapeResult,
    dates_to_iso_strings,
    merge_fomc_dates,
    parse_forward_calendar_html,
    parse_historical_index_years,
    parse_historical_year_html,
    validate_fomc_dates,
)

_FED_BASE = "https://www.federalreserve.gov/monetarypolicy"
_HISTORICAL_INDEX = f"{_FED_BASE}/fomc_historical_year.htm"
_FORWARD_CALENDAR = f"{_FED_BASE}/fomccalendars.htm"
_USER_AGENT = "Trading-Algo-CalendarScraper/1.0 (+https://github.com/)"
_TIMEOUT_SEC = 15
_HISTORICAL_END_DEFAULT = 2020
_FORWARD_START_DEFAULT = 2021


def _fetch(url: str) -> str:
    response = requests.get(
        url,
        timeout=_TIMEOUT_SEC,
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    return response.text


def scrape_fomc_dates(
    start_year: int,
    end_year: int,
    *,
    fetch: Callable[[str], str] = _fetch,
) -> FomcScrapeResult:
    warnings: list[str] = []
    historical_dates: list[date] = []
    hist_end = min(end_year, _HISTORICAL_END_DEFAULT)
    if start_year <= hist_end:
        index_html = fetch(_HISTORICAL_INDEX)
        linked_years = parse_historical_index_years(index_html)
        for year in range(start_year, hist_end + 1):
            if year not in linked_years:
                warnings.append(f"{year}: no fomchistorical{year}.htm link on index page")
                continue
            year_html = fetch(f"{_FED_BASE}/fomchistorical{year}.htm")
            historical_dates.extend(parse_historical_year_html(year_html))

    forward_dates: list[date] = []
    fwd_start = max(start_year, _FORWARD_START_DEFAULT)
    if end_year >= fwd_start:
        calendar_html = fetch(_FORWARD_CALENDAR)
        forward_dates.extend(
            parse_forward_calendar_html(calendar_html, fwd_start, end_year)
        )

    merged = merge_fomc_dates(historical_dates, forward_dates)
    warnings.extend(validate_fomc_dates(merged))
    return FomcScrapeResult(dates=merged, warnings=tuple(warnings))


def write_fomc_json(
    output_path: Path,
    dates: tuple[date, ...],
    *,
    strict: bool,
    warnings: tuple[str, ...],
) -> None:
    if strict and warnings:
        for msg in warnings:
            print(f"WARNING: {msg}", file=sys.stderr)
        raise SystemExit(1)

    for msg in warnings:
        print(f"WARNING: {msg}", file=sys.stderr)

    payload = {
        "schema_version": 1,
        "source": _HISTORICAL_INDEX,
        "forward_source": _FORWARD_CALENDAR,
        "scraped_at": datetime.now(timezone.utc).date().isoformat(),
        "event_type": "fomc_decision",
        "dates": dates_to_iso_strings(dates),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload['dates'])} FOMC decision dates to {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape FOMC decision dates from the Fed website.")
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2027)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/events/calendar/fomc_decision_dates.json"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if validation warnings are emitted.",
    )
    args = parser.parse_args(argv)
    repo_root = ensure_project_root_on_path()
    output = args.output if args.output.is_absolute() else repo_root / args.output

    result = scrape_fomc_dates(args.start_year, args.end_year)
    write_fomc_json(output, result.dates, strict=args.strict, warnings=result.warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
