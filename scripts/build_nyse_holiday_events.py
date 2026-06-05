#!/usr/bin/env python3
"""Build NYSE holiday D0 events JSON using ES daily sessions as trading calendar."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts._bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from data_platform.events.nyse_holidays import build_holiday_events  # noqa: E402
from lib.core.enums import Ticker, TimeFrame  # noqa: E402
from lib.core import helpers  # noqa: E402


def load_es_trading_sessions(start_year: int, end_year: int) -> list:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    df = helpers.load_data(Ticker.ES, TimeFrame.D, start=start, end=end)
    return sorted(set(df["datetime"].dt.date))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build NYSE holiday calendar JSON.")
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2027)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/events/calendar/nyse_holiday_events.json"),
    )
    args = parser.parse_args(argv)
    repo_root = ensure_project_root_on_path()
    output = args.output if args.output.is_absolute() else repo_root / args.output

    try:
        sessions = load_es_trading_sessions(args.start_year, args.end_year)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    events = build_holiday_events(args.start_year, args.end_year, sessions)
    payload = {
        "schema_version": 1,
        "source": "NYSE closure rules + ES daily sessions from data/ohlc_data",
        "built_at": datetime.now(timezone.utc).date().isoformat(),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "events": events,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(events)} holiday events to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
