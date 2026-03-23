"""Bulk ingest repository-backed OHLC source data into the central cache."""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from typing import Any, Sequence

from utils.cache.bootstrap_source_candles import bootstrap_source_candles
from utils.core.enums import Ticker, TimeFrame


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def ingest_source_candles(
    tickers: Sequence[Ticker] | None = None,
    timeframes: Sequence[TimeFrame] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    reset_existing: bool = False,
) -> dict[str, Any]:
    """Deprecated compatibility alias for ``bootstrap_source_candles``."""
    warnings.warn(
        "ingest_source_candles(...) is deprecated; use bootstrap_source_candles(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bootstrap_source_candles(
        tickers=tickers,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        reset_existing=reset_existing,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bulk ingest source OHLC data from data/ohlc_data into the central cache.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Ticker symbols to ingest (default: all).",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="Timeframes to ingest, for example D W M (default: D W M).",
    )
    parser.add_argument(
        "--start",
        type=_parse_datetime,
        default=None,
        help="Inclusive ISO datetime/date lower bound.",
    )
    parser.add_argument(
        "--end",
        type=_parse_datetime,
        default=None,
        help="Inclusive ISO datetime/date upper bound.",
    )
    parser.add_argument(
        "--reset-existing",
        action="store_true",
        help="Clear the existing central candle cache before ingesting.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    tickers = [Ticker[name] for name in args.tickers] if args.tickers else None
    timeframes = [TimeFrame[name] for name in args.timeframes] if args.timeframes else None
    summary = ingest_source_candles(
        tickers=tickers,
        timeframes=timeframes,
        start_date=args.start,
        end_date=args.end,
        reset_existing=args.reset_existing,
    )
    print(
        "Source candle ingest complete: "
        f"{summary['success']} success, {summary['failed']} failed, {summary['total']} total"
    )
    failures = [detail for detail in summary["details"] if detail["status"] == "failed"]
    if failures:
        print("Failures:")
        for failure in failures:
            print(
                f"  - {failure['ticker']}/{failure['tf']}: {failure.get('message', 'unknown error')}"
            )


if __name__ == "__main__":
    main()
