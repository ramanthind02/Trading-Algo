"""Explicit bootstrap helper for loading repository-backed candles into the runtime cache."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Sequence

from utils.cache.cache_manager import CacheManager
from utils.core.enums import Ticker, TimeFrame


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def bootstrap_source_candles(
    tickers: Sequence[Ticker] | None = None,
    timeframes: Sequence[TimeFrame] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    reset_existing: bool = False,
) -> dict[str, Any]:
    """Read repository-backed candles from ``data/ohlc_data`` into the runtime cache."""
    manager = CacheManager()
    return manager.bootstrap_source_candles(
        tickers=tickers,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        reset_existing=reset_existing,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap source OHLC data from data/ohlc_data into the runtime cache.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Ticker symbols to bootstrap (default: all).",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="Timeframes to bootstrap, for example D W M (default: D W M).",
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
        help="Clear the existing central candle cache before bootstrapping.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    tickers = [Ticker[name] for name in args.tickers] if args.tickers else None
    timeframes = [TimeFrame[name] for name in args.timeframes] if args.timeframes else None
    summary = bootstrap_source_candles(
        tickers=tickers,
        timeframes=timeframes,
        start_date=args.start,
        end_date=args.end,
        reset_existing=args.reset_existing,
    )
    print(
        "Source candle bootstrap complete: "
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
