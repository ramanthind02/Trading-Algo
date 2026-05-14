"""Back-adjust intraday data using Norgate-guided roll detection."""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from utils.core.enums import Ticker
from data_cleaning.back_adjustment.orchestrator import (
    process_ticker,
    _process_ticker_safe,
    _DEFAULT_INPUT,
    _DEFAULT_OUTPUT,
    _DEFAULT_METADATA,
    _DEFAULT_NORGATE_RAW,
)

logger = logging.getLogger(__name__)


def _discover_tickers() -> list[Ticker]:
    return [
        ticker
        for ticker in Ticker
        if (_DEFAULT_INPUT / ticker.name).is_dir()
        and any((_DEFAULT_INPUT / ticker.name).glob("*.parquet"))
    ]


def run_adjustment(ticker_name: str | None = None, parallel: bool = False) -> None:
    print("=" * 60)
    print("Back-adjusting intraday data using Norgate roll guidance")
    print("=" * 60)

    tickers = [Ticker[ticker_name.upper()]] if ticker_name else _discover_tickers()
    if not tickers:
        print(f"No intraday source data found in {_DEFAULT_INPUT}/")
        return

    print(f"Processing {len(tickers)} ticker(s): {', '.join(t.name for t in tickers)}")

    if parallel and len(tickers) > 1:
        with ProcessPoolExecutor() as pool:
            futures = {
                pool.submit(
                    _process_ticker_safe,
                    ticker.name,
                    str(_DEFAULT_INPUT),
                    str(_DEFAULT_OUTPUT),
                    str(_DEFAULT_METADATA),
                    str(_DEFAULT_NORGATE_RAW),
                ): ticker.name
                for ticker in tickers
            }
            for future in as_completed(futures):
                print(future.result())
        return

    for ticker in tickers:
        try:
            process_ticker(
                ticker,
                _DEFAULT_INPUT,
                _DEFAULT_OUTPUT,
                _DEFAULT_METADATA,
                _DEFAULT_NORGATE_RAW,
            )
            print(f"{ticker.name}: OK")
        except Exception as exc:
            logger.error("%s: FAILED - %s", ticker.name, exc)
            print(f"{ticker.name}: FAILED - {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-adjust intraday futures data")
    parser.add_argument("--ticker", type=str, help="Single ticker to process (e.g. ES)")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_adjustment(ticker_name=args.ticker, parallel=args.parallel)
    print("\nDone.")


if __name__ == "__main__":
    main()
