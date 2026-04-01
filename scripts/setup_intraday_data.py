"""Unified setup for intraday back-adjusted data.

Stage 1: Extract intraday parquets from kibot_data.zip → data/intraday_original/
Stage 2: Back-adjust using Norgate-guided roll detection → data/intraday_adjusted/

Usage:
    python scripts/setup_intraday_data.py              # both stages, all tickers
    python scripts/setup_intraday_data.py --parallel    # parallel back-adjustment
    python scripts/setup_intraday_data.py --extract-only
    python scripts/setup_intraday_data.py --adjust-only --ticker ES
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from utils.core.enums import Ticker
from scripts.extract_kibot_data import main as extract_main
from data_cleaning.back_adjustment.orchestrator import (
    process_ticker,
    _process_ticker_safe,
    _DEFAULT_INPUT,
    _DEFAULT_OUTPUT,
    _DEFAULT_METADATA,
    _DEFAULT_NORGATE_RAW,
)

logger = logging.getLogger(__name__)


def _discover_tickers(input_dir: Path) -> list[Ticker]:
    """Return Ticker enums that have parquet files in input_dir."""
    return [
        t for t in Ticker
        if (input_dir / t.name).is_dir()
        and any((input_dir / t.name).glob("*.parquet"))
    ]


def run_extraction() -> None:
    """Stage 1: extract intraday data from kibot_data.zip."""
    print("=" * 60)
    print("Stage 1: Extracting intraday data from kibot_data.zip")
    print("=" * 60)
    extract_main()


def run_adjustment(
    ticker_name: str | None = None,
    parallel: bool = False,
) -> None:
    """Stage 2: back-adjust intraday data for one or all tickers."""
    print("=" * 60)
    print("Stage 2: Back-adjusting intraday data")
    print("=" * 60)

    input_dir = _DEFAULT_INPUT
    output_dir = _DEFAULT_OUTPUT
    metadata_dir = _DEFAULT_METADATA
    norgate_dir = _DEFAULT_NORGATE_RAW

    if ticker_name:
        ticker = Ticker[ticker_name.upper()]
        tickers = [ticker]
    else:
        tickers = _discover_tickers(input_dir)

    if not tickers:
        print(f"No tickers found in {input_dir}/. Run extraction first.")
        return

    print(f"Processing {len(tickers)} ticker(s): {', '.join(t.name for t in tickers)}")

    if parallel and len(tickers) > 1:
        with ProcessPoolExecutor() as pool:
            futures = {
                pool.submit(
                    _process_ticker_safe,
                    t.name, str(input_dir), str(output_dir),
                    str(metadata_dir), str(norgate_dir),
                ): t.name
                for t in tickers
            }
            for future in as_completed(futures):
                print(future.result())
    else:
        for t in tickers:
            try:
                process_ticker(t, input_dir, output_dir, metadata_dir, norgate_dir)
                print(f"{t.name}: OK")
            except Exception as e:
                logger.error("%s: FAILED - %s", t.name, e)
                print(f"{t.name}: FAILED - {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and back-adjust intraday futures data",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Only run Stage 1 (extract from zip)",
    )
    parser.add_argument(
        "--adjust-only", action="store_true",
        help="Only run Stage 2 (back-adjustment)",
    )
    parser.add_argument(
        "--ticker", type=str,
        help="Single ticker to process (e.g. ES). Default: all tickers.",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Use parallel processing for back-adjustment",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.extract_only and args.adjust_only:
        parser.error("Cannot use --extract-only and --adjust-only together")

    if not args.adjust_only:
        run_extraction()

    if not args.extract_only:
        run_adjustment(ticker_name=args.ticker, parallel=args.parallel)

    print("\nDone.")


if __name__ == "__main__":
    main()
