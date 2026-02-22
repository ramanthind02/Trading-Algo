"""Orchestrate the full back-adjustment pipeline for one or all tickers.

Usage:
    python -m data_cleaning.back_adjustment.orchestrator --ticker ES
    python -m data_cleaning.back_adjustment.orchestrator --all
    python -m data_cleaning.back_adjustment.orchestrator --all --parallel
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from utils.enums import Ticker
from data_cleaning.back_adjustment.roll_rules import get_roll_rule
from data_cleaning.back_adjustment.roll_detector import detect_roll_dates
from data_cleaning.back_adjustment.gap_calculator import calculate_adjustments
from data_cleaning.back_adjustment.back_adjuster import (
    apply_back_adjustment,
    validate_adjusted_data,
)

logger = logging.getLogger(__name__)

_DEFAULT_INPUT = Path("data/intraday_1min_original")
_DEFAULT_OUTPUT = Path("data/intraday_1min_adjusted")
_DEFAULT_METADATA = Path("data/adjustment_metadata")


@dataclass(frozen=True)
class AdjustmentMetadata:
    ticker: Ticker
    processing_date: str
    num_rolls_detected: int
    total_adjustment_range: float
    source_file: str
    output_file: str
    roll_dates: List[str]
    gap_points: List[float]
    cumulative_adjustments: List[float]


def _serialize_metadata(meta: AdjustmentMetadata) -> dict:
    d = asdict(meta)
    d["ticker"] = meta.ticker.name
    return d


def process_ticker(
    ticker: Ticker,
    input_dir: Path = _DEFAULT_INPUT,
    output_dir: Path = _DEFAULT_OUTPUT,
    metadata_dir: Path = _DEFAULT_METADATA,
) -> AdjustmentMetadata:
    source_path = input_dir / f"{ticker.name}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    logger.info("Processing %s from %s", ticker.name, source_path)

    df = pd.read_parquet(source_path)
    rule = get_roll_rule(ticker)
    roll_events = detect_roll_dates(df, rule)
    logger.info("%s: %d rolls detected", ticker.name, len(roll_events))

    adjustments = calculate_adjustments(roll_events)
    adjusted = apply_back_adjustment(df, adjustments)

    if adjustments:
        validate_adjusted_data(df, adjusted, adjustments)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker.name}.parquet"
    adjusted.to_parquet(output_path, index=False)

    adj_range = 0.0
    if adjustments:
        all_cum = [a.cumulative_adjustment for a in adjustments]
        total_gaps = sum(a.gap_points for a in adjustments)
        adj_range = max(abs(total_gaps), max(abs(c) for c in all_cum))

    meta = AdjustmentMetadata(
        ticker=ticker,
        processing_date=datetime.utcnow().isoformat(),
        num_rolls_detected=len(roll_events),
        total_adjustment_range=adj_range,
        source_file=str(source_path),
        output_file=str(output_path),
        roll_dates=[e.roll_date.isoformat() for e in roll_events],
        gap_points=[e.gap_points for e in roll_events],
        cumulative_adjustments=[a.cumulative_adjustment for a in adjustments],
    )

    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metadata_dir / f"{ticker.name}.json"
    meta_path.write_text(json.dumps(_serialize_metadata(meta), indent=2))

    logger.info(
        "%s: done. rolls=%d, adj_range=%.2f",
        ticker.name, meta.num_rolls_detected, meta.total_adjustment_range,
    )
    return meta


def _process_ticker_safe(
    ticker_name: str,
    input_dir: str,
    output_dir: str,
    metadata_dir: str,
) -> str:
    try:
        ticker = Ticker[ticker_name]
        process_ticker(ticker, Path(input_dir), Path(output_dir), Path(metadata_dir))
        return f"{ticker_name}: OK"
    except Exception as e:
        logger.error("%s: FAILED - %s", ticker_name, e)
        return f"{ticker_name}: FAILED - {e}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Back-adjust futures OHLC data")
    parser.add_argument("--ticker", type=str, help="Single ticker to process")
    parser.add_argument("--all", action="store_true", help="Process all tickers")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--input-dir", type=str, default=str(_DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--metadata-dir", type=str, default=str(_DEFAULT_METADATA))
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)

    if args.all:
        tickers = [t for t in Ticker if (input_dir / f"{t.name}.parquet").exists()]
        logger.info("Processing %d tickers", len(tickers))
        if args.parallel:
            with ProcessPoolExecutor() as pool:
                futures = {
                    pool.submit(
                        _process_ticker_safe,
                        t.name, str(input_dir), str(output_dir), str(metadata_dir),
                    ): t.name
                    for t in tickers
                }
                for future in as_completed(futures):
                    print(future.result())
        else:
            for t in tickers:
                try:
                    process_ticker(t, input_dir, output_dir, metadata_dir)
                except Exception as e:
                    logger.error("%s: FAILED - %s", t.name, e)
    elif args.ticker:
        ticker = Ticker[args.ticker.upper()]
        process_ticker(ticker, input_dir, output_dir, metadata_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
