"""Orchestrate the full back-adjustment pipeline for one or all tickers.

Detects rolls once per ticker (from the highest-resolution file), then
applies the same adjustments to all intraday timeframes.

Directory layout:
    Input:    data/intraday_original/{TICKER}/{TF}_{TICKER}.parquet
    Output:   data/intraday_adjusted/{TICKER}/{TF}_{TICKER}.parquet
    Metadata: data/adjustment_metadata/{TICKER}.json

Usage:
    python -m data_platform.providers.norgate.backadjust.orchestrator --ticker ES
    python -m data_platform.providers.norgate.backadjust.orchestrator --all
    python -m data_platform.providers.norgate.backadjust.orchestrator --all --parallel
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from lib.core.enums import Ticker
from .roll_rules import get_roll_rule
from .roll_detector import detect_roll_dates
from .gap_calculator import calculate_adjustments
from .back_adjuster import (
    apply_back_adjustment,
    validate_adjusted_data,
)

logger = logging.getLogger(__name__)

_DEFAULT_INPUT = Path("data/intraday_original")
_DEFAULT_OUTPUT = Path("data/intraday_adjusted")
_DEFAULT_METADATA = Path("data/adjustment_metadata")
_DEFAULT_NORGATE_RAW = Path("data/norgate/continuous_futures/unadjusted")

# Resolution order: prefer highest-resolution file for roll detection
_TF_RESOLUTION_ORDER = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "M10", "M15", "M30", "H1", "H2", "H4",
]

_INTRADAY_PATTERN = re.compile(r"^(M\d+|H\d+)_[A-Z]+\.parquet$")


@dataclass(frozen=True)
class AdjustmentMetadata:
    """Audit trail for one ticker's back-adjustment run."""

    ticker: Ticker
    processing_date: str
    num_rolls_detected: int
    total_adjustment_range: float
    source_file: str
    output_dir: str
    timeframes_adjusted: List[str]
    roll_dates: List[str]
    gap_points: List[float]
    cumulative_adjustments: List[float]


def _serialize_metadata(meta: AdjustmentMetadata) -> dict:
    d = asdict(meta)
    d["ticker"] = meta.ticker.name
    return d


def _load_norgate_unadjusted(ticker: Ticker, norgate_dir: Path) -> pd.DataFrame | None:
    """Load Norgate unadjusted data for Norgate-guided roll detection."""
    norgate_path = norgate_dir / f"{ticker.name}.parquet"
    if norgate_path.exists():
        return pd.read_parquet(norgate_path)
    return None


def _find_ticker_files(input_dir: Path, ticker: str) -> list[Path]:
    """Find all intraday parquet files for a ticker."""
    ticker_dir = input_dir / ticker
    if not ticker_dir.is_dir():
        return []
    return sorted(
        p for p in ticker_dir.iterdir()
        if p.suffix == ".parquet" and _INTRADAY_PATTERN.match(p.name)
    )


def _pick_detection_file(files: list[Path]) -> Path | None:
    """Pick the highest-resolution file for roll detection."""
    by_tf: dict[str, Path] = {}
    for f in files:
        tf = f.stem.split("_")[0]
        by_tf[tf] = f
    for tf in _TF_RESOLUTION_ORDER:
        if tf in by_tf:
            return by_tf[tf]
    return files[0] if files else None


def process_ticker(
    ticker: Ticker,
    input_dir: Path = _DEFAULT_INPUT,
    output_dir: Path = _DEFAULT_OUTPUT,
    metadata_dir: Path = _DEFAULT_METADATA,
    norgate_dir: Path = _DEFAULT_NORGATE_RAW,
) -> AdjustmentMetadata:
    """Run the full back-adjustment pipeline for all timeframes of a ticker.

    1. Find all intraday files for the ticker.
    2. Detect rolls from the highest-resolution file (M1 preferred).
    3. Calculate cumulative adjustment factors.
    4. Apply the same adjustments to every timeframe file.
    """
    ticker_name = ticker.name
    files = _find_ticker_files(input_dir, ticker_name)
    if not files:
        raise FileNotFoundError(
            f"No intraday files found in {input_dir / ticker_name}/"
        )

    # 1. Pick detection file and detect rolls
    detect_file = _pick_detection_file(files)
    logger.info("Processing %s: %d timeframes, detecting from %s",
                ticker_name, len(files), detect_file.name)

    detect_df = pd.read_parquet(detect_file)
    rule = get_roll_rule(ticker)

    norgate_raw = _load_norgate_unadjusted(ticker, norgate_dir)
    if norgate_raw is not None:
        logger.info("%s: using Norgate-guided roll detection", ticker_name)

    roll_events = detect_roll_dates(detect_df, rule, norgate_unadjusted=norgate_raw)
    adjustments = calculate_adjustments(roll_events)
    logger.info("%s: %d rolls detected", ticker_name, len(roll_events))

    # 2. Apply adjustments to every timeframe
    ticker_out = output_dir / ticker_name
    ticker_out.mkdir(parents=True, exist_ok=True)
    adjusted_tfs: list[str] = []

    for filepath in files:
        tf = filepath.stem.split("_")[0]
        df = pd.read_parquet(filepath)
        adjusted = apply_back_adjustment(df, adjustments)

        if adjustments:
            validate_adjusted_data(df, adjusted, adjustments)

        out_path = ticker_out / filepath.name
        adjusted.to_parquet(out_path, index=False)
        adjusted_tfs.append(tf)
        logger.info("%s/%s: adjusted (%d rows)", ticker_name, tf, len(adjusted))

    # 3. Compute metadata
    adj_range = 0.0
    if adjustments:
        all_cum = [a.cumulative_adjustment for a in adjustments]
        total_gaps = sum(a.gap_points for a in adjustments)
        adj_range = max(abs(total_gaps), max(abs(c) for c in all_cum))

    meta = AdjustmentMetadata(
        ticker=ticker,
        processing_date=datetime.now(timezone.utc).isoformat(),
        num_rolls_detected=len(roll_events),
        total_adjustment_range=adj_range,
        source_file=str(detect_file),
        output_dir=str(ticker_out),
        timeframes_adjusted=adjusted_tfs,
        roll_dates=[e.roll_date.isoformat() for e in roll_events],
        gap_points=[e.gap_points for e in roll_events],
        cumulative_adjustments=[a.cumulative_adjustment for a in adjustments],
    )

    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metadata_dir / f"{ticker_name}.json"
    meta_path.write_text(json.dumps(_serialize_metadata(meta), indent=2))

    logger.info(
        "%s: done. rolls=%d, timeframes=%d, adj_range=%.2f",
        ticker_name, meta.num_rolls_detected, len(adjusted_tfs),
        meta.total_adjustment_range,
    )
    return meta


def _process_ticker_safe(
    ticker_name: str,
    input_dir: str,
    output_dir: str,
    metadata_dir: str,
    norgate_dir: str,
) -> str:
    """Wrapper for parallel execution (picklable arguments)."""
    try:
        ticker = Ticker[ticker_name]
        process_ticker(
            ticker, Path(input_dir), Path(output_dir),
            Path(metadata_dir), Path(norgate_dir),
        )
        return f"{ticker_name}: OK"
    except Exception as e:
        logger.error("%s: FAILED - %s", ticker_name, e)
        return f"{ticker_name}: FAILED - {e}"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Back-adjust futures OHLC data across all intraday timeframes",
    )
    parser.add_argument("--ticker", type=str, help="Single ticker to process (e.g. ES)")
    parser.add_argument("--all", action="store_true", help="Process all tickers")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--input-dir", type=str, default=str(_DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--metadata-dir", type=str, default=str(_DEFAULT_METADATA))
    parser.add_argument("--norgate-dir", type=str, default=str(_DEFAULT_NORGATE_RAW))
    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    norgate_dir = Path(args.norgate_dir)

    if args.all:
        # Find tickers that have subdirectories with parquet files
        tickers = [
            t for t in Ticker
            if (input_dir / t.name).is_dir()
            and any((input_dir / t.name).glob("*.parquet"))
        ]
        logger.info("Processing %d tickers", len(tickers))

        if args.parallel:
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
                except Exception as e:
                    logger.error("%s: FAILED - %s", t.name, e)

    elif args.ticker:
        ticker = Ticker[args.ticker.upper()]
        process_ticker(ticker, input_dir, output_dir, metadata_dir, norgate_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
