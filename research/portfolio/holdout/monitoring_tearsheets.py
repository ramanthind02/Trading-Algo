"""QuantStats tearsheets for strategy holdout monitoring (full timeline)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ensemble.portfolio_impl.portfolio_tester import (
    aggregate_intraday_returns_to_daily,
    calculate_baseline_returns,
)
from research.portfolio.config import PortfolioResearchConfig
from research.portfolio.holdout.monitoring_policy import StrategyReferenceSlices
from research.portfolio.pipelines.portfolio_test import (
    _benchmark_tearsheet_title,
    _benchmark_ticker_key,
    _load_candles,
)
from lib.core.enums import TimeFrame
from research.evaluation.walkforward.runner import _sanitize_tearsheet_name


def full_period_strategy_returns(
    train_returns: pd.Series,
    validation_returns: pd.Series,
    holdout_returns: pd.Series,
) -> pd.Series:
    """Stitch train, validation, and holdout daily returns into one series."""

    parts = [
        train_returns.dropna(),
        validation_returns.dropna(),
        holdout_returns.dropna(),
    ]
    combined = pd.concat(parts).sort_index()
    if combined.empty:
        return combined
    return combined[~combined.index.duplicated(keep="last")]


def _load_benchmark_returns_for_index(
    config: PortfolioResearchConfig,
    index: pd.DatetimeIndex,
) -> pd.Series | None:
    """Align configured benchmark (default ES buy-and-hold) to the strategy index."""

    benchmark_key = _benchmark_ticker_key(config)
    if benchmark_key is None or index.empty:
        return None

    daily_candles = _load_candles(
        config,
        TimeFrame.D,
        start=pd.Timestamp(index.min()),
        end=pd.Timestamp(index.max()),
    )
    if daily_candles.empty:
        return None

    baseline = calculate_baseline_returns(
        daily_candles,
        equal_weight=(config.baseline_mode == "equal_weight"),
        benchmark_ticker=benchmark_key,
    )
    baseline = aggregate_intraday_returns_to_daily(baseline)
    aligned = baseline.reindex(index, method="ffill").fillna(0.0)
    aligned.name = f"{benchmark_key}_buy_hold"
    return aligned


def write_strategy_monitoring_tearsheet(
    config: PortfolioResearchConfig,
    *,
    strategy_name: str,
    reference: StrategyReferenceSlices,
    output_dir: Path,
) -> Path | None:
    """Write one HTML tearsheet for the full train+validation+holdout return history."""

    if not config.holdout_robustness.export_strategy_monitoring_tearsheets:
        return None

    returns = full_period_strategy_returns(
        reference.train_returns,
        reference.validation_returns,
        reference.full_holdout_returns,
    )
    if returns.shape[0] < 2:
        return None

    from analysis.plotting.graphing.quantstats_reports import generate_tearsheet

    baseline = _load_benchmark_returns_for_index(config, returns.index)
    safe_name = _sanitize_tearsheet_name(strategy_name)
    output_path = output_dir / f"{safe_name}_full_period_tearsheet.html"
    generate_tearsheet(
        strategy_returns=returns,
        baseline_returns=baseline,
        feature_name=strategy_name,
        output_file=str(output_path),
        mode="html",
        timeframe=config.timeframe,
        target_annual_volatility=config.target_volatility,
        benchmark_title=_benchmark_tearsheet_title(config),
    )
    return output_path
