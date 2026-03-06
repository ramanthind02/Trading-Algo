"""Integration tests for the rule-based EDA research pipeline.

Imports run_rule_based_eda_pipeline directly from feature_research so any
regression in the researcher's script is immediately caught here.

Default config: rsi_signal rsi_period=2, Ticker.ES, 2020-2023.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from feature_research.config import FeatureType
from feature_research.in_sample.config import (
    PermutationResearchConfig,
    ResearchConfig,
)
from feature_research.pipeline import (
    run_eda_pipeline,
    run_permutation_pipeline,
)
from utils.core.enums import Ticker, TimeFrame


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _skip_if_no_data() -> None:
    candle_dir = _project_root() / "data" / "ohlc_data"
    if not candle_dir.exists():
        pytest.skip(f"Missing persisted candle directory: {candle_dir}")


def _skip_if_missing_data_prereq(exc: Exception) -> None:
    message = str(exc)
    if isinstance(exc, FileNotFoundError):
        pytest.skip(f"Missing persisted data prerequisite: {message}")
    if isinstance(exc, ValueError) and (
        "Unable to load feature/target data" in message
        or "Feature extraction returned no data" in message
    ):
        pytest.skip(f"Missing data prerequisite for permutation suite: {message}")


@pytest.mark.integration
def test_rule_based_eda_pipeline_smoke(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    rsi_period: int = 2,
) -> None:
    """Smoke test: single rsi_signal param combo, ES daily, 2020-2023.

    Verifies:
    - Pipeline runs without exception
    - Exactly one result entry returned
    - Output directory contains expected JSON + plot files

    All key config inputs are exposed as parameters so researchers can call
    this directly with custom values for interactive validation.
    """
    _skip_if_no_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ResearchConfig(
            feature_type=FeatureType.RULE_BASED,
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": rsi_period,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
            target_col="log_return",
            strategy="long",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
        )
        results = run_eda_pipeline(config, Path(tmpdir))

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        report_path = list(results.values())[0]
        assert report_path.exists(), f"Report path does not exist: {report_path}"
        assert (report_path / "metadata.json").exists()
        assert (report_path / "common_stats.json").exists()
        assert (report_path / "feature_stats.json").exists()
        assert (report_path / "diagnostics.json").exists()
        plots_dir = report_path / "plots"
        assert plots_dir.is_dir()
        expected_plots = [
            "time_series_fig.png",
            "level_plot_fig.png",
        ]
        for plot_file in expected_plots:
            assert (plots_dir / plot_file).exists(), f"Missing plot: {plot_file}"


@pytest.mark.integration
def test_rule_based_eda_pipeline_multi_combo(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    rsi_periods: list[int] | None = None,
) -> None:
    """Multi-combo smoke test: rsi_period=[2, 3], ES daily, 2020-2023.

    Verifies:
    - Pipeline produces one result per param combo
    - Separate output directories created for each combo

    Exposed as parameters for researcher-driven exploration.
    """
    _skip_if_no_data()
    rsi_periods = rsi_periods or [2, 3]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ResearchConfig(
            feature_type=FeatureType.RULE_BASED,
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": rsi_periods,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
            target_col="log_return",
            strategy="long",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
        )
        results = run_eda_pipeline(config, Path(tmpdir))

        assert len(results) == len(rsi_periods), (
            f"Expected {len(rsi_periods)} results, got {len(results)}"
        )


@pytest.mark.integration
def test_rule_based_pipeline_can_run_permutation_suite_mode(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    rsi_periods: list[int] | None = None,
) -> None:
    """Smoke test for rule-based permutation-suite pipeline entrypoint."""
    _skip_if_no_data()
    rsi_periods = rsi_periods or [2, 3]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ResearchConfig(
            feature_type=FeatureType.RULE_BASED,
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": rsi_periods,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
            target_col="log_return",
            strategy="long",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
            permutation=PermutationResearchConfig(
                enabled=True,
                nreps_stage1=10,
            ),
        )

        try:
            suite = run_permutation_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

        assert suite.feature_type == "rule_based"
        assert suite.funnel_stats.total_params == len(rsi_periods)
        assert len(suite.stage1_reports) == len(rsi_periods)
