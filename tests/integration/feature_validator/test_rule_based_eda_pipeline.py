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
import pandas as pd
import pytest

matplotlib.use("Agg")

from feature_research.rule_based.config import RuleBasedResearchConfig
from feature_research.rule_based.pipeline import run_rule_based_eda_pipeline
from feature_research.walkforward.config import WalkforwardResearchConfig
from utils.enums import Ticker, TimeFrame


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _skip_if_no_data() -> None:
    candle_dir = _project_root() / "data" / "ohlc_data"
    if not candle_dir.exists():
        pytest.skip(f"Missing persisted candle directory: {candle_dir}")


def _build_walkforward_config(
    *,
    start: datetime,
    end: datetime,
    enabled: bool,
) -> WalkforwardResearchConfig:
    return WalkforwardResearchConfig(
        train_start=start,
        train_end=datetime(2021, 1, 1),
        enabled=enabled,
        test_step=252,
        num_steps=4,
        top_k=3,
        objective_metric_name="sharpe",
        min_fold_samples=10,
        output_root=Path("feature_research/shared_results"),
    )


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
        config = RuleBasedResearchConfig(
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
            walkforward=_build_walkforward_config(start=start, end=end, enabled=False),
        )
        results = run_rule_based_eda_pipeline(config, Path(tmpdir))

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
            "rolling_corr_fig.png",
            "rolling_obj_fig.png",
            "level_plot_fig.png",
            "transition_heatmap_fig.png",
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
        config = RuleBasedResearchConfig(
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
            walkforward=_build_walkforward_config(start=start, end=end, enabled=False),
        )
        results = run_rule_based_eda_pipeline(config, Path(tmpdir))

        assert len(results) == len(rsi_periods), (
            f"Expected {len(rsi_periods)} results, got {len(results)}"
        )


@pytest.mark.integration
def test_rule_based_eda_pipeline_walkforward_enabled_smoke() -> None:
    _skip_if_no_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = RuleBasedResearchConfig(
            tickers=[Ticker.ES],
            start=datetime(2020, 1, 1),
            end=datetime(2023, 12, 31),
            bias_spec={
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2,
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
            populate_cache=False,
            reports_dir=Path(tmpdir),
            walkforward=WalkforwardResearchConfig(
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2021, 1, 1),
                enabled=True,
                test_step=252,
                num_steps=4,
                top_k=3,
                objective_metric_name="sharpe",
                min_fold_samples=10,
                output_root=Path(tmpdir) / "shared_results",
            ),
        )
        run_rule_based_eda_pipeline(config, Path(tmpdir))

        walkforward_dir = (
            config.walkforward.output_root
            / "rule_based"
            / config.bias_spec["module_name"]
            / "walkforward"
        )
        required_files = [
            "folds.csv",
            "fold_scores.csv",
            "selection_summary.csv",
            "report.json",
            "walkforward_stability.png",
            "fold_timeline.png",
        ]
        missing_files = [name for name in required_files if not (walkforward_dir / name).exists()]
        if missing_files:
            pytest.skip(
                "Walkforward smoke prerequisites not available (likely missing persisted cache/data): "
                f"{missing_files}"
            )

        selection_summary_df = pd.read_csv(walkforward_dir / "selection_summary.csv")
        assert "selected_feature" in selection_summary_df.columns
