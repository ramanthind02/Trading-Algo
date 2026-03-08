# tests/integration/feature_validator/test_continuous_eda_pipeline.py
"""Integration test for the continuous binning EDA research pipeline.

Imports run_continuous_eda_pipeline directly from feature_research so any
regression in the researcher's script is immediately caught here.

Default config: RSI lookback=5, Ticker.ES, 2020-2023.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from feature_research.config import OBJECTIVE_METRIC_PRESETS
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
def test_continuous_eda_pipeline_smoke(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    lookback: int = 5,
) -> None:
    """Smoke test: single RSI param combo, ES daily, 2020-2023.

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
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": lookback},
            },
            target_col="log_return",
            strategy="long-short",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
        )
        results = run_eda_pipeline(config, Path(tmpdir))

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        label = f"lookback_{lookback}"
        assert label in results, f"Expected key '{label}' in results, got {list(results.keys())}"

        report_path = results[label]
        assert report_path.exists(), f"Report path does not exist: {report_path}"
        assert (report_path / "metadata.json").exists()
        assert (report_path / "common_stats.json").exists()
        assert (report_path / "feature_stats.json").exists()
        assert (report_path / "diagnostics.json").exists()
        plots_dir = report_path / "plots"
        assert plots_dir.is_dir()
        expected_plots = [
            "time_series_fig.png",
            "decile_plot_fig.png",
            "histogram_fig.png",
            "quintile_spread_fig.png",
        ]
        for plot_file in expected_plots:
            assert (plots_dir / plot_file).exists(), f"Missing plot: {plot_file}"


@pytest.mark.integration
def test_continuous_eda_pipeline_multi_combo(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    lookbacks: list[int] | None = None,
) -> None:
    """Multi-combo smoke test: RSI lookbacks [5, 10], ES daily, 2020-2023.

    Verifies:
    - Pipeline produces one result per param combo
    - Separate output directories created for each combo

    Exposed as parameters for researcher-driven exploration.
    """
    _skip_if_no_data()
    lookbacks = lookbacks or [5, 10]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ResearchConfig(
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": lookbacks},
            },
            target_col="log_return",
            strategy="long-short",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
        )
        results = run_eda_pipeline(config, Path(tmpdir))

        assert len(results) == len(lookbacks), (
            f"Expected {len(lookbacks)} results, got {len(results)}"
        )
        for lb in lookbacks:
            assert f"lookback_{lb}" in results


@pytest.mark.integration
def test_continuous_pipeline_can_run_permutation_suite_mode(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    lookbacks: list[int] | None = None,
) -> None:
    """Smoke test for continuous permutation-suite pipeline entrypoint."""
    _skip_if_no_data()
    lookbacks = lookbacks or [3, 5]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ResearchConfig(
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": lookbacks},
            },
            target_col="log_return",
            strategy="long-short",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
            in_sample_permutation=PermutationResearchConfig(
                objective_metric=OBJECTIVE_METRIC_PRESETS["t_stat"],
                top_k=2,
                min_folds_stable=1,
                fold_years=1,
                enabled=True,
                nreps_stage1=10,
            ),
        )

        try:
            suite = run_permutation_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

        assert suite.feature_type == "continuous"
        # For CONTINUOUS features, param grid is expanded by bin_count (default [10, 8, 5, 3])
        default_bin_count_len = 4  # BinningAnalysisConfig.bin_counts default
        expected_total = len(lookbacks) * default_bin_count_len
        assert suite.funnel_stats.total_params == expected_total
        assert len(suite.stage1_reports) == expected_total
