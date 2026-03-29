# tests/integration/feature_validator/test_continuous_eda_pipeline.py
"""Integration test for the continuous binning EDA research pipeline.

Imports run_continuous_eda_pipeline directly from feature_research so any
regression in the researcher's script is immediately caught here.

Default config: RSI lookback=5, Ticker.ES, 2020-2023.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from feature_research.config import (
    BinningAnalysisConfig,
    FeatureType,
    InSampleDefaultsCatalog,
    InSamplePhaseDefaultsConfig,
    PermutationResearchConfig,
    ResearchConfig,
    build_objective_metric_presets,
)
from feature_research.pipeline import (
    run_eda_pipeline,
    run_permutation_pipeline,
)
from utils.core.enums import Direction, Ticker, TimeFrame


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
    if "Artifact is missing" in message:
        pytest.skip(f"Missing cached artifact prerequisite: {message}")
    if isinstance(exc, ValueError) and (
        "Unable to load feature/target data" in message
        or "Feature extraction returned no data" in message
    ):
        pytest.skip(f"Missing data prerequisite for permutation suite: {message}")


def _make_test_config(
    *,
    tickers: list[Ticker],
    start: datetime,
    end: datetime,
    bias_spec: dict,
    reports_dir: Path,
    objective_metric_name: str = "t_stat",
    permutation: PermutationResearchConfig | None = None,
) -> ResearchConfig:
    """Build a ResearchConfig suitable for continuous EDA integration tests."""
    presets = build_objective_metric_presets(TimeFrame.D)
    if permutation is None:
        permutation = PermutationResearchConfig(
            objective_metric=presets[objective_metric_name],
            enabled=False,
        )
    catalog = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=bias_spec,
            target_col="log_return",
            strategy=Direction.LONG_SHORT,
            reports_dir=reports_dir,
        ),
        signed_signal=InSampleDefaultsCatalog.default_for().signed_signal,
    )
    return ResearchConfig(
        tickers=tickers,
        start=start,
        end=end,
        use_cache=True,
        populate_cache=True,
        permutation=permutation,
        objective_metric_presets=presets,
        binning_params=BinningAnalysisConfig(),
        feature_type=FeatureType.CONTINUOUS,
        in_sample_defaults=catalog,
    )


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
        config = _make_test_config(
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": lookback},
            },
            reports_dir=Path(tmpdir),
        )
        try:
            results = run_eda_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

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
        config = _make_test_config(
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": lookbacks},
            },
            reports_dir=Path(tmpdir),
        )
        try:
            results = run_eda_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

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

    presets = build_objective_metric_presets(TimeFrame.D)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": lookbacks},
            },
            reports_dir=Path(tmpdir),
            permutation=PermutationResearchConfig(
                objective_metric=presets["t_stat"],
                enabled=True,
                nreps_stage1=10,
            ),
        )

        try:
            suite = run_permutation_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

        assert suite.feature_type == "signed_signal"
        # Param grid is (bin_count × lookback), so total_params can exceed len(lookbacks).
        assert suite.funnel_stats.total_params == len(suite.stage1_reports)
        assert len(suite.stage1_reports) >= len(lookbacks)
