"""Integration tests for the signed-signal EDA research pipeline.

Imports run_signed_signal_eda_pipeline directly from feature_research so any
regression in the researcher's script is immediately caught here.

Default config: rsi_signal rsi_period=2, Ticker.ES, 2020-2023.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

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
from ._support import (
    skip_if_missing_data_prereq as _skip_if_missing_data_prereq,
    skip_if_no_data as _skip_if_no_data,
)
from utils.core.enums import Direction, Ticker, TimeFrame


def _make_signed_signal_config(
    *,
    tickers: list[Ticker],
    start: datetime,
    end: datetime,
    bias_spec: dict,
    reports_dir: Path,
    objective_metric_name: str = "t_stat",
    permutation: PermutationResearchConfig | None = None,
) -> ResearchConfig:
    """Build a ResearchConfig suitable for signed-signal EDA integration tests."""
    presets = build_objective_metric_presets(TimeFrame.D)
    if permutation is None:
        permutation = PermutationResearchConfig(
            objective_metric=presets[objective_metric_name],
            enabled=False,
        )
    catalog = InSampleDefaultsCatalog(
        continuous=InSampleDefaultsCatalog.default_for().continuous,
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=bias_spec,
            target_col="log_return",
            strategy=Direction.LONG,
            reports_dir=reports_dir,
        ),
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
        feature_type=FeatureType.SIGNED_SIGNAL,
        in_sample_defaults=catalog,
    )


@pytest.mark.integration
def test_signed_signal_eda_pipeline_smoke(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    rsi_period: int = 2,
) -> None:
    """Smoke test: single rsi_signal param combo, ES daily, 2020-2023.

    Verifies:
    - Pipeline runs without exception
    - Exactly one result entry returned
    - Output directory contains expected JSON summaries

    All key config inputs are exposed as parameters so researchers can call
    this directly with custom values for interactive validation.
    """
    _skip_if_no_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_signed_signal_config(
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
            reports_dir=Path(tmpdir),
        )
        try:
            results = run_eda_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        report_path = list(results.values())[0]
        assert report_path.exists(), f"Report path does not exist: {report_path}"
        assert (report_path / "metadata.json").exists()
        assert (report_path / "common_stats.json").exists()
        assert (report_path / "feature_stats.json").exists()
        assert (report_path / "diagnostics.json").exists()
        assert not (report_path / "plots").exists()


@pytest.mark.integration
def test_signed_signal_eda_pipeline_multi_combo(
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
        config = _make_signed_signal_config(
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
            reports_dir=Path(tmpdir),
        )
        try:
            results = run_eda_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

        assert len(results) == len(rsi_periods), (
            f"Expected {len(rsi_periods)} results, got {len(results)}"
        )


@pytest.mark.integration
def test_signed_signal_pipeline_can_run_permutation_suite_mode(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    rsi_periods: list[int] | None = None,
) -> None:
    """Smoke test for rule-based permutation-suite pipeline entrypoint."""
    _skip_if_no_data()
    rsi_periods = rsi_periods or [2, 3]

    presets = build_objective_metric_presets(TimeFrame.D)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_signed_signal_config(
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
        assert suite.funnel_stats.total_params == len(rsi_periods)
        assert len(suite.stage1_reports) == len(rsi_periods)
