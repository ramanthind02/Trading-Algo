# tests/integration/feature_validator/test_eda_research_pipelines_integration.py
"""Integration tests for continuous and signed-signal EDA / permutation research pipelines."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

import pytest

from feature_research.config import (
    BinningAnalysisConfig,
    FeatureType,
    InSampleDefaultsCatalog,
    InSamplePhaseDefaultsConfig,
    PermutationResearchConfig,
    ResearchConfig,
    load_config,
    build_objective_metric_presets,
)
from feature_research.pipeline import (
    run_eda_pipeline,
    run_permutation_pipeline,
)
from utils.core.enums import Direction, Ticker, TimeFrame

from ._support import (
    skip_if_missing_data_prereq as _skip_if_missing_data_prereq,
    skip_if_no_data as _skip_if_no_data,
)

Mode = Literal["continuous", "signed_signal"]


def _make_research_config(
    *,
    mode: Mode,
    tickers: list[Ticker],
    start: datetime,
    end: datetime,
    bias_spec: dict,
    reports_dir: Path,
    objective_metric_name: str = "mean_return",
    permutation: PermutationResearchConfig | None = None,
) -> ResearchConfig:
    presets = build_objective_metric_presets(TimeFrame.D)
    if permutation is None:
        permutation = PermutationResearchConfig(
            objective_metric=presets[objective_metric_name],
            enabled=False,
        )
    defaults = load_config().in_sample_defaults
    if mode == "continuous":
        catalog = InSampleDefaultsCatalog(
            continuous=InSamplePhaseDefaultsConfig(
                bias_spec=bias_spec,
                target_col="log_return",
                strategy=Direction.LONG_SHORT,
                reports_dir=reports_dir,
            ),
            signed_signal=defaults.signed_signal,
        )
        feature_type = FeatureType.CONTINUOUS
    else:
        catalog = InSampleDefaultsCatalog(
            continuous=defaults.continuous,
            signed_signal=InSamplePhaseDefaultsConfig(
                bias_spec=bias_spec,
                target_col="log_return",
                strategy=Direction.LONG,
                reports_dir=reports_dir,
            ),
        )
        feature_type = FeatureType.SIGNED_SIGNAL
    return ResearchConfig(
        tickers=tickers,
        start=start,
        end=end,
        permutation=permutation,
        objective_metric_presets=presets,
        binning_params=BinningAnalysisConfig(),
        in_sample_defaults=catalog,
        feature_type=feature_type,
    )


def _rsi_continuous_bias(lookback: int) -> dict:
    return {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": lookback},
    }


def _rsi_signal_bias(rsi_period: int | list[int]) -> dict:
    return {
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
    }


@pytest.mark.parametrize("mode", ["continuous", "signed_signal"])
@pytest.mark.integration
def test_eda_pipeline_smoke(
    mode: Mode,
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
) -> None:
    _skip_if_no_data()
    tickers = tickers or [Ticker.ES]
    lookback = 5
    rsi_period = 2
    bias = _rsi_continuous_bias(lookback) if mode == "continuous" else _rsi_signal_bias(rsi_period)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_research_config(
            mode=mode,
            tickers=tickers,
            start=start,
            end=end,
            bias_spec=bias,
            reports_dir=Path(tmpdir),
        )
        try:
            results = run_eda_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        if mode == "continuous":
            label = f"lookback_{lookback}"
            assert label in results, f"Expected key '{label}' in results, got {list(results.keys())}"
            report_path = results[label]
        else:
            report_path = list(results.values())[0]

        assert report_path.exists(), f"Report path does not exist: {report_path}"
        assert (report_path / "metadata.json").exists()
        assert (report_path / "common_stats.json").exists()
        assert (report_path / "feature_stats.json").exists()
        assert (report_path / "diagnostics.json").exists()
        assert not (report_path / "plots").exists()


@pytest.mark.parametrize(
    "mode,lookbacks,rsi_periods",
    [
        ("continuous", [5, 10], [2, 3]),
        ("signed_signal", [5, 10], [2, 3]),
    ],
)
@pytest.mark.integration
def test_eda_pipeline_multi_combo(
    mode: Mode,
    lookbacks: list[int],
    rsi_periods: list[int],
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
) -> None:
    _skip_if_no_data()
    tickers = tickers or [Ticker.ES]
    bias = (
        _rsi_continuous_bias(lookbacks)
        if mode == "continuous"
        else _rsi_signal_bias(rsi_periods)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_research_config(
            mode=mode,
            tickers=tickers,
            start=start,
            end=end,
            bias_spec=bias,
            reports_dir=Path(tmpdir),
        )
        try:
            results = run_eda_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

        if mode == "continuous":
            assert len(results) == len(lookbacks)
            for lb in lookbacks:
                assert f"lookback_{lb}" in results
        else:
            assert len(results) == len(rsi_periods)


@pytest.mark.parametrize(
    "mode,rsi_periods",
    [
        ("signed_signal", [2, 3]),
    ],
)
@pytest.mark.integration
def test_eda_pipeline_permutation_suite_mode(
    mode: Mode,
    rsi_periods: list[int],
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
) -> None:
    _skip_if_no_data()
    tickers = tickers or [Ticker.ES]
    presets = build_objective_metric_presets(TimeFrame.D)
    bias = _rsi_signal_bias(rsi_periods)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_research_config(
            mode=mode,
            tickers=tickers,
            start=start,
            end=end,
            bias_spec=bias,
            reports_dir=Path(tmpdir),
            permutation=PermutationResearchConfig(
                objective_metric=presets["mean_return"],
                enabled=True,
                nreps=10,
            ),
        )
        try:
            suite, _ = run_permutation_pipeline(config, Path(tmpdir))
        except Exception as exc:  # pragma: no cover - integration environment guard
            _skip_if_missing_data_prereq(exc)
            raise

        assert suite.feature_type == "signed_signal"
        assert suite.funnel_stats.total_params == len(rsi_periods)
        assert len(suite.stage1_reports) == len(rsi_periods)
