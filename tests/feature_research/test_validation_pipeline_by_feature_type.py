"""Validation pipeline smoke: signed-signal configs with mocked feature loads."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, cast

import pandas as pd
import pytest

from feature_research.config import (
    FeatureType,
    InSampleDefaultsCatalog,
    InSamplePhaseDefaultsConfig,
    ResearchWindowConfig,
)
from feature_research.config import ResearchConfig, load_config
from feature_research.pipeline import run_validation_pipeline
from utils.core.enums import Ticker, TimeFrame
from utils.evaluation.walkforward.runner import WalkforwardRunReport


def _build_signed_signal_config_rsi_lookback_grid(tmp_path: Path) -> ResearchConfig:
    """Signed-signal validation smoke using a 2-combo param grid (mocked features)."""
    base = load_config()
    bias_spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": [2, 3]},
    }
    branch_defaults = InSamplePhaseDefaultsConfig(
        bias_spec=bias_spec,
        target_col="log_return",
        strategy="long",
        reports_dir=tmp_path / "reports",
        binning_params_overrides={},
    )
    in_sample_defaults = InSampleDefaultsCatalog(
        continuous=branch_defaults,
        signed_signal=branch_defaults,
    )
    return base.__class__(
        tickers=[Ticker.ES],
        start=datetime(2020, 1, 1),
        end=datetime(2020, 4, 29),
        permutation=base.permutation,
        objective_metric_presets=base.objective_metric_presets,
        binning_params=base.binning_params,
        timeframe=TimeFrame.D,
        feature_type=FeatureType.SIGNED_SIGNAL,
        in_sample_defaults=in_sample_defaults,
        param_sensitivity=base.param_sensitivity,
        research_window=ResearchWindowConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 2, 10),
            val_start=datetime(2020, 2, 11),
            val_end=datetime(2020, 3, 10),
        ),
        n_jobs=base.n_jobs,
        output_root=tmp_path / "shared_results",
        generate_ticker_tearsheets=base.generate_ticker_tearsheets,
        vault_save=base.vault_save,
        sector_allocation_config_path=base.sector_allocation_config_path,
        portfolio_vault_correlation=base.portfolio_vault_correlation,
        evaluation_defaults=None,
    )


def _build_signed_signal_config(tmp_path: Path) -> ResearchConfig:
    base = load_config()
    bias_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": {
            "rsi_period": [2, 3],
            "oversold": 25.0,
            "overbought": 65.0,
            "strategy_mode": "long",
            "exit_policy": "threshold_or_bars",
            "exit_bars": 5,
        },
    }
    signed_signal_defaults = InSamplePhaseDefaultsConfig(
        bias_spec=bias_spec,
        target_col="log_return",
        strategy="long",
        reports_dir=tmp_path / "reports",
        binning_params_overrides={},
    )
    in_sample_defaults = InSampleDefaultsCatalog(
        continuous=base.in_sample_defaults.continuous,
        signed_signal=signed_signal_defaults,
    )
    return base.__class__(
        tickers=[Ticker.ES],
        start=datetime(2020, 1, 1),
        end=datetime(2020, 4, 29),
        permutation=base.permutation,
        objective_metric_presets=base.objective_metric_presets,
        binning_params=base.binning_params,
        timeframe=TimeFrame.D,
        feature_type=FeatureType.SIGNED_SIGNAL,
        in_sample_defaults=in_sample_defaults,
        param_sensitivity=base.param_sensitivity,
        research_window=ResearchWindowConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 2, 10),
            val_start=datetime(2020, 2, 11),
            val_end=datetime(2020, 3, 10),
        ),
        n_jobs=base.n_jobs,
        output_root=tmp_path / "shared_results",
        generate_ticker_tearsheets=base.generate_ticker_tearsheets,
        vault_save=base.vault_save,
        sector_allocation_config_path=base.sector_allocation_config_path,
        portfolio_vault_correlation=base.portfolio_vault_correlation,
        evaluation_defaults=None,
    )


def _series_continuous(
    params: dict[str, object],
) -> tuple[pd.Series, pd.Series, str, pd.Series]:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    lookback = int(cast(int, params["lookback"]))
    feature = pd.Series([float(lookback)] * len(index), index=index, name=f"feature_{lookback}")
    target = pd.Series([0.01] * len(index), index=index, name="target")
    ticker_s = pd.Series(["ES"] * len(index), index=index, name="ticker")
    return feature, target, str(feature.name), ticker_s


def _series_signed(params: dict[str, object]) -> tuple[pd.Series, pd.Series, str, pd.Series]:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    period = int(cast(int, params["rsi_period"]))
    feature = pd.Series([float(period)] * len(index), index=index, name=f"feature_{period}")
    target = pd.Series([0.01] * len(index), index=index, name="target")
    ticker_s = pd.Series(["ES"] * len(index), index=index, name="ticker")
    return feature, target, str(feature.name), ticker_s


def _mock_candles() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    return pd.DataFrame(
        {
            "datetime": index,
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [100.5] * len(index),
            "volume": [1000.0] * len(index),
            "ticker": ["ES"] * len(index),
            "timeframe": [TimeFrame.D] * len(index),
        },
        index=index,
    )


@pytest.mark.parametrize(
    "feature_type,build_config,expand_bias,series_fn",
    [
        (
            FeatureType.SIGNED_SIGNAL,
            _build_signed_signal_config_rsi_lookback_grid,
            lambda: [
                {"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 2}},
                {"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 3}},
            ],
            _series_continuous,
        ),
        (
            FeatureType.SIGNED_SIGNAL,
            _build_signed_signal_config,
            lambda: [
                {
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
                {
                    "module_name": "rsi_signal",
                    "timeframes": [TimeFrame.D],
                    "params": {
                        "rsi_period": 3,
                        "oversold": 25.0,
                        "overbought": 65.0,
                        "strategy_mode": "long",
                        "exit_policy": "threshold_or_bars",
                        "exit_bars": 5,
                    },
                },
            ],
            _series_signed,
        ),
    ],
)
def test_run_validation_pipeline_returns_report_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    feature_type: FeatureType,
    build_config: Callable[[Path], ResearchConfig],
    expand_bias: Callable[[], list[dict[str, object]]],
    series_fn: Callable[
        [dict[str, object]], tuple[pd.Series, pd.Series, str, pd.Series]
    ],
) -> None:
    _ = feature_type
    config = build_config(tmp_path)
    specs = expand_bias()

    monkeypatch.setattr(
        "feature_research.pipelines._shared.populate_cache_if_needed",
        lambda _config, **_kwargs: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.expand_bias_specs",
        lambda _bias_spec: specs,
    )
    monkeypatch.setattr(
        "utils.evaluation.walkforward.research_data.load_features_for_combo",
        lambda single_spec, _config, **_kwargs: series_fn(single_spec["params"]),
    )
    monkeypatch.setattr(
        "feature_research.pipelines._shared.load_portfolio_candles",
        lambda _config: _mock_candles(),
    )

    report = run_validation_pipeline(config, tmp_path / "validation_out")

    assert isinstance(report, WalkforwardRunReport)
    assert list((tmp_path / "shared_results").rglob("report.json"))
