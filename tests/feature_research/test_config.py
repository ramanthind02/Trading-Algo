from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from feature_research.config import (
    BaseResearchConfig,
    FeatureType,
    InSampleDefaultsCatalog,
    OBJECTIVE_METRIC_PRESETS,
    OOSWindowConfig,
    PermutationResearchConfig,
    build_objective_metric_presets,
    load_config as load_base_config,
)
from feature_research.in_sample.config import ResearchConfig, load_config
from feature_research.pipeline import run_oos_pipeline
from utils.core.enums import Ticker, TimeFrame


def test_load_config_returns_research_config():
    config = load_config()
    assert isinstance(config, ResearchConfig)


def test_load_config_defaults():
    config = load_config()
    assert Ticker.ES in config.tickers
    assert Ticker.NQ in config.tickers
    assert config.start == datetime(2000, 1, 1)
    assert config.end == datetime(2017, 12, 31)
    assert config.bias_spec["module_name"] == "cyclical_rsi"
    assert "short_period" in config.bias_spec["params"]
    assert isinstance(config.bias_spec["params"]["short_period"], list)
    assert config.target_col == "log_return_atr"
    assert config.use_cache is True
    assert config.populate_cache is True


def test_timeframe_bars_per_year_values() -> None:
    assert TimeFrame.H1.bars_per_year == 5200
    assert TimeFrame.H4.bars_per_year == 1300
    assert TimeFrame.D.bars_per_year == 252


def test_build_objective_metric_presets_uses_timeframe_bars_per_year() -> None:
    weekly_presets = build_objective_metric_presets(TimeFrame.W)
    assert weekly_presets["sharpe_annualized"].kwargs == {"annualization_factor": 52.0}
    assert weekly_presets["sortino_annualized"].kwargs == {"annualization_factor": 52.0}
    assert weekly_presets["calmar_annualized"].kwargs == {"annualization_factor": 52.0}


def test_in_sample_defaults_catalog_default_for_timeframe() -> None:
    weekly_defaults = InSampleDefaultsCatalog.default_for(TimeFrame.W)
    assert weekly_defaults.continuous.bias_spec["timeframes"] == [TimeFrame.W]
    assert weekly_defaults.rule_based.bias_spec["timeframes"] == [TimeFrame.W]


def test_reports_dir_includes_module_name():
    config = load_config()
    assert "continuous" in str(config.reports_dir)


def _make_research_config(*, tickers: list[Ticker], target_col: str) -> ResearchConfig:
    return ResearchConfig(
                feature_type=FeatureType.CONTINUOUS,
        tickers=tickers,
        start=datetime(2000, 1, 1),
        end=datetime(2024, 12, 31),
        bias_spec={
            "module_name": "rsi",
            "timeframes": [TimeFrame.D],
            "params": {"lookback": 5},
        },
        target_col=target_col,
        strategy="long",
        use_cache=True,
        populate_cache=False,
        reports_dir=Path("/tmp/test_reports"),
    )


def test_research_config_rejects_log_return_with_multiple_tickers() -> None:
    with pytest.raises(ValueError, match="log_return.*multiple tickers"):
        _make_research_config(tickers=[Ticker.ES, Ticker.NQ], target_col="log_return")


def test_research_config_rejects_raw_return_with_multiple_tickers() -> None:
    with pytest.raises(ValueError, match="raw_return.*multiple tickers"):
        _make_research_config(
            tickers=[Ticker.ES, Ticker.NQ, Ticker.YM],
            target_col="raw_return",
        )


def test_research_config_allows_log_return_with_single_ticker() -> None:
    config = _make_research_config(tickers=[Ticker.ES], target_col="log_return")
    assert config.target_col == "log_return"


def test_research_config_allows_normalized_targets_with_multiple_tickers() -> None:
    ewsd_config = _make_research_config(
        tickers=[Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY],
        target_col="log_return_ewsd",
    )
    atr_config = _make_research_config(
        tickers=[Ticker.ES, Ticker.NQ],
        target_col="log_return_atr",
    )

    assert ewsd_config.target_col == "log_return_ewsd"
    assert atr_config.target_col == "log_return_atr"


def test_load_config_includes_oos_window() -> None:
    config = load_config()
    assert config.oos_window is not None
    assert config.oos_window.train_start == datetime(2000, 1, 1)
    assert config.oos_window.train_end == datetime(2022, 12, 31)
    assert config.oos_window.test_start == datetime(2023, 1, 1)
    assert config.oos_window.test_end == datetime(2025, 9, 18)


def test_load_config_includes_validation_window() -> None:
    config = load_config()
    assert config.validation_window is not None
    assert config.validation_window.train_start == datetime(2000, 1, 1)
    assert config.validation_window.train_end == datetime(2017, 12, 31)
    assert config.validation_window.test_start == datetime(2018, 1, 1)
    assert config.validation_window.test_end == datetime(2022, 12, 31)


def test_load_config_has_flat_eval_fields() -> None:
    """New flat evaluation fields replace WalkforwardDefaultsConfig complexity."""
    config = load_config()
    assert config.top_k == 1
    assert config.objective_metric_name == "t_stat"
    assert config.smoothing_self_weight == 3.0
    assert config.n_jobs == 8
    assert config.output_root == Path("feature_research/shared_results")


def test_base_config_has_flat_eval_fields() -> None:
    """BaseResearchConfig exposes the same flat eval fields."""
    base = load_base_config()
    assert base.top_k == 1
    assert base.objective_metric_key == "t_stat"
    assert base.smoothing_self_weight == 3.0
    assert base.n_jobs == 8
    assert base.output_root == Path("feature_research/shared_results")


def test_run_oos_pipeline_raises_when_oos_window_none() -> None:
    config = load_config()
    config_no_oos = replace(config, oos_window=None)
    with pytest.raises(ValueError, match="OOS window is not set"):
        run_oos_pipeline(config_no_oos)
