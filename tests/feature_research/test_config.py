from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from feature_research.config import (
    BaseResearchConfig,
    FeatureType,
    OBJECTIVE_METRIC_PRESETS,
    OOSWindowConfig,
    PermutationResearchConfig,
    WalkforwardDefaultsConfig,
    load_config as load_base_config,
)
from feature_research.in_sample.config import ResearchConfig, load_config
from feature_research.pipeline import run_oos_pipeline
from feature_research.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
    WeightLayerAlgorithm,
)
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


def test_reports_dir_includes_module_name():
    config = load_config()
    assert "continuous" in str(config.reports_dir)


def test_load_config_includes_walkforward_defaults() -> None:
    config = load_config()

    assert isinstance(config.walkforward, WalkforwardResearchConfig)
    assert config.walkforward.enabled is False
    assert config.walkforward.train_start >= config.start
    assert config.walkforward.train_end < config.end
    assert config.walkforward.output_root == Path("feature_research/shared_results")


def test_load_config_enables_per_fold_tearsheets_for_oos() -> None:
    config = load_config()

    assert config.walkforward.output_per_fold_tearsheets is False


def test_load_config_exposes_walkforward_selection_control() -> None:
    config = load_config()

    assert config.walkforward_selection_method == WalkforwardSelectionMethod.TOP_K
    assert config.walkforward.selection_method == config.walkforward_selection_method


def test_load_config_exposes_weight_layer_algorithm_control() -> None:
    config = load_config()

    assert (
        config.weight_layer_algorithm
        == WeightLayerAlgorithm.INVERSE_CORRELATION
    )
    assert config.walkforward.weight_layer_algorithm == config.weight_layer_algorithm


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


def test_research_config_coerces_top_level_controls_from_strings() -> None:
    walkforward = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2023, 1, 1),
        selection_method="top_k",
        weight_layer_algorithm="equal_grouped",
    )

    config = ResearchConfig(
                feature_type=FeatureType.CONTINUOUS,
        tickers=[Ticker.ES],
        start=datetime(2000, 1, 1),
        end=datetime(2024, 12, 31),
        bias_spec={
            "module_name": "rsi",
            "timeframes": [TimeFrame.D],
            "params": {"lookback": 5},
        },
        target_col="log_return",
        strategy="long",
        use_cache=True,
        populate_cache=False,
        reports_dir=Path("/tmp/test_reports"),
        walkforward_selection_method="top_k",
        weight_layer_algorithm="equal_grouped",
        walkforward=walkforward,
    )

    assert config.walkforward_selection_method == WalkforwardSelectionMethod.TOP_K
    assert config.weight_layer_algorithm == WeightLayerAlgorithm.EQUAL_GROUPED


def test_research_config_rejects_walkforward_selection_method_mismatch() -> None:
    walkforward = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2023, 1, 1),
        selection_method=WalkforwardSelectionMethod.ENHANCED,
    )

    with pytest.raises(ValueError, match="walkforward_selection_method"):
        ResearchConfig(
                feature_type=FeatureType.CONTINUOUS,
            tickers=[Ticker.ES],
            start=datetime(2000, 1, 1),
            end=datetime(2024, 12, 31),
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": 5},
            },
            target_col="log_return",
            strategy="long",
            use_cache=True,
            populate_cache=False,
            reports_dir=Path("/tmp/test_reports"),
            walkforward_selection_method=WalkforwardSelectionMethod.TOP_K,
            walkforward=walkforward,
        )


def test_research_config_rejects_weight_layer_algorithm_mismatch() -> None:
    walkforward = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2023, 1, 1),
        weight_layer_algorithm=WeightLayerAlgorithm.EQUAL_FLAT,
    )

    with pytest.raises(ValueError, match="weight_layer_algorithm"):
        ResearchConfig(
                feature_type=FeatureType.CONTINUOUS,
            tickers=[Ticker.ES],
            start=datetime(2000, 1, 1),
            end=datetime(2024, 12, 31),
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": 5},
            },
            target_col="log_return",
            strategy="long",
            use_cache=True,
            populate_cache=False,
            reports_dir=Path("/tmp/test_reports"),
            weight_layer_algorithm=WeightLayerAlgorithm.INVERSE_CORRELATION,
            walkforward=walkforward,
        )


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


def test_build_walkforward_derives_bounds_from_window_and_steps() -> None:
    """build_walkforward() derives train_start/train_end from train_window_years, test_window_years, num_steps."""
    base = BaseResearchConfig(
        tickers=[Ticker.ES],
        start=datetime(2000, 1, 1),
        end=datetime(2023, 12, 31),
        use_cache=False,
        populate_cache=False,
        permutation=PermutationResearchConfig(
            OBJECTIVE_METRIC_PRESETS["t_stat"],
            top_k=3,
            min_folds_stable=1,
            fold_years=1,
        ),
        walkforward_defaults=WalkforwardDefaultsConfig(
            top_k=3,
            train_window_years=15.0,
            test_window_years=2.0,
            num_steps=4,
            selection_method=WalkforwardSelectionMethod.TOP_K,
            min_folds_stable=1,
            fold_years=1,
            objective_metric_name="t_stat",
        ),
    )
    wf = base.build_walkforward()
    # First fold train ends at end - num_steps * test_step_days; test_step_days = 730
    assert wf.num_steps == 4
    assert wf.test_step == 730
    assert wf.train_start >= base.start
    assert wf.train_end < base.end
    assert (wf.train_end - wf.train_start).days >= 365 * 14  # ~15 years


def test_run_oos_pipeline_raises_when_oos_window_none() -> None:
    config = load_config()
    config_no_oos = replace(config, oos_window=None)
    with pytest.raises(ValueError, match="OOS window is not set"):
        run_oos_pipeline(config_no_oos)
