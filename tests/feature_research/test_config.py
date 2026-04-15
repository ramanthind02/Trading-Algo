from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from feature_research.config import (
    BinningAnalysisConfig,
    EvaluationDefaultsCatalog,
    EvaluationPhaseDefaultsConfig,
    FeatureType,
    InSampleDefaultsCatalog,
    InSamplePhaseDefaultsConfig,
    OOSWindowConfig,
    PermutationResearchConfig,
    ResearchConfig,
    build_filter_gate_bias_spec,
    build_objective_metric_presets,
    load_config,
    load_config as load_base_config,
)
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from feature_research.pipeline import run_oos_pipeline
from nodes._taxonomy import CANONICAL_MODULE_IMPORTS
from utils.core.enums import Direction, Ticker, TimeFrame


def _minimal_in_sample_catalog(
    timeframe: TimeFrame,
    *,
    continuous_bias_spec: dict[str, Any],
) -> InSampleDefaultsCatalog:
    """Test-only catalog; production specs live in ``load_config()``."""
    mod = str(continuous_bias_spec.get("module_name") or "feature")
    return InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=continuous_bias_spec,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/continuous") / mod,
            binning_params_overrides={"bin_counts": [10]},
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec={
                "module_name": "rebalancing",
                "timeframes": [timeframe],
                "params": {"cross_tickers": ["TLT"]},
            },
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/signed_signal"),
            binning_params_overrides={},
        ),
    )


def test_load_config_returns_research_config() -> None:
    config = load_config()
    assert isinstance(config, ResearchConfig)


def test_permutation_research_config_maps_to_vector_shuffle_only() -> None:
    perm = PermutationResearchConfig(objective_metric=ObjectiveMetricSpec(builtin="sharpe"))
    suite_cfg = perm.to_permutation_test_config()
    assert suite_cfg.in_sample.nreps == perm.nreps
    assert suite_cfg.in_sample.run_stage1 == perm.run_vector_shuffle
    assert suite_cfg.in_sample.n_jobs_combos == perm.n_jobs_combos
    assert suite_cfg.in_sample.n_jobs_reps == perm.n_jobs_reps


def test_load_config_defaults() -> None:
    config = load_config()
    assert config.tickers
    assert config.start < config.end


def test_load_config_signed_signal_specs_use_registered_modules() -> None:
    """Shape + taxonomy only — changing ``load_config()`` node grids does not require test edits."""
    config = load_config()
    for label, spec in (
        ("in_sample", config.bias_spec),
        ("eval", config.eval_bias_spec),
    ):
        mod = spec.get("module_name")
        assert isinstance(mod, str) and mod.strip(), f"{label}: module_name required"
        assert mod in CANONICAL_MODULE_IMPORTS, f"{label}: unknown module {mod!r}"
        assert isinstance(spec.get("params"), dict), f"{label}: params must be a dict"
        assert spec.get("timeframes"), f"{label}: timeframes required"


def test_load_config_permutation_has_objective_metric() -> None:
    config = load_config()
    assert isinstance(config.permutation.objective_metric.builtin, str)
    assert config.permutation.objective_metric.builtin


def test_timeframe_bars_per_year_values() -> None:
    assert TimeFrame.H1.bars_per_year == 5200
    assert TimeFrame.H4.bars_per_year == 1300
    assert TimeFrame.D.bars_per_year == 252


def test_build_objective_metric_presets_uses_timeframe_bars_per_year() -> None:
    weekly_presets = build_objective_metric_presets(TimeFrame.W)
    assert weekly_presets["sharpe_annualized"].kwargs == {"annualization_factor": 52.0}
    assert weekly_presets["sortino_annualized"].kwargs == {"annualization_factor": 52.0}
    assert weekly_presets["calmar_annualized"].kwargs == {"annualization_factor": 52.0}


def test_eval_bias_spec_splits_research_and_evaluation_catalogs() -> None:
    tf = TimeFrame.D
    signed_research = {
        "module_name": "cyclical_rsi_signal",
        "timeframes": [tf],
        "params": {
            "short_period": list(range(2, 7)),
            "long_period": [120],
            "rsi_period": [2],
            "oversold": [-20.0],
            "overbought": [20.0],
            "strategy_mode": ["long"],
            "exit_policy": ["threshold_or_bars"],
            "exit_bars": [5],
        },
    }
    signed_eval = {
        "module_name": "cyclical_rsi_signal",
        "timeframes": [tf],
        "params": {
            "short_period": [4],
            "long_period": [120],
            "rsi_period": [2],
            "oversold": [-20.0],
            "overbought": [20.0],
            "strategy_mode": ["long"],
            "exit_policy": ["threshold_or_bars"],
            "exit_bars": [5],
        },
    }
    research = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=signed_research,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/continuous/dummy"),
            binning_params_overrides={"bin_counts": [10]},
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=signed_research,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/signed_signal"),
            binning_params_overrides={},
        ),
    )
    evaluation_defaults = EvaluationDefaultsCatalog(
        continuous=EvaluationPhaseDefaultsConfig(bias_spec=signed_eval),
        signed_signal=EvaluationPhaseDefaultsConfig(bias_spec=signed_eval),
    )
    cfg = ResearchConfig(
        tickers=[Ticker.ES],
        start=datetime(2000, 1, 1),
        end=datetime(2010, 1, 1),
        permutation=PermutationResearchConfig(
            objective_metric=ObjectiveMetricSpec(builtin="sharpe"),
        ),
        objective_metric_presets=build_objective_metric_presets(tf),
        binning_params=BinningAnalysisConfig(),
        in_sample_defaults=research,
        timeframe=tf,
        feature_type=FeatureType.SIGNED_SIGNAL,
        evaluation_defaults=evaluation_defaults,
    )
    assert cfg.bias_spec != cfg.eval_bias_spec
    assert cfg.eval_bias_spec["params"] == signed_eval["params"]


def test_eval_bias_spec_matches_bias_spec_when_evaluation_defaults_none() -> None:
    tf = TimeFrame.D
    signed_spec = {
        "module_name": "cyclical_rsi_signal",
        "timeframes": [tf],
        "params": {
            "short_period": [4],
            "long_period": [120],
            "rsi_period": [2],
            "oversold": [-20.0],
            "overbought": [20.0],
            "strategy_mode": ["long"],
            "exit_policy": ["threshold_or_bars"],
            "exit_bars": [5],
        },
    }
    research = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=signed_spec,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/continuous/dummy"),
            binning_params_overrides={"bin_counts": [10]},
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=signed_spec,
            target_col="log_return_ewsd",
            strategy=Direction.LONG,
            reports_dir=Path("feature_research/in_sample/results/signed_signal"),
            binning_params_overrides={},
        ),
    )
    cfg = ResearchConfig(
        tickers=[Ticker.ES],
        start=datetime(2000, 1, 1),
        end=datetime(2010, 1, 1),
        permutation=PermutationResearchConfig(
            objective_metric=ObjectiveMetricSpec(builtin="sharpe"),
        ),
        objective_metric_presets=build_objective_metric_presets(tf),
        binning_params=BinningAnalysisConfig(),
        in_sample_defaults=research,
        timeframe=tf,
        feature_type=FeatureType.SIGNED_SIGNAL,
        evaluation_defaults=None,
    )
    assert cfg.eval_bias_spec == cfg.bias_spec


def test_in_sample_defaults_catalog_respects_timeframe() -> None:
    tf = TimeFrame.W
    weekly_defaults = _minimal_in_sample_catalog(
        tf,
        continuous_bias_spec={
            "module_name": "filter_gate",
            "timeframes": [tf],
            "params": {
                "filter_module": "adx_filter",
                "filter_params": {"length": 20, "threshold": 20.0, "compare": "below"},
                "signal_module": "cyclical_rsi",
                "signal_params": {"short_period": [4], "long_period": [120], "rsi_period": [2]},
            },
        },
    )
    assert weekly_defaults.continuous.bias_spec["timeframes"] == [TimeFrame.W]
    assert "timeframes" in weekly_defaults.signed_signal.bias_spec
    assert weekly_defaults.for_feature_type(FeatureType.SIGNED_SIGNAL) is weekly_defaults.signed_signal


def test_build_filter_gate_bias_spec_is_node_agnostic() -> None:
    tf = TimeFrame.D
    spec = build_filter_gate_bias_spec(
        tf,
        filter_module="adx_filter",
        filter_params={"length": 20, "threshold": 20.0},
        signal_module="cyclical_rsi",
        signal_params={"short_period": [4], "long_period": [120], "rsi_period": [2]},
    )
    assert spec["module_name"] == "filter_gate"
    assert spec["params"]["filter_module"] == "adx_filter"
    assert spec["params"]["signal_module"] == "cyclical_rsi"


def test_manual_in_sample_catalog_with_custom_gate() -> None:
    tf = TimeFrame.D
    gated = build_filter_gate_bias_spec(
        tf,
        filter_module="buy_hold",
        filter_params={},
        signal_module="cyclical_rsi",
        signal_params={"short_period": [4], "long_period": [120], "rsi_period": [2]},
    )
    cat = _minimal_in_sample_catalog(tf, continuous_bias_spec=gated)
    assert cat.continuous.bias_spec["module_name"] == "filter_gate"
    assert "filter_gate" in str(cat.continuous.reports_dir)


def test_reports_dir_includes_module_name() -> None:
    config = load_config()
    signed_signal_config = config.in_sample_defaults.signed_signal
    assert signed_signal_config is not None


def test_load_config_includes_oos_window() -> None:
    config = load_config()
    assert config.oos_window is not None
    assert config.oos_window.train_start == datetime(2000, 1, 1)
    assert config.oos_window.train_end == datetime(2022, 12, 31)
    assert config.oos_window.test_start == datetime(2023, 1, 1)
    assert config.oos_window.test_end == datetime(2025, 9, 18)


def test_training_window_bounds_prefers_oos_train() -> None:
    config = load_config()
    assert config.oos_window is not None
    assert config.training_window_bounds == (
        config.oos_window.train_start,
        config.oos_window.train_end,
    )


def test_training_window_bounds_falls_back_to_validation_train() -> None:
    config = load_config()
    config_no_oos = replace(config, oos_window=None)
    assert config_no_oos.validation_window is not None
    assert config_no_oos.training_window_bounds == (
        config_no_oos.validation_window.train_start,
        config_no_oos.validation_window.train_end,
    )


def test_training_window_bounds_falls_back_to_global_range() -> None:
    config = load_config()
    stripped = replace(config, oos_window=None, validation_window=None)
    assert stripped.training_window_bounds == (stripped.start, stripped.end)


def test_load_config_includes_validation_window() -> None:
    config = load_config()
    assert config.validation_window is not None
    assert config.validation_window.train_start == datetime(2000, 1, 1)
    assert config.validation_window.train_end == datetime(2018, 12, 31)
    assert config.validation_window.test_start == datetime(2019, 1, 1)
    assert config.validation_window.test_end == datetime(2022, 12, 31)


def test_load_config_has_flat_eval_fields() -> None:
    """Flat evaluation fields: n_jobs, output_root (walkforward params removed)."""
    config = load_config()
    assert config.n_jobs == 8
    assert config.output_root == Path("feature_research/shared_results")


def test_base_config_has_flat_eval_fields() -> None:
    """ResearchConfig exposes the same flat eval fields."""
    base = load_base_config()
    assert base.n_jobs == 8
    assert base.output_root == Path("feature_research/shared_results")


def test_run_oos_pipeline_raises_when_oos_window_none() -> None:
    config = load_config()
    config_no_oos = replace(config, oos_window=None)
    with pytest.raises(ValueError, match="OOS window is not set"):
        run_oos_pipeline(config_no_oos)
