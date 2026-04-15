"""Unit tests for eda_reporter.py (T004)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_selection.eda.eda_dataclasses import (
    BootstrapCI,
    BootstrapCIResults,
    CommonEDAStats,
    ContinuousEDAReport,
    ContinuousEDAStats,
    CorrelationAnalysis,
    DecileAnalysis,
    DecileBinStats,
    DescriptiveStats,
    DiagnosticFlags,
    DistributionDiagnostics,
    EDAConfig,
    EDAMetadata,
    LevelStats,
    PerLevelStats,
    QuintileSpread,
    RuleBasedEDAReport,
    RuleBasedEDAStats,
)
from feature_selection.eda.eda_reporter import (
    _param_combo_hash,
    compute_diagnostic_flags,
    load_eda_report,
    run_eda_for_continuous_feature,
    run_eda_for_rule_based_feature,
    save_eda_report,
)
from feature_selection.eda.rule_based_eda import compute_per_level_stats
from utils.core.enums import Ticker, TimeFrame


def _metadata(feature_name: str = "feat") -> EDAMetadata:
    return EDAMetadata(
        feature_name=feature_name,
        param_combo={"lookback": 20, "threshold": 0.5},
        timeframe=TimeFrame.D,
        ticker=Ticker.ES,
        timestamp=datetime(2024, 1, 1),
    )


def _common_stats(
    *,
    nan_pct: float = 0.0,
    sample_size: int = 300,
    std: float = 1.0,
    skew: float = 0.0,
) -> CommonEDAStats:
    base_desc = DescriptiveStats(
        min_val=-1.0,
        max_val=1.0,
        mean=0.0,
        median=0.0,
        std=std,
        skew=skew,
        kurtosis=0.0,
        nan_count=int(sample_size * nan_pct),
        nan_pct=nan_pct,
        sample_size=sample_size,
    )
    target_desc = DescriptiveStats(
        min_val=-1.0,
        max_val=1.0,
        mean=0.0,
        median=0.0,
        std=1.0,
        skew=0.0,
        kurtosis=0.0,
        nan_count=0,
        nan_pct=0.0,
        sample_size=sample_size,
    )
    return CommonEDAStats(
        feature_stats=base_desc,
        target_stats=target_desc,
        correlation_analysis=CorrelationAnalysis(pearson=0.4, spearman=0.35, lagged_correlations={1: 0.2}),
    )


def _continuous_feature_stats() -> ContinuousEDAStats:
    n_bins = 5
    return ContinuousEDAStats(
        decile_analysis=DecileAnalysis(
            bin_stats=DecileBinStats(
                bin_edges=np.linspace(-1.0, 1.0, n_bins + 1),
                mean_return=np.linspace(-0.2, 0.2, n_bins),
                volatility=np.ones(n_bins),
                sharpe=np.linspace(-0.2, 0.2, n_bins),
                t_stat=np.linspace(-1.0, 1.0, n_bins),
                sample_count=np.full(n_bins, 20),
            ),
            overall_trend="monotonic_increasing",
        ),
        distribution_diagnostics=DistributionDiagnostics(
            skewness=0.1,
            kurtosis=0.0,
            normality_test_stat=0.98,
            normality_p_value=0.2,
            is_normal=True,
        ),
        quintile_spread=QuintileSpread(
            quintile_means=np.array([-0.02, -0.01, 0.0, 0.01, 0.02]),
            spread=0.04,
        ),
    )


def _rule_feature_stats() -> RuleBasedEDAStats:
    per_level = PerLevelStats(
        stats_by_level={
            -1: LevelStats(level=-1, mean_return=-0.01, volatility=0.1, sharpe=-0.1, adjusted_sharpe=-0.1, sample_count=50, is_reliable=True),
            0: LevelStats(level=0, mean_return=0.0, volatility=0.1, sharpe=0.0, adjusted_sharpe=0.0, sample_count=50, is_reliable=True),
            1: LevelStats(level=1, mean_return=0.01, volatility=0.1, sharpe=0.1, adjusted_sharpe=0.1, sample_count=50, is_reliable=True),
        }
    )
    bootstrap = BootstrapCIResults(
        ci_by_level={
            -1: BootstrapCI(level=-1, mean_return=-0.01, ci_lower=-0.02, ci_upper=0.0, bootstrap_distribution=np.array([-0.01])),
            0: BootstrapCI(level=0, mean_return=0.0, ci_lower=-0.01, ci_upper=0.01, bootstrap_distribution=np.array([0.0])),
            1: BootstrapCI(level=1, mean_return=0.01, ci_lower=0.0, ci_upper=0.02, bootstrap_distribution=np.array([0.01])),
        }
    )
    return RuleBasedEDAStats(
        per_level_stats=per_level,
        bootstrap_ci_results=bootstrap,
    )


def test_high_nan_warning() -> None:
    flags = compute_diagnostic_flags(_common_stats(nan_pct=0.11), _continuous_feature_stats())
    assert any("nan" in warning.lower() for warning in flags.warnings)


def test_low_sample_warning() -> None:
    flags = compute_diagnostic_flags(_common_stats(sample_size=200), _continuous_feature_stats())
    assert any("sample" in warning.lower() for warning in flags.warnings)


def test_zero_variance_red_flag_and_not_viable() -> None:
    flags = compute_diagnostic_flags(_common_stats(std=0.0), _continuous_feature_stats())
    assert any("variance" in red_flag.lower() for red_flag in flags.red_flags)
    assert flags.is_viable is False


def test_extreme_skew_red_flag_and_not_viable() -> None:
    flags = compute_diagnostic_flags(_common_stats(skew=6.0), _continuous_feature_stats())
    assert any("skew" in red_flag.lower() for red_flag in flags.red_flags)
    assert flags.is_viable is False


def test_all_nan_feature_red_flag() -> None:
    flags = compute_diagnostic_flags(
        _common_stats(nan_pct=1.0, sample_size=100),
        _continuous_feature_stats(),
    )
    assert any("all nan" in red_flag.lower() for red_flag in flags.red_flags)
    assert flags.is_viable is False


def test_clean_data_viable() -> None:
    flags = compute_diagnostic_flags(_common_stats(), _continuous_feature_stats())
    assert flags.red_flags == []
    assert flags.is_viable is True


def test_persistence_and_reload_roundtrip(tmp_path: Path) -> None:
    n = 320
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    feature = pd.Series(rng.normal(size=n), index=idx)
    target = pd.Series(0.25 * feature.values + rng.normal(scale=0.2, size=n), index=idx)
    metadata = _metadata(feature_name="cont_feat")
    report = run_eda_for_continuous_feature(
        feature=feature,
        target=target,
        timestamps=idx,
        metadata=metadata,
        config=EDAConfig(n_bins=10, rolling_window=50, max_lag=3),
    )

    report_path = save_eda_report(report=report, output_dir=tmp_path)
    loaded = load_eda_report(report_path)

    assert isinstance(loaded, ContinuousEDAReport)
    assert loaded.metadata == report.metadata
    assert loaded.diagnostics == report.diagnostics
    assert loaded.common_stats.feature_stats.sample_size == report.common_stats.feature_stats.sample_size
    assert loaded.continuous_stats.decile_analysis.overall_trend == report.continuous_stats.decile_analysis.overall_trend


def test_directory_structure_created(tmp_path: Path) -> None:
    n = 320
    idx = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(0)
    feature = pd.Series(rng.normal(size=n), index=idx)
    target = pd.Series(rng.normal(size=n), index=idx)
    report = run_eda_for_continuous_feature(feature, target, idx, _metadata("tree_feat"), EDAConfig(n_bins=10, rolling_window=40))
    report_path = save_eda_report(report=report, output_dir=tmp_path)

    assert (report_path / "metadata.json").exists()
    assert (report_path / "common_stats.json").exists()
    assert (report_path / "feature_stats.json").exists()
    assert (report_path / "diagnostics.json").exists()
    assert not (report_path / "plots").exists()


def test_overwrite_false_raises(tmp_path: Path) -> None:
    n = 320
    idx = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(1)
    feature = pd.Series(rng.normal(size=n), index=idx)
    target = pd.Series(rng.normal(size=n), index=idx)
    report = run_eda_for_continuous_feature(feature, target, idx, _metadata("overwrite_feat"), EDAConfig(n_bins=10, rolling_window=40))

    save_eda_report(report=report, output_dir=tmp_path, overwrite=False)
    with pytest.raises(FileExistsError):
        save_eda_report(report=report, output_dir=tmp_path, overwrite=False)


def test_param_combo_hash_deterministic() -> None:
    hash_a = _param_combo_hash({"a": 1, "b": 2, "c": [1, 2, 3]})
    hash_b = _param_combo_hash({"c": [1, 2, 3], "b": 2, "a": 1})
    assert hash_a == hash_b
    assert len(hash_a) == 8


def test_per_level_stats_use_strategy_returns_for_short_leg() -> None:
    """Short (f=-1) stats must use signal*target, not raw target, or Sharpe looks inverted."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    feature = pd.Series([-1.0, -1.0, 1.0], index=idx, dtype=float)
    target = pd.Series([0.01, -0.02, 0.01], index=idx, dtype=float)
    stats = compute_per_level_stats(feature, target)
    # At -1: strategy returns -0.01, +0.02 -> mean +0.005
    assert stats.stats_by_level[-1].mean_return == pytest.approx(0.005)
    # Flat: strategy P&L is always 0
    feature_flat = pd.Series([0.0, 0.0, 1.0], index=idx, dtype=float)
    stats_flat = compute_per_level_stats(feature_flat, target)
    assert stats_flat.stats_by_level[0].mean_return == pytest.approx(0.0)


def test_per_level_stats_infers_stacked_discrete_levels() -> None:
    """Stacked nodes (e.g. BasicMR / BasicBreakout) use integers outside {-1, 0, 1}."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    feature = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0], index=idx, dtype=float)
    target = pd.Series([0.01, -0.01, 0.0, 0.02, -0.02], index=idx, dtype=float)
    stats = compute_per_level_stats(feature, target)
    assert set(stats.stats_by_level.keys()) == {-2, -1, 0, 1, 2}


def test_per_level_stats_explicit_ternary_rejects_stacked_values() -> None:
    idx = pd.bdate_range("2020-01-01", periods=2)
    feature = pd.Series([-2.0, 1.0], index=idx, dtype=float)
    target = pd.Series([0.01, 0.01], index=idx, dtype=float)
    with pytest.raises(ValueError, match="unexpected feature level"):
        compute_per_level_stats(feature, target, levels=[-1, 0, 1])


def test_smoke_run_for_both_report_types() -> None:
    n = 320
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(123)

    continuous_feature = pd.Series(rng.normal(size=n), index=idx)
    continuous_target = pd.Series(0.2 * continuous_feature.values + rng.normal(scale=0.15, size=n), index=idx)
    cont_report = run_eda_for_continuous_feature(
        feature=continuous_feature,
        target=continuous_target,
        timestamps=idx,
        metadata=_metadata("smoke_cont"),
        config=EDAConfig(n_bins=10, rolling_window=50, max_lag=3),
    )
    assert isinstance(cont_report, ContinuousEDAReport)
    assert isinstance(cont_report.diagnostics, DiagnosticFlags)

    rule_feature = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx, dtype=float)
    rule_target = pd.Series(0.1 * rule_feature.values + rng.normal(scale=0.2, size=n), index=idx)
    rule_report = run_eda_for_rule_based_feature(
        feature=rule_feature,
        target=rule_target,
        timestamps=idx,
        metadata=_metadata("smoke_rule"),
        config=EDAConfig(rolling_window=50, bootstrap_iterations=150),
    )
    assert isinstance(rule_report, RuleBasedEDAReport)
    assert isinstance(rule_report.diagnostics, DiagnosticFlags)
