"""Smoke test: all dataclasses importable and instantiatable."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from feature_selection.eda.eda_dataclasses import (
    DescriptiveStats, TemporalStability, CorrelationAnalysis,
    CommonEDAPlots, CommonEDAStats,
    DecileBinStats, DecileAnalysis, MonotonicityTest,
    DistributionDiagnostics, ContinuousEDAPlots, ContinuousEDAStats,
    LevelStats, PerLevelStats, BootstrapCI, BootstrapCIResults,
    TransitionMatrix, RuleBasedEDAPlots, RuleBasedEDAStats,
)


def _dummy_fig() -> plt.Figure:
    fig, _ = plt.subplots()
    plt.close(fig)
    return fig


def test_descriptive_stats_instantiates() -> None:
    s = DescriptiveStats(
        min_val=0.0, max_val=1.0, mean=0.5, median=0.5,
        std=0.1, skew=0.0, kurtosis=3.0,
        nan_count=0, nan_pct=0.0, sample_size=100,
    )
    assert s.mean == 0.5


def test_level_stats_is_reliable_flag() -> None:
    ls = LevelStats(
        level=1, mean_return=0.01, volatility=0.1,
        sharpe=0.1, adjusted_sharpe=0.1,
        sample_count=5, is_reliable=False,
    )
    assert ls.is_reliable is False


def test_dataclasses_are_frozen() -> None:
    s = DescriptiveStats(
        min_val=0.0, max_val=1.0, mean=0.5, median=0.5,
        std=0.1, skew=0.0, kurtosis=3.0,
        nan_count=0, nan_pct=0.0, sample_size=100,
    )
    with pytest.raises(Exception):
        s.mean = 99.0  # type: ignore[misc]
