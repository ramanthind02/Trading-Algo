from __future__ import annotations

import numpy as np
import pandas as pd

from research.evaluation.holdout_robustness import HoldoutRobustnessConfig, run_holdout_robustness_pipeline


def test_run_holdout_robustness_pipeline_without_rank_correlation() -> None:
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    is_returns = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, size=120), index=index)
    holdout_returns = pd.Series(np.random.default_rng(2).normal(0.0003, 0.01, size=80), index=index[:80])
    report = run_holdout_robustness_pipeline(
        is_returns,
        holdout_returns,
        config=HoldoutRobustnessConfig(n_bootstrap=50, random_seed=1),
        include_rank_correlation=False,
    )
    assert report.rank_correlation.passed is True
    assert report.sharpe_comparison.sr_is == report.sharpe_comparison.sr_is


def test_run_holdout_robustness_shrinks_rolling_window_for_short_holdout() -> None:
    index = pd.date_range("2020-01-01", periods=500, freq="B")
    is_returns = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, size=500), index=index)
    holdout_returns = pd.Series(
        np.random.default_rng(4).normal(0.0003, 0.01, size=29),
        index=pd.date_range("2024-01-01", periods=29, freq="B"),
    )
    report = run_holdout_robustness_pipeline(
        is_returns,
        holdout_returns,
        config=HoldoutRobustnessConfig(
            n_bootstrap=50,
            random_seed=1,
            rolling_window=60,
        ),
        include_rank_correlation=False,
    )
    assert report.rolling_sharpe_zscore.window == 29
    assert report.rolling_sharpe_zscore.n_obs_val == 29
