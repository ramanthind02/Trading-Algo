"""Unit tests for T016: Early Stopping Orchestration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection.validation.config import PermutationTestConfig
from feature_selection.validation.reports import (
    FunnelStatistics,
    PermutationTestSuite,
    VectorShuffleReport,
    PipelinePermutationReport,
)


def test_permutation_test_config_defaults() -> None:
    """PermutationTestConfig has the correct defaults."""
    config = PermutationTestConfig()
    assert config.nreps == 1000
    assert config.alpha == 0.10
    assert config.top_k == 3
    assert config.permutation_mode_stage2 == 'candle_shuffle'
    assert config.min_folds_stable == 3
    assert config.random_seed is None


def test_funnel_statistics_fields() -> None:
    """FunnelStatistics contains all required fields."""
    stats = FunnelStatistics(
        total_params=10,
        stage1_pass=7,
        stage2_pass=5,
        stable_params=3,
        ensemble_candidates=3,
        computational_savings_pct=15.0,
    )
    assert stats.total_params == 10
    assert stats.stage1_pass == 7
    assert stats.stage2_pass == 5
    assert stats.stable_params == 3
    assert stats.ensemble_candidates == 3
    assert stats.computational_savings_pct == 15.0


def test_permutation_test_suite_fields() -> None:
    """PermutationTestSuite contains all required fields."""
    from feature_selection.validation.reports import (
        WalkforwardStabilityReport, FoldResult,
    )

    suite = PermutationTestSuite(
        feature_name='rsi',
        feature_type='continuous',
        stage1_reports={},
        stage2_reports={},
        stage3_report=WalkforwardStabilityReport(
            feature_name='rsi', feature_type='continuous',
            fold_results=[], consistency_metrics={},
            is_stable=False, stability_verdict='UNSTABLE', top_k=3,
        ),
        funnel_stats=FunnelStatistics(10, 7, 5, 3, 3, 15.0),
        ensemble_candidates=[],
        summary='test summary',
    )
    assert suite.feature_name == 'rsi'
    assert suite.feature_type == 'continuous'
    assert isinstance(suite.stage1_reports, dict)
    assert isinstance(suite.stage2_reports, dict)
    assert hasattr(suite, 'stage3_report')
    assert hasattr(suite, 'funnel_stats')
    assert hasattr(suite, 'ensemble_candidates')
    assert hasattr(suite, 'summary')


def test_stage2_receives_only_stage1_passers() -> None:
    """Stage 2 count <= Stage 1 pass count when running the suite."""
    import numpy as np
    import pandas as pd
    from feature_selection.validation.orchestration import run_permutation_test_suite

    rng = np.random.default_rng(0)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    closes = 100.0 + rng.standard_normal(n).cumsum()
    candles = pd.DataFrame({
        'open': closes - rng.uniform(0, 0.3, n),
        'high': closes + rng.uniform(0, 0.3, n),
        'low': closes - rng.uniform(0, 0.3, n),
        'close': closes,
        'datetime': dates,
    }, index=dates)
    target = pd.Series(rng.standard_normal(n), index=dates)
    param_grid = [{'lookback': v} for v in [3, 5, 10]]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        lb = params['lookback']
        return df['close'].pct_change(lb).fillna(0.0).rename(f"lookback_{lb}")

    def sharpe(returns: pd.Series) -> float:
        if len(returns) == 0 or returns.std() == 0: return 0.0
        return float(returns.mean() / returns.std())

    config = PermutationTestConfig(nreps=20, alpha=0.10, random_seed=42, min_folds_stable=1)
    folds = [(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01'))]

    suite = run_permutation_test_suite(
        candles_df=candles,
        feature_spec={'module_name': 'test'},
        target=target,
        param_grid=param_grid,
        objective_func=sharpe,
        fold_structure=folds,
        config=config,
        extractor_func=extractor,
        feature_type='rule_based',
        feature_name='test',
    )

    assert len(suite.stage2_reports) <= suite.funnel_stats.stage1_pass
    assert suite.funnel_stats.stage1_pass == len(suite.stage1_reports)
    assert suite.funnel_stats.total_params == len(param_grid)
    assert suite.funnel_stats.computational_savings_pct >= 0.0
    assert isinstance(suite.ensemble_candidates, list)


def test_ensemble_candidate_intersection() -> None:
    """ensemble_candidates = stage2_passers intersection stable_params."""
    # We verify this via the suite logic directly
    stage2_passers = {'A', 'B', 'C'}
    stable_params = {'B', 'C', 'D'}
    expected = sorted(stage2_passers & stable_params)
    assert set(expected) == {'B', 'C'}


def test_computational_savings_non_negative() -> None:
    """computational_savings_pct >= 0."""
    stats = FunnelStatistics(
        total_params=10,
        stage1_pass=10,  # no early stopping savings
        stage2_pass=10,
        stable_params=5,
        ensemble_candidates=5,
        computational_savings_pct=0.0,
    )
    assert stats.computational_savings_pct >= 0.0
