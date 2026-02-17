"""Unit tests for T015: Walkforward Stability Analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection.validation.stability_analysis import (
    _compute_neighbor_smoothed_objectives,
    _compute_consistency_metrics,
    _param_combo_name,
    run_walkforward_stability,
)
from feature_selection.validation.reports import FoldResult, WalkforwardStabilityReport


def _sharpe(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std())


def _make_candles(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 + rng.standard_normal(n).cumsum()
    opens = closes - rng.uniform(0.0, 0.3, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.0, 0.3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.0, 0.3, n)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    return pd.DataFrame(
        {'open': opens, 'high': highs, 'low': lows, 'close': closes,
         'datetime': dates},
        index=dates,
    )


def test_neighbor_smoothing_1d() -> None:
    """1D grid neighbor averages match hand-calculated values."""
    param_grid = [{'lookback': v} for v in [3, 5, 10, 14]]
    raw = {
        'lookback_3': 0.2,
        'lookback_5': 0.4,
        'lookback_10': 0.6,
        'lookback_14': 0.3,
    }
    smoothed = _compute_neighbor_smoothed_objectives(['lookback'], param_grid, raw)

    # lookback_3: neighbors=[5]; smoothed = mean([0.2, 0.4]) = 0.3
    assert abs(smoothed['lookback_3'] - np.mean([0.2, 0.4])) < 1e-9
    # lookback_5: neighbors=[3, 10]; smoothed = mean([0.4, 0.2, 0.6])
    assert abs(smoothed['lookback_5'] - np.mean([0.4, 0.2, 0.6])) < 1e-9
    # lookback_10: neighbors=[5, 14]; smoothed = mean([0.6, 0.4, 0.3])
    assert abs(smoothed['lookback_10'] - np.mean([0.6, 0.4, 0.3])) < 1e-9
    # lookback_14: neighbors=[10]; smoothed = mean([0.3, 0.6]) = 0.45
    assert abs(smoothed['lookback_14'] - np.mean([0.3, 0.6])) < 1e-9


def test_neighbor_smoothing_single_point() -> None:
    """Single-point grid: smoothed equals original (no neighbors)."""
    param_grid = [{'lookback': 5}]
    raw = {'lookback_5': 0.7}
    smoothed = _compute_neighbor_smoothed_objectives(['lookback'], param_grid, raw)
    assert abs(smoothed['lookback_5'] - 0.7) < 1e-9


def test_neighbor_smoothing_2d_uniform() -> None:
    """2D grid with uniform objectives: all smoothed values == 1.0."""
    param_grid = [
        {'fast': f, 'slow': s}
        for f in [8, 16, 32]
        for s in [32, 64]
    ]
    raw = {_param_combo_name(p): 1.0 for p in param_grid}
    smoothed = _compute_neighbor_smoothed_objectives(['fast', 'slow'], param_grid, raw)
    for k, v in smoothed.items():
        assert abs(v - 1.0) < 1e-9, f'Expected 1.0 for {k}, got {v}'


def test_fold_result_fields() -> None:
    """FoldResult contains all required fields."""
    fr = FoldResult(
        fold_id='fold_0',
        fold_period=('2020-01-01', '2021-12-31'),
        top_k_params=['lookback_3', 'lookback_5'],
        smoothed_objectives={'lookback_3': 0.5, 'lookback_5': 0.6},
        passed_permutation_overlay=[True, False],
    )
    assert fr.fold_id == 'fold_0'
    assert fr.fold_period == ('2020-01-01', '2021-12-31')
    assert fr.top_k_params == ['lookback_3', 'lookback_5']
    assert fr.smoothed_objectives == {'lookback_3': 0.5, 'lookback_5': 0.6}
    assert fr.passed_permutation_overlay == [True, False]


def test_walkforward_stability_report_fields() -> None:
    """WalkforwardStabilityReport contains all required fields."""
    report = WalkforwardStabilityReport(
        feature_name='rsi',
        feature_type='continuous',
        fold_results=[],
        consistency_metrics={'overlap_rate': 0.8},
        is_stable=True,
        stability_verdict='STABLE',
        top_k=3,
    )
    assert report.feature_name == 'rsi'
    assert report.feature_type == 'continuous'
    assert report.fold_results == []
    assert 'overlap_rate' in report.consistency_metrics
    assert report.is_stable is True
    assert report.top_k == 3


def test_stable_feature_detection() -> None:
    """Folds with high top-K overlap -> is_stable=True."""
    candles = _make_candles(100)
    target = pd.Series(np.random.default_rng(0).standard_normal(100), index=candles.index)
    param_grid = [{'lookback': v} for v in [3, 5, 10]]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        lb = params['lookback']
        ret = df['close'].pct_change(lb).fillna(0.0)
        return ret.rename(f'lookback_{lb}')

    folds = [
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-04-01')),
        (pd.Timestamp('2020-04-01'), pd.Timestamp('2020-07-01')),
        (pd.Timestamp('2020-07-01'), pd.Timestamp('2020-10-01')),
    ]

    report = run_walkforward_stability(
        candles_df=candles,
        extractor_func=extractor,
        target=target,
        objective_func=_sharpe,
        param_grid=param_grid,
        fold_structure=folds,
        top_k=1,
        feature_type='rule_based',
        feature_name='test',
    )
    assert isinstance(report, WalkforwardStabilityReport)
    assert len(report.fold_results) >= 1
    assert 'overlap_rate' in report.consistency_metrics
    assert isinstance(report.is_stable, bool)
    assert isinstance(report.stability_verdict, str)


def test_permutation_overlay_flags() -> None:
    """passed_permutation_overlay matches permutation_passers membership."""
    candles = _make_candles(80)
    target = pd.Series(np.random.default_rng(1).standard_normal(80), index=candles.index)
    param_grid = [{'lookback': v} for v in [3, 5, 10]]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        return df['close'].pct_change(params['lookback']).fillna(0.0).rename(f"lookback_{params['lookback']}")

    passers = {'lookback_3', 'lookback_10'}
    folds = [(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01'))]

    report = run_walkforward_stability(
        candles_df=candles,
        extractor_func=extractor,
        target=target,
        objective_func=_sharpe,
        param_grid=param_grid,
        fold_structure=folds,
        top_k=2,
        permutation_passers=passers,
        feature_type='rule_based',
    )
    fold = report.fold_results[0]
    for param_name, flag in zip(fold.top_k_params, fold.passed_permutation_overlay):
        assert flag == (param_name in passers), \
            f'Wrong overlay for {param_name}: expected {param_name in passers}, got {flag}'


def test_consistency_metrics_two_folds_identical_top_k() -> None:
    """Two folds with identical top_k -> overlap_rate=1.0."""
    fr1 = FoldResult(
        fold_id='fold_0', fold_period=('2020-01-01', '2020-06-01'),
        top_k_params=['a', 'b'], smoothed_objectives={'a': 0.8, 'b': 0.6},
        passed_permutation_overlay=[True, True],
    )
    fr2 = FoldResult(
        fold_id='fold_1', fold_period=('2020-06-01', '2020-12-31'),
        top_k_params=['a', 'b'], smoothed_objectives={'a': 0.7, 'b': 0.5},
        passed_permutation_overlay=[True, True],
    )
    metrics = _compute_consistency_metrics([fr1, fr2], top_k=2)
    assert abs(metrics['overlap_rate'] - 1.0) < 1e-9


def test_consistency_metrics_two_folds_disjoint_top_k() -> None:
    """Two folds with disjoint top_k -> overlap_rate=0.0."""
    fr1 = FoldResult(
        fold_id='fold_0', fold_period=('2020-01-01', '2020-06-01'),
        top_k_params=['a', 'b'], smoothed_objectives={'a': 0.8, 'b': 0.6},
        passed_permutation_overlay=[True, True],
    )
    fr2 = FoldResult(
        fold_id='fold_1', fold_period=('2020-06-01', '2020-12-31'),
        top_k_params=['c', 'd'], smoothed_objectives={'c': 0.8, 'd': 0.6},
        passed_permutation_overlay=[False, False],
    )
    metrics = _compute_consistency_metrics([fr1, fr2], top_k=2)
    assert abs(metrics['overlap_rate'] - 0.0) < 1e-9


def test_compute_neighbor_smoothed_grid_parameter_analyzer() -> None:
    """ParameterAnalyzer.compute_neighbor_smoothed_grid() adds smoothed_objective column."""
    import pandas as pd
    from eda.parameter_analysis import ParameterAnalyzer

    param_df = pd.DataFrame({
        'lookback': [3, 5, 10, 14],
        'objective': [0.2, 0.4, 0.6, 0.3],
    })

    # Instantiate without calling __init__ since it requires complex args
    analyzer = object.__new__(ParameterAnalyzer)

    result = analyzer.compute_neighbor_smoothed_grid(param_df, ['lookback'], 'objective')

    assert 'smoothed_objective' in result.columns
    assert len(result) == len(param_df)

    # lookback=3 neighbors={5}; smoothed = mean([0.2, 0.4]) = 0.3
    row3 = result[result['lookback'] == 3].iloc[0]
    assert abs(row3['smoothed_objective'] - np.mean([0.2, 0.4])) < 1e-9

    # lookback=5 neighbors={3, 10}; smoothed = mean([0.4, 0.2, 0.6])
    row5 = result[result['lookback'] == 5].iloc[0]
    assert abs(row5['smoothed_objective'] - np.mean([0.4, 0.2, 0.6])) < 1e-9
