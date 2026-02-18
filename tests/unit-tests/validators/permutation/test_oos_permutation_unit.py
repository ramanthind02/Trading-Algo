"""Unit tests for Task 4: OOS permutation runner."""
from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytest

import feature_selection.validation.permutation_tests as permutation_tests
from feature_selection.validation.reports import (
    OutOfSamplePermutationReport,
    PipelinePermutationReport,
    VectorShuffleReport,
)


def _sharpe(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std())


def _make_candles(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 + rng.standard_normal(n).cumsum()
    opens = closes - rng.uniform(0.0, 0.5, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.0, 0.5, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.0, 0.5, n)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    return pd.DataFrame(
        {'open': opens, 'high': highs, 'low': lows, 'close': closes, 'datetime': dates},
        index=dates,
    )


def _make_vector_report(*, passed: bool, param_combo: str = 'p1') -> VectorShuffleReport:
    return VectorShuffleReport(
        param_combo=param_combo,
        original_metric=0.4,
        null_distribution=np.array([0.1, 0.2, 0.3], dtype=float),
        critical_value=0.25,
        p_value=0.05,
        passed=passed,
        alpha=0.10,
        nreps=3,
    )


def _make_pipeline_report(
    *,
    passed: bool,
    feature_type: Literal['continuous', 'rule_based'],
    param_combo: str = 'p1',
) -> PipelinePermutationReport:
    return PipelinePermutationReport(
        param_combo=param_combo,
        feature_type=feature_type,
        permutation_mode='candle_shuffle',
        original_metric=0.5,
        null_distribution=np.array([0.1, 0.2, 0.3], dtype=float),
        critical_value=0.25,
        p_value=0.03,
        passed=passed,
        alpha=0.10,
        nreps=3,
        no_trade_permutations=0,
    )


def test_oos_permutation_skips_candle_when_vector_fails(monkeypatch: Any) -> None:
    """Vector-first gate skips candle permutation on vector failure."""
    called = {'continuous': False}

    def fake_vector(*args: Any, **kwargs: Any) -> VectorShuffleReport:
        return _make_vector_report(passed=False)

    def fake_continuous(*args: Any, **kwargs: Any) -> PipelinePermutationReport:
        called['continuous'] = True
        return _make_pipeline_report(passed=True, feature_type='continuous')

    monkeypatch.setattr(permutation_tests, 'run_vector_shuffle_test', fake_vector)
    monkeypatch.setattr(
        permutation_tests,
        'run_pipeline_permutation_continuous',
        fake_continuous,
    )

    candles = _make_candles()
    target = pd.Series(np.random.default_rng(1).standard_normal(len(candles)), index=candles.index)
    fitted_feature = candles['close'].pct_change().fillna(0.0)

    report = permutation_tests.run_oos_permutation_for_param(
        param_combo='p1',
        feature_type='continuous',
        fitted_feature=fitted_feature,
        candles_df=candles,
        target=target,
        objective_func=_sharpe,
        bias_node_extractor=None,
        binning_model=None,
        random_seed=11,
        nreps=25,
        alpha=0.10,
    )

    assert isinstance(report, OutOfSamplePermutationReport)
    assert report.vector_report.passed is False
    assert report.candle_report is None
    assert report.passed is False
    assert called['continuous'] is False


def test_oos_permutation_routes_continuous_when_vector_passes(monkeypatch: Any) -> None:
    """Continuous OOS path runs continuous pipeline when vector passes."""
    calls: list[tuple[str, int | None]] = []
    forwarded: dict[str, Any] = {}

    def fake_vector(*args: Any, **kwargs: Any) -> VectorShuffleReport:
        calls.append(('vector', kwargs.get('random_seed')))
        return _make_vector_report(passed=True)

    def fake_continuous(*args: Any, **kwargs: Any) -> PipelinePermutationReport:
        calls.append(('continuous', kwargs.get('random_seed')))
        forwarded.update(kwargs)
        return _make_pipeline_report(passed=True, feature_type='continuous')

    monkeypatch.setattr(permutation_tests, 'run_vector_shuffle_test', fake_vector)
    monkeypatch.setattr(
        permutation_tests,
        'run_pipeline_permutation_continuous',
        fake_continuous,
    )

    candles = _make_candles()
    target = pd.Series(np.random.default_rng(2).standard_normal(len(candles)), index=candles.index)
    fitted_feature = candles['close'].pct_change().fillna(0.0)

    report = permutation_tests.run_oos_permutation_for_param(
        param_combo='p1',
        feature_type='continuous',
        fitted_feature=fitted_feature,
        candles_df=candles,
        target=target,
        objective_func=_sharpe,
        bias_node_extractor=lambda df: df['close'].pct_change().fillna(0.0),
        binning_model=cast(Any, object()),
        random_seed=123,
        nreps=15,
        alpha=0.10,
        permutation_mode='feature_shuffle',
    )

    assert report.vector_report.passed is True
    assert report.candle_report is not None
    assert report.candle_report.feature_type == 'continuous'
    assert report.passed is True
    assert calls == [('vector', 123), ('continuous', 123)]
    assert forwarded['param_combo'] == 'p1'
    assert forwarded['permutation_mode'] == 'feature_shuffle'
    assert forwarded['nreps'] == 15
    assert forwarded['alpha'] == 0.10
    assert forwarded['metric_threshold'] == 0.0


def test_oos_permutation_routes_rule_based_when_vector_passes(monkeypatch: Any) -> None:
    """Rule-based OOS path runs rule-based candle permutation after vector pass."""
    called = {'rule_based': False}

    def fake_vector(*args: Any, **kwargs: Any) -> VectorShuffleReport:
        return _make_vector_report(passed=True)

    def fake_rule_based(*args: Any, **kwargs: Any) -> PipelinePermutationReport:
        called['rule_based'] = True
        return _make_pipeline_report(passed=False, feature_type='rule_based')

    monkeypatch.setattr(permutation_tests, 'run_vector_shuffle_test', fake_vector)
    monkeypatch.setattr(
        permutation_tests,
        'run_pipeline_permutation_rule_based',
        fake_rule_based,
    )

    candles = _make_candles()
    target = pd.Series(np.random.default_rng(3).standard_normal(len(candles)), index=candles.index)
    fitted_feature = candles['close'].pct_change().fillna(0.0)

    report = permutation_tests.run_oos_permutation_for_param(
        param_combo='p1',
        feature_type='rule_based',
        fitted_feature=fitted_feature,
        candles_df=candles,
        target=target,
        objective_func=_sharpe,
        rule_extractor=lambda df: df['close'].pct_change().fillna(0.0).apply(np.sign),
        random_seed=88,
        nreps=10,
        alpha=0.10,
    )

    assert called['rule_based'] is True
    assert report.candle_report is not None
    assert report.candle_report.feature_type == 'rule_based'
    assert report.passed is False


def test_oos_permutation_raises_for_invalid_feature_type(monkeypatch: Any) -> None:
    """Invalid feature_type fails fast with ValueError."""

    def fake_vector(*args: Any, **kwargs: Any) -> VectorShuffleReport:
        return _make_vector_report(passed=True)

    monkeypatch.setattr(permutation_tests, 'run_vector_shuffle_test', fake_vector)

    candles = _make_candles()
    target = pd.Series(np.random.default_rng(4).standard_normal(len(candles)), index=candles.index)
    fitted_feature = candles['close'].pct_change().fillna(0.0)

    with pytest.raises(ValueError, match='Unknown feature_type'):
        permutation_tests.run_oos_permutation_for_param(
            param_combo='p1',
            feature_type=cast(Any, 'invalid_feature_type'),
            fitted_feature=fitted_feature,
            candles_df=candles,
            target=target,
            objective_func=_sharpe,
            random_seed=99,
            nreps=10,
            alpha=0.10,
        )


def test_oos_permutation_raises_for_invalid_feature_type_when_vector_would_fail(
    monkeypatch: Any,
) -> None:
    """Invalid feature_type raises before vector-stage fail path."""
    called = {'vector': False}

    def fake_vector(*args: Any, **kwargs: Any) -> VectorShuffleReport:
        called['vector'] = True
        return _make_vector_report(passed=False)

    monkeypatch.setattr(permutation_tests, 'run_vector_shuffle_test', fake_vector)

    candles = _make_candles()
    target = pd.Series(np.random.default_rng(6).standard_normal(len(candles)), index=candles.index)
    fitted_feature = candles['close'].pct_change().fillna(0.0)

    with pytest.raises(ValueError, match='Unknown feature_type'):
        permutation_tests.run_oos_permutation_for_param(
            param_combo='p1',
            feature_type=cast(Any, 'invalid_feature_type'),
            fitted_feature=fitted_feature,
            candles_df=candles,
            target=target,
            objective_func=_sharpe,
            random_seed=101,
            nreps=10,
            alpha=0.10,
        )

    assert called['vector'] is False


def test_oos_permutation_raises_for_invalid_rule_based_permutation_mode(
    monkeypatch: Any,
) -> None:
    """Rule-based OOS mode validation rejects non-candle shuffle mode."""

    def fake_vector(*args: Any, **kwargs: Any) -> VectorShuffleReport:
        return _make_vector_report(passed=True)

    monkeypatch.setattr(permutation_tests, 'run_vector_shuffle_test', fake_vector)

    candles = _make_candles()
    target = pd.Series(np.random.default_rng(5).standard_normal(len(candles)), index=candles.index)
    fitted_feature = candles['close'].pct_change().fillna(0.0)

    with pytest.raises(ValueError, match='rule_based.*candle_shuffle'):
        permutation_tests.run_oos_permutation_for_param(
            param_combo='p1',
            feature_type='rule_based',
            fitted_feature=fitted_feature,
            candles_df=candles,
            target=target,
            objective_func=_sharpe,
            rule_extractor=lambda df: df['close'].pct_change().fillna(0.0).apply(np.sign),
            permutation_mode='feature_shuffle',
            random_seed=100,
            nreps=10,
            alpha=0.10,
        )
