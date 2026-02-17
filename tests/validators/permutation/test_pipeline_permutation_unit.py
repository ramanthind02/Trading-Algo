"""Unit tests for T014: Pipeline Permutation Test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection.validation.permutation_tests import (
    run_pipeline_permutation_continuous,
    run_pipeline_permutation_rule_based,
)
from feature_selection.validation.reports import PipelinePermutationReport


def _sharpe(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std())


def _make_candles(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily OHLCV with DatetimeIndex."""
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


def _simple_extractor(df: pd.DataFrame) -> pd.Series:
    """Simple feature: close pct_change."""
    return df['close'].pct_change().fillna(0.0).rename('feat')


def _rule_extractor(df: pd.DataFrame) -> pd.Series:
    """Simple rule: sign of close pct_change."""
    ret = df['close'].pct_change().fillna(0.0)
    return ret.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).rename('rule')


def test_pipeline_permutation_report_fields() -> None:
    """PipelinePermutationReport contains all required fields."""
    report = PipelinePermutationReport(
        param_combo='test',
        feature_type='continuous',
        permutation_mode='feature_shuffle',
        original_metric=0.5,
        null_distribution=np.array([0.1] * 50),
        critical_value=0.4,
        p_value=0.1,
        passed=True,
        alpha=0.10,
        nreps=50,
        no_trade_permutations=2,
    )
    assert hasattr(report, 'param_combo')
    assert hasattr(report, 'feature_type')
    assert hasattr(report, 'permutation_mode')
    assert hasattr(report, 'original_metric')
    assert hasattr(report, 'null_distribution')
    assert hasattr(report, 'critical_value')
    assert hasattr(report, 'p_value')
    assert hasattr(report, 'passed')
    assert hasattr(report, 'alpha')
    assert hasattr(report, 'nreps')
    assert hasattr(report, 'no_trade_permutations')


def test_deterministic_feature_shuffle() -> None:
    """Same random_seed → identical null distribution for feature_shuffle."""
    from feature_selection.base_models.continuous_binning import ContinuousBinningModel

    candles = _make_candles(120)
    target = pd.Series(
        np.random.default_rng(1).standard_normal(120), index=candles.index
    )
    model = ContinuousBinningModel(n_bins=5)

    r1 = run_pipeline_permutation_continuous(
        candles, _simple_extractor, model, target, _sharpe,
        permutation_mode='feature_shuffle', nreps=20, random_seed=42,
    )
    r2 = run_pipeline_permutation_continuous(
        candles, _simple_extractor, model, target, _sharpe,
        permutation_mode='feature_shuffle', nreps=20, random_seed=42,
    )
    np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)
    assert r1.p_value == r2.p_value


def test_feature_type_continuous() -> None:
    """run_pipeline_permutation_continuous always returns feature_type='continuous'."""
    from feature_selection.base_models.continuous_binning import ContinuousBinningModel

    candles = _make_candles(100)
    target = pd.Series(np.random.default_rng(2).standard_normal(100), index=candles.index)
    model = ContinuousBinningModel(n_bins=5)

    report = run_pipeline_permutation_continuous(
        candles, _simple_extractor, model, target, _sharpe,
        permutation_mode='feature_shuffle', nreps=10, random_seed=0,
    )
    assert report.feature_type == 'continuous'
    assert report.permutation_mode == 'feature_shuffle'


def test_rule_based_uses_candle_shuffle_only() -> None:
    """run_pipeline_permutation_rule_based always sets permutation_mode='candle_shuffle'."""
    candles = _make_candles(100)
    target = pd.Series(np.random.default_rng(3).standard_normal(100), index=candles.index)

    report = run_pipeline_permutation_rule_based(
        candles, _rule_extractor, target, _sharpe, nreps=10, random_seed=0,
    )
    assert isinstance(report, PipelinePermutationReport)
    assert report.feature_type == 'rule_based'
    assert report.permutation_mode == 'candle_shuffle'


def test_invalid_permutation_mode_raises() -> None:
    """Unknown permutation_mode raises ValueError."""
    from feature_selection.base_models.continuous_binning import ContinuousBinningModel

    candles = _make_candles(80)
    target = pd.Series(np.random.default_rng(4).standard_normal(80), index=candles.index)
    model = ContinuousBinningModel(n_bins=5)

    with pytest.raises(ValueError, match='Unknown permutation_mode'):
        run_pipeline_permutation_continuous(
            candles, _simple_extractor, model, target, _sharpe,
            permutation_mode='bad_mode',  # type: ignore[arg-type]
            nreps=5, random_seed=0,
        )


def test_null_distribution_length() -> None:
    """null_distribution has exactly nreps entries."""
    from feature_selection.base_models.continuous_binning import ContinuousBinningModel

    candles = _make_candles(100)
    target = pd.Series(np.random.default_rng(5).standard_normal(100), index=candles.index)
    model = ContinuousBinningModel(n_bins=5)

    for nreps in (15, 30):
        report = run_pipeline_permutation_continuous(
            candles, _simple_extractor, model, target, _sharpe,
            permutation_mode='feature_shuffle', nreps=nreps, random_seed=0,
        )
        assert len(report.null_distribution) == nreps
        assert report.nreps == nreps


def test_p_value_in_unit_interval_continuous() -> None:
    """p_value ∈ [0, 1] for continuous feature shuffle."""
    from feature_selection.base_models.continuous_binning import ContinuousBinningModel

    candles = _make_candles(120)
    target = pd.Series(np.random.default_rng(6).standard_normal(120), index=candles.index)
    model = ContinuousBinningModel(n_bins=5)

    report = run_pipeline_permutation_continuous(
        candles, _simple_extractor, model, target, _sharpe,
        permutation_mode='feature_shuffle', nreps=25, random_seed=0,
    )
    assert 0.0 <= report.p_value <= 1.0


def test_p_value_in_unit_interval_rule_based() -> None:
    """p_value ∈ [0, 1] for rule-based candle shuffle."""
    candles = _make_candles(100)
    target = pd.Series(np.random.default_rng(7).standard_normal(100), index=candles.index)

    report = run_pipeline_permutation_rule_based(
        candles, _rule_extractor, target, _sharpe, nreps=20, random_seed=0,
    )
    assert 0.0 <= report.p_value <= 1.0


def test_no_trade_permutations_non_negative() -> None:
    """no_trade_permutations is always >= 0."""
    from feature_selection.base_models.continuous_binning import ContinuousBinningModel

    candles = _make_candles(120)
    target = pd.Series(np.random.default_rng(8).standard_normal(120), index=candles.index)
    model = ContinuousBinningModel(n_bins=5)

    report = run_pipeline_permutation_continuous(
        candles, _simple_extractor, model, target, _sharpe,
        permutation_mode='feature_shuffle', nreps=20, random_seed=0,
    )
    assert report.no_trade_permutations >= 0
    assert report.no_trade_permutations <= report.nreps


def test_candle_shuffle_deterministic_rule_based() -> None:
    """Same seed → identical rule-based candle_shuffle null distribution."""
    candles = _make_candles(100)
    target = pd.Series(np.random.default_rng(9).standard_normal(100), index=candles.index)

    r1 = run_pipeline_permutation_rule_based(
        candles, _rule_extractor, target, _sharpe, nreps=15, random_seed=77,
    )
    r2 = run_pipeline_permutation_rule_based(
        candles, _rule_extractor, target, _sharpe, nreps=15, random_seed=77,
    )
    np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)
    assert r1.p_value == r2.p_value
