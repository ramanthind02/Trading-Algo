"""Unit tests for the signed-signal OOS permutation runner."""
from __future__ import annotations

from typing import Any

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
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "datetime": dates},
        index=dates,
    )


def _vector_report(*, passed: bool, param_combo: str = "p1") -> VectorShuffleReport:
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


def _pipeline_report(*, passed: bool, param_combo: str = "p1") -> PipelinePermutationReport:
    return PipelinePermutationReport(
        param_combo=param_combo,
        feature_type="signed_signal",
        permutation_mode="candle_shuffle",
        original_metric=0.5,
        null_distribution=np.array([0.1, 0.2, 0.3], dtype=float),
        critical_value=0.25,
        p_value=0.03,
        passed=passed,
        alpha=0.10,
        nreps=3,
        no_trade_permutations=0,
    )


def _signal(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().fillna(0.0).rename("signal")


def test_oos_permutation_skips_candle_when_vector_fails(monkeypatch: Any) -> None:
    called = {"pipeline": False}

    def fake_vector(*args: Any, **kwargs: Any) -> VectorShuffleReport:
        return _vector_report(passed=False)

    def fake_pipeline(*args: Any, **kwargs: Any) -> PipelinePermutationReport:
        called["pipeline"] = True
        return _pipeline_report(passed=True)

    monkeypatch.setattr(permutation_tests, "run_vector_shuffle_test", fake_vector)
    monkeypatch.setattr(permutation_tests, "run_pipeline_permutation", fake_pipeline)

    candles = _make_candles()
    target = pd.Series(np.random.default_rng(1).standard_normal(len(candles)), index=candles.index)

    report = permutation_tests.run_oos_permutation_for_param(
        param_combo="p1",
        fitted_feature=_signal(candles),
        candles_df=candles,
        target=target,
        objective_func=_sharpe,
        signal_extractor=_signal,
        random_seed=11,
        nreps=25,
        alpha=0.10,
    )

    assert isinstance(report, OutOfSamplePermutationReport)
    assert report.vector_report.passed is False
    assert report.candle_report is None
    assert report.passed is False
    assert called["pipeline"] is False


def test_oos_permutation_vector_only_when_vector_passes(monkeypatch: Any) -> None:
    calls: list[tuple[str, int | None]] = []

    def fake_vector(*args: Any, **kwargs: Any) -> VectorShuffleReport:
        calls.append(("vector", kwargs.get("random_seed")))
        return _vector_report(passed=True)

    def fake_pipeline(*args: Any, **kwargs: Any) -> PipelinePermutationReport:
        raise AssertionError("run_pipeline_permutation should not be called")

    monkeypatch.setattr(permutation_tests, "run_vector_shuffle_test", fake_vector)
    monkeypatch.setattr(permutation_tests, "run_pipeline_permutation", fake_pipeline)

    candles = _make_candles()
    target = pd.Series(np.random.default_rng(2).standard_normal(len(candles)), index=candles.index)

    report = permutation_tests.run_oos_permutation_for_param(
        param_combo="p1",
        fitted_feature=_signal(candles),
        candles_df=candles,
        target=target,
        objective_func=_sharpe,
        signal_extractor=_signal,
        random_seed=123,
        nreps=15,
        alpha=0.10,
        permutation_mode="feature_shuffle",
    )

    assert report.vector_report.passed is True
    assert report.candle_report is None
    assert report.passed is True
    assert calls == [("vector", 123)]
