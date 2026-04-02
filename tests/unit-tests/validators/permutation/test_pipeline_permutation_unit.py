"""Unit tests for the signed-signal pipeline permutation test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection.validation.permutation_tests import run_pipeline_permutation
from feature_selection.validation.reports import PipelinePermutationReport


def _sharpe(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std())


def _make_candles(n: int = 200, seed: int = 0) -> pd.DataFrame:
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


def _signed_signal(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().fillna(0.0).apply(np.sign).rename("signal")


def test_pipeline_permutation_report_fields() -> None:
    report = PipelinePermutationReport(
        param_combo="test",
        feature_type="signed_signal",
        permutation_mode="feature_shuffle",
        original_metric=0.5,
        null_distribution=np.array([0.1] * 50),
        critical_value=0.4,
        p_value=0.1,
        passed=True,
        alpha=0.10,
        nreps=50,
        no_trade_permutations=2,
    )
    assert report.feature_type == "signed_signal"
    assert report.permutation_mode == "feature_shuffle"


def test_feature_shuffle_is_deterministic() -> None:
    candles = _make_candles(120)
    target = pd.Series(np.random.default_rng(1).standard_normal(120), index=candles.index)

    r1 = run_pipeline_permutation(
        candles_df=candles,
        signal_extractor=_signed_signal,
        target=target,
        objective_func=_sharpe,
        permutation_mode="feature_shuffle",
        nreps=20,
        random_seed=42,
        param_combo="p0",
    )
    r2 = run_pipeline_permutation(
        candles_df=candles,
        signal_extractor=_signed_signal,
        target=target,
        objective_func=_sharpe,
        permutation_mode="feature_shuffle",
        nreps=20,
        random_seed=42,
        param_combo="p0",
    )
    np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)
    assert r1.p_value == r2.p_value


def test_candle_shuffle_is_deterministic() -> None:
    candles = _make_candles(100)
    target = pd.Series(np.random.default_rng(2).standard_normal(100), index=candles.index)

    r1 = run_pipeline_permutation(
        candles_df=candles,
        signal_extractor=_signed_signal,
        target=target,
        objective_func=_sharpe,
        permutation_mode="candle_shuffle",
        nreps=15,
        random_seed=77,
        param_combo="p1",
    )
    r2 = run_pipeline_permutation(
        candles_df=candles,
        signal_extractor=_signed_signal,
        target=target,
        objective_func=_sharpe,
        permutation_mode="candle_shuffle",
        nreps=15,
        random_seed=77,
        param_combo="p1",
    )
    np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)
    assert r1.p_value == r2.p_value


def test_invalid_permutation_mode_raises() -> None:
    candles = _make_candles(80)
    target = pd.Series(np.random.default_rng(3).standard_normal(80), index=candles.index)

    with pytest.raises(ValueError, match="Unknown permutation_mode"):
        run_pipeline_permutation(
            candles_df=candles,
            signal_extractor=_signed_signal,
            target=target,
            objective_func=_sharpe,
            permutation_mode="bad_mode",  # type: ignore[arg-type]
            nreps=5,
            random_seed=0,
        )
