from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from feature_selection.validation.permutation_tests import (
    _SignedSignalPermutationBatchItem,
    _run_pipeline_permutation_batch,
    run_pipeline_permutation,
)


def _make_candles(n: int = 64) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    base = 100.0 + np.linspace(0.0, 5.0, n) + 0.8 * np.sin(np.arange(n) / 3.0)
    opens = base + 0.1 * np.cos(np.arange(n) / 5.0)
    closes = opens + 0.2 * np.sin(np.arange(n) / 2.0)
    highs = np.maximum(opens, closes) + 0.4
    lows = np.minimum(opens, closes) - 0.4
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.arange(n) + 1,
        },
        index=dates,
    )


def _make_target(candles: pd.DataFrame) -> pd.Series:
    return candles["close"].pct_change().fillna(0.0).rename("target")


def _objective(returns: pd.Series) -> float:
    return float(returns.mean()) if len(returns) else 0.0


def _signal_factory(scale: float) -> Callable[[pd.DataFrame], pd.Series]:
    def extractor(df: pd.DataFrame) -> pd.Series:
        idx = pd.to_datetime(df["datetime"])
        values = (df["close"] - df["open"]).to_numpy(dtype=float) + scale * np.sin(np.arange(len(df)) / 4.0)
        return pd.Series(np.sign(values), index=idx, name=f"signal_{scale:.1f}")

    return extractor


def test_batch_matches_wrapper_for_feature_shuffle() -> None:
    candles = _make_candles()
    target = _make_target(candles)
    extractor = _signal_factory(0.5)

    wrapper = run_pipeline_permutation(
        candles_df=candles,
        signal_extractor=extractor,
        target=target,
        objective_func=_objective,
        permutation_mode="feature_shuffle",
        nreps=11,
        alpha=0.1,
        random_seed=123,
        param_combo="p0",
    )
    batch = _run_pipeline_permutation_batch(
        candles_df=candles,
        items=[_SignedSignalPermutationBatchItem(param_combo="p0", signal_extractor=extractor)],
        target=target,
        objective_func=_objective,
        permutation_mode="feature_shuffle",
        nreps=11,
        alpha=0.1,
        random_seed=123,
    )["p0"]

    assert wrapper.original_metric == batch.original_metric
    assert wrapper.no_trade_permutations == batch.no_trade_permutations
    assert wrapper.p_value == batch.p_value
    np.testing.assert_allclose(wrapper.null_distribution, batch.null_distribution)


def test_batch_matches_wrapper_for_candle_shuffle() -> None:
    candles = _make_candles()
    target = _make_target(candles)
    extractor = _signal_factory(1.0)

    wrapper = run_pipeline_permutation(
        candles_df=candles,
        signal_extractor=extractor,
        target=target,
        objective_func=_objective,
        permutation_mode="candle_shuffle",
        nreps=9,
        alpha=0.1,
        random_seed=7,
        param_combo="p1",
    )
    batch = _run_pipeline_permutation_batch(
        candles_df=candles,
        items=[_SignedSignalPermutationBatchItem(param_combo="p1", signal_extractor=extractor)],
        target=target,
        objective_func=_objective,
        permutation_mode="candle_shuffle",
        nreps=9,
        alpha=0.1,
        random_seed=7,
    )["p1"]

    assert wrapper.original_metric == batch.original_metric
    assert wrapper.no_trade_permutations == batch.no_trade_permutations
    assert wrapper.p_value == batch.p_value
    np.testing.assert_allclose(wrapper.null_distribution, batch.null_distribution)


def test_batch_multi_combo_returns_expected_shapes() -> None:
    candles = _make_candles()
    target = _make_target(candles)
    reports = _run_pipeline_permutation_batch(
        candles_df=candles,
        items=[
            _SignedSignalPermutationBatchItem(param_combo="a", signal_extractor=_signal_factory(0.3)),
            _SignedSignalPermutationBatchItem(param_combo="b", signal_extractor=_signal_factory(0.9)),
        ],
        target=target,
        objective_func=_objective,
        permutation_mode="candle_shuffle",
        nreps=7,
        alpha=0.1,
        random_seed=5,
    )

    assert set(reports) == {"a", "b"}
    for report in reports.values():
        assert report.feature_type == "signed_signal"
        assert report.permutation_mode == "candle_shuffle"
        assert len(report.null_distribution) == 7
        assert 0.0 <= report.p_value <= 1.0


def test_batch_n_jobs_reps_parity() -> None:
    candles = _make_candles()
    target = _make_target(candles)
    items = [
        _SignedSignalPermutationBatchItem(param_combo="a", signal_extractor=_signal_factory(0.3)),
        _SignedSignalPermutationBatchItem(param_combo="b", signal_extractor=_signal_factory(0.9)),
    ]
    common = dict(
        candles_df=candles,
        items=items,
        target=target,
        objective_func=_objective,
        permutation_mode="candle_shuffle",
        nreps=11,
        alpha=0.1,
        random_seed=42,
    )

    reports_seq = _run_pipeline_permutation_batch(**common, n_jobs_reps=1)
    reports_par = _run_pipeline_permutation_batch(**common, n_jobs_reps=2)

    assert set(reports_seq) == set(reports_par)
    for key in reports_seq:
        assert reports_seq[key].p_value == reports_par[key].p_value
        assert reports_seq[key].passed == reports_par[key].passed
