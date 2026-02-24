from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validation.permutation_tests import (
    _ContinuousPermutationBatchItem,
    _RuleBasedPermutationBatchItem,
    _run_pipeline_permutation_continuous_batch,
    _run_pipeline_permutation_rule_based_batch,
    run_pipeline_permutation_continuous,
    run_pipeline_permutation_rule_based,
)


class DummyBinningModel(BinningModelBase):
    model_type = "continuous_binning"

    def __init__(self) -> None:
        super().__init__(
            n_bins=2,
            normalize_by=None,
            metric_threshold=-1e9,
            t_threshold=-1e9,
            min_region_width=1,
        )

    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        _ = target_data
        threshold = float(feature_data.median())
        bins = np.where(feature_data.to_numpy(dtype=float) >= threshold, 2, 1)
        return pd.Series(bins, index=feature_data.index, dtype=int)


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
    returns = candles["close"].pct_change().fillna(0.0)
    return returns.rename("target")


def _objective(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float(returns.mean())


def _continuous_extractor_factory(scale: float) -> Callable[[pd.DataFrame], pd.Series]:
    def extractor(df: pd.DataFrame) -> pd.Series:
        dt_index = pd.to_datetime(df["datetime"])
        values = (
            (df["close"] - df["open"]).to_numpy(dtype=float)
            + scale * np.sin(np.arange(len(df)) / 4.0)
        )
        return pd.Series(values, index=dt_index, name=f"feature_{scale:.1f}")

    return extractor


def _rule_extractor_factory(period: int) -> Callable[[pd.DataFrame], pd.Series]:
    def extractor(df: pd.DataFrame) -> pd.Series:
        dt_index = pd.to_datetime(df["datetime"])
        close = pd.Series(df["close"].to_numpy(dtype=float), index=dt_index)
        signal = np.sign(close.diff(period).fillna(0.0))
        return signal.astype(float).rename(f"rule_{period}")

    return extractor


def test_continuous_batch_matches_wrapper_for_feature_shuffle() -> None:
    candles = _make_candles()
    target = _make_target(candles)
    extractor = _continuous_extractor_factory(0.5)
    model = DummyBinningModel()

    wrapper = run_pipeline_permutation_continuous(
        candles_df=candles,
        bias_node_extractor=extractor,
        binning_model=model,
        target=target,
        objective_func=_objective,
        permutation_mode="feature_shuffle",
        nreps=11,
        alpha=0.1,
        random_seed=123,
        param_combo="p0",
    )
    batch = _run_pipeline_permutation_continuous_batch(
        candles_df=candles,
        items=[
            _ContinuousPermutationBatchItem(
                param_combo="p0",
                bias_node_extractor=extractor,
                binning_model=model,
            )
        ],
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


def test_continuous_batch_matches_wrapper_for_candle_shuffle() -> None:
    candles = _make_candles()
    target = _make_target(candles)
    extractor = _continuous_extractor_factory(1.0)
    model = DummyBinningModel()

    wrapper = run_pipeline_permutation_continuous(
        candles_df=candles,
        bias_node_extractor=extractor,
        binning_model=model,
        target=target,
        objective_func=_objective,
        permutation_mode="candle_shuffle",
        nreps=9,
        alpha=0.1,
        random_seed=7,
        param_combo="p1",
    )
    batch = _run_pipeline_permutation_continuous_batch(
        candles_df=candles,
        items=[
            _ContinuousPermutationBatchItem(
                param_combo="p1",
                bias_node_extractor=extractor,
                binning_model=model,
            )
        ],
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


def test_rule_based_batch_matches_wrapper_for_candle_shuffle() -> None:
    candles = _make_candles()
    target = _make_target(candles)
    extractor = _rule_extractor_factory(2)

    wrapper = run_pipeline_permutation_rule_based(
        candles_df=candles,
        rule_extractor=extractor,
        target=target,
        objective_func=_objective,
        nreps=9,
        alpha=0.1,
        random_seed=99,
        param_combo="rb2",
    )
    batch = _run_pipeline_permutation_rule_based_batch(
        candles_df=candles,
        items=[_RuleBasedPermutationBatchItem(param_combo="rb2", rule_extractor=extractor)],
        target=target,
        objective_func=_objective,
        nreps=9,
        alpha=0.1,
        random_seed=99,
    )["rb2"]

    assert wrapper.original_metric == batch.original_metric
    assert wrapper.no_trade_permutations == batch.no_trade_permutations
    assert wrapper.p_value == batch.p_value
    np.testing.assert_allclose(wrapper.null_distribution, batch.null_distribution)


def test_continuous_batch_multi_combo_returns_expected_shapes() -> None:
    candles = _make_candles()
    target = _make_target(candles)
    reports = _run_pipeline_permutation_continuous_batch(
        candles_df=candles,
        items=[
            _ContinuousPermutationBatchItem(
                param_combo="a",
                bias_node_extractor=_continuous_extractor_factory(0.3),
                binning_model=DummyBinningModel(),
            ),
            _ContinuousPermutationBatchItem(
                param_combo="b",
                bias_node_extractor=_continuous_extractor_factory(0.9),
                binning_model=DummyBinningModel(),
            ),
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
        assert report.feature_type == "continuous"
        assert report.permutation_mode == "candle_shuffle"
        assert len(report.null_distribution) == 7
        assert 0.0 <= report.p_value <= 1.0
