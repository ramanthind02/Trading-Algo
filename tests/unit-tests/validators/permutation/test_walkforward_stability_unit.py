"""Unit tests for the signed-signal walkforward stability analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_selection.validation.reports import FoldResult, WalkforwardStabilityReport
from feature_selection.validation.stability_analysis import (
    _compute_consistency_metrics,
    _compute_neighbor_smoothed_objectives,
    _param_combo_name,
    run_walkforward_stability,
)


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
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "datetime": dates},
        index=dates,
    )


def test_neighbor_smoothing_1d() -> None:
    param_grid = [{"lookback": v} for v in [3, 5, 10, 14]]
    raw = {"lookback_3": 0.2, "lookback_5": 0.4, "lookback_10": 0.6, "lookback_14": 0.3}
    smoothed = _compute_neighbor_smoothed_objectives(["lookback"], param_grid, raw)
    assert abs(smoothed["lookback_3"] - np.mean([0.2, 0.4])) < 1e-9
    assert abs(smoothed["lookback_5"] - np.mean([0.4, 0.2, 0.6])) < 1e-9
    assert abs(smoothed["lookback_10"] - np.mean([0.6, 0.4, 0.3])) < 1e-9
    assert abs(smoothed["lookback_14"] - np.mean([0.3, 0.6])) < 1e-9


def test_walkforward_stability_report_fields() -> None:
    report = WalkforwardStabilityReport(
        feature_name="rsi",
        feature_type="signed_signal",
        fold_results=[],
        consistency_metrics={"overlap_rate": 0.8},
        is_stable=True,
        stability_verdict="STABLE",
        top_k=3,
    )
    assert report.feature_type == "signed_signal"
    assert report.is_stable is True


def test_walkforward_stability_uses_signed_signals() -> None:
    candles = _make_candles(100)
    target = pd.Series(np.random.default_rng(0).standard_normal(100), index=candles.index)
    param_grid = [{"lookback": v} for v in [3, 5, 10]]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        lb = params["lookback"]
        return df["close"].pct_change(lb).fillna(0.0).apply(np.sign).rename(f"lookback_{lb}")

    folds = [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")),
        (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-07-01")),
        (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-10-01")),
    ]

    report = run_walkforward_stability(
        candles_df=candles,
        extractor_func=extractor,
        target=target,
        objective_func=_sharpe,
        param_grid=param_grid,
        fold_structure=folds,
        top_k=1,
        feature_name="test",
    )

    assert isinstance(report, WalkforwardStabilityReport)
    assert report.feature_type == "signed_signal"
    assert len(report.fold_results) >= 1
    assert "overlap_rate" in report.consistency_metrics


def test_permutation_overlay_flags() -> None:
    candles = _make_candles(80)
    target = pd.Series(np.random.default_rng(1).standard_normal(80), index=candles.index)
    param_grid = [{"lookback": v} for v in [3, 5, 10]]

    def extractor(df: pd.DataFrame, params: dict) -> pd.Series:
        return df["close"].pct_change(params["lookback"]).fillna(0.0).apply(np.sign).rename(
            f"lookback_{params['lookback']}"
        )

    passers = {"lookback_3", "lookback_10"}
    report = run_walkforward_stability(
        candles_df=candles,
        extractor_func=extractor,
        target=target,
        objective_func=_sharpe,
        param_grid=param_grid,
        fold_structure=[(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"))],
        top_k=2,
        permutation_passers=passers,
    )

    fold = report.fold_results[0]
    for param_name, flag in zip(fold.top_k_params, fold.passed_permutation_overlay):
        assert flag == (param_name in passers)


def test_consistency_metrics_two_folds_identical_top_k() -> None:
    fr1 = FoldResult(
        fold_id="fold_0",
        fold_period=("2020-01-01", "2020-06-01"),
        top_k_params=["a", "b"],
        smoothed_objectives={"a": 0.8, "b": 0.6},
        passed_permutation_overlay=[True, True],
    )
    fr2 = FoldResult(
        fold_id="fold_1",
        fold_period=("2020-06-01", "2020-12-31"),
        top_k_params=["a", "b"],
        smoothed_objectives={"a": 0.7, "b": 0.5},
        passed_permutation_overlay=[True, True],
    )
    metrics = _compute_consistency_metrics([fr1, fr2], top_k=2)
    assert abs(metrics["overlap_rate"] - 1.0) < 1e-9
