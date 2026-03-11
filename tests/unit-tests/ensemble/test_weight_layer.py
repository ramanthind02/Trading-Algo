"""Unit tests for the clustered WeightLayer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ensemble.weight_layer import WeightLayer, WeightLayerConfig


def _make_forecasts(
    columns: dict[str, list[float]],
    *,
    ticker: str = "ES",
) -> list[pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=len(next(iter(columns.values()))), freq="D")
    vectors: list[pd.DataFrame] = []
    for model_name, values in columns.items():
        vectors.append(
            pd.DataFrame(
                {
                    "ticker": [ticker] * len(values),
                    "datetime": index,
                    "model_name": [model_name] * len(values),
                    "forecast": values,
                    "signal": [int(v != 0.0) for v in values],
                }
            )
        )
    return vectors


def _make_returns(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D"))


def test_weight_layer_config_accepts_only_two_modes() -> None:
    assert WeightLayerConfig(weighting_method="cluster_equal").weighting_method == "cluster_equal"
    assert (
        WeightLayerConfig(weighting_method="cluster_corr_ulcer").weighting_method
        == "cluster_corr_ulcer"
    )

    with pytest.raises(ValueError, match="weighting_method must be one of"):
        WeightLayerConfig(weighting_method="inverse_correlation")


def test_weight_layer_rejects_removed_risk_tilt_alpha_kwarg() -> None:
    with pytest.raises(ValueError, match="risk_tilt_alpha is no longer supported"):
        WeightLayer(weight_method="cluster_corr_ulcer", risk_tilt_alpha=0.5)


def test_cluster_equal_clusters_highly_correlated_members_together() -> None:
    layer = WeightLayer(rho_cut=0.7)
    forecasts = _make_forecasts(
        {
            "model_a": [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            "model_b": [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            "model_c": [0.2, 0.4, 0.2, 0.4, 0.2, 0.4],
        }
    )

    layer.fit(forecasts, signals=pd.DataFrame())
    diag = layer.get_diagnostics()
    assignments = diag["tickers"]["ES"]["cluster_assignments"]

    assert assignments["model_a"] == assignments["model_b"]
    assert assignments["model_a"] != assignments["model_c"]


def test_cluster_equal_splits_weight_equally_across_clusters_and_members() -> None:
    layer = WeightLayer(rho_cut=0.7)
    forecasts = _make_forecasts(
        {
            "fast_1": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "fast_2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "slow_1": [0.1, 0.3, 0.1, 0.3, 0.1, 0.3],
        }
    )

    layer.fit(forecasts, signals=pd.DataFrame())
    weights = layer.weights_["ES"]
    diag = layer.get_diagnostics()["tickers"]["ES"]

    assert pytest.approx(0.25, rel=1e-6) == weights["fast_1"]
    assert pytest.approx(0.25, rel=1e-6) == weights["fast_2"]
    assert pytest.approx(0.50, rel=1e-6) == weights["slow_1"]
    assert sorted(diag["cluster_weights"].values()) == pytest.approx([0.5, 0.5])


def test_cluster_corr_ulcer_does_not_require_returns() -> None:
    layer = WeightLayer(weight_method="cluster_corr_ulcer")
    forecasts = _make_forecasts(
        {
            "model_a": [1.0, 0.5, 1.0, 0.5],
            "model_b": [0.4, 0.1, 0.4, 0.1],
        }
    )

    layer.fit(forecasts, signals=pd.DataFrame(), returns=None)

    assert layer.is_fitted_ is True


def test_cluster_corr_ulcer_downweights_more_correlated_cluster() -> None:
    layer = WeightLayer(
        weight_method="cluster_corr_ulcer",
        rho_cut=0.7,
        group_weight_cap=1.0,
    )
    rng = np.random.default_rng(123)
    shared = rng.normal(size=300)
    model_a = shared + rng.normal(scale=1.0, size=300)
    model_b = shared + rng.normal(scale=1.0, size=300)
    model_c = rng.normal(size=300)
    forecasts = _make_forecasts(
        {
            "model_a": model_a.tolist(),
            "model_b": model_b.tolist(),
            "model_c": model_c.tolist(),
        }
    )

    layer.fit(forecasts, signals=pd.DataFrame())
    diag = layer.get_diagnostics()["tickers"]["ES"]
    weights = layer.weights_["ES"]
    assignments = diag["cluster_assignments"]
    cluster_weights = diag["cluster_weights"]

    corr_only_cluster = assignments["model_c"]
    corr_heavy_cluster_a = assignments["model_a"]
    corr_heavy_cluster_b = assignments["model_b"]

    assert corr_heavy_cluster_a != corr_heavy_cluster_b
    assert cluster_weights[corr_only_cluster] > cluster_weights[corr_heavy_cluster_a]
    assert cluster_weights[corr_only_cluster] > cluster_weights[corr_heavy_cluster_b]
    assert weights["model_c"] > weights["model_a"]
    assert weights["model_c"] > weights["model_b"]
    metrics = layer.get_diagnostics()["tickers"]["ES"]["cluster_metrics"]
    assert all(info["ulcer_index"] is None for info in metrics.values())


def test_cluster_corr_ulcer_ignores_returns_input() -> None:
    forecasts = _make_forecasts(
        {
            "model_a": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "model_b": [0.1, 0.3, 0.1, 0.3, 0.1, 0.3],
            "model_c": [0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
        }
    )
    series_returns = _make_returns([0.01, -0.30, 0.01, -0.30, 0.01, -0.30])
    frame_returns = pd.DataFrame(
        {"ES": series_returns, "NQ": series_returns * 0.5},
        index=series_returns.index,
    )

    no_returns = WeightLayer(weight_method="cluster_corr_ulcer", rho_cut=0.7, group_weight_cap=1.0)
    series_layer = WeightLayer(weight_method="cluster_corr_ulcer", rho_cut=0.7, group_weight_cap=1.0)
    frame_layer = WeightLayer(weight_method="cluster_corr_ulcer", rho_cut=0.7, group_weight_cap=1.0)

    no_returns.fit(forecasts, signals=pd.DataFrame(), returns=None)
    series_layer.fit(forecasts, signals=pd.DataFrame(), returns=series_returns)
    frame_layer.fit(forecasts, signals=pd.DataFrame(), returns=frame_returns)

    pd.testing.assert_series_equal(no_returns.weights_["ES"], series_layer.weights_["ES"])
    pd.testing.assert_series_equal(no_returns.weights_["ES"], frame_layer.weights_["ES"])


def test_insufficient_forecast_history_falls_back_to_one_cluster() -> None:
    layer = WeightLayer()
    forecasts = [
        pd.DataFrame(
            {
                "ticker": ["ES", "ES"],
                "datetime": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "model_name": ["model_a", "model_b"],
                "forecast": [0.2, 0.3],
                "signal": [1, 1],
            }
        )
    ]

    layer.fit(forecasts, signals=pd.DataFrame())
    diag = layer.get_diagnostics()["tickers"]["ES"]

    assert diag["cluster_weights"] == {"cluster_1": 1.0}
    assert diag["fdm"] == 1.0


def test_fdm_uses_cluster_correlations_and_respects_cap() -> None:
    rng = np.random.default_rng(42)
    model_a = rng.normal(size=80)
    model_b = rng.normal(size=80)
    layer = WeightLayer(fdm_max=1.3)
    forecasts = _make_forecasts({"model_a": model_a.tolist(), "model_b": model_b.tolist()})

    layer.fit(forecasts, signals=pd.DataFrame())

    assert layer.fdm_["ES"] <= 1.3
    assert layer.get_diagnostics()["tickers"]["ES"]["mean_cluster_correlation"] < 1.0


def test_combine_returns_datetime_level_forecasts() -> None:
    layer = WeightLayer()
    forecasts = _make_forecasts(
        {
            "model_a": [0.2, 0.4, 0.6],
            "model_b": [0.2, 0.4, 0.6],
        }
    )

    layer.fit(forecasts, signals=pd.DataFrame())
    combined = layer.combine(forecasts)

    assert list(combined.columns) == ["ticker", "datetime", "forecast_score"]
    assert len(combined) == 3
