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


def test_cluster_corr_ulcer_requires_returns() -> None:
    layer = WeightLayer(weight_method="cluster_corr_ulcer")
    forecasts = _make_forecasts(
        {
            "model_a": [1.0, 0.5, 1.0, 0.5],
            "model_b": [0.4, 0.1, 0.4, 0.1],
        }
    )

    with pytest.raises(ValueError, match="requires returns"):
        layer.fit(forecasts, signals=pd.DataFrame(), returns=None)


def test_cluster_corr_ulcer_downweights_worse_ulcer_cluster() -> None:
    layer = WeightLayer(weight_method="cluster_corr_ulcer", rho_cut=0.7, risk_tilt_alpha=1.0)
    forecasts = _make_forecasts(
        {
            "stable_a": [0.10, 0.10, 0.20, 0.20, 0.10, 0.10],
            "stable_b": [0.10, 0.10, 0.20, 0.20, 0.10, 0.10],
            "choppy": [0.80, 0.10, 0.80, 0.10, 0.80, 0.10],
        }
    )
    returns = _make_returns([0.01, 0.01, 0.01, -0.30, 0.01, -0.30])

    layer.fit(forecasts, signals=pd.DataFrame(), returns=returns)
    weights = layer.weights_["ES"]

    assert weights["choppy"] < weights["stable_a"] + weights["stable_b"]
    metrics = layer.get_diagnostics()["tickers"]["ES"]["cluster_metrics"]
    ulcer_values = {cluster: info["ulcer_index"] for cluster, info in metrics.items()}
    assert max(v for v in ulcer_values.values() if v is not None) > min(
        v for v in ulcer_values.values() if v is not None
    )


def test_risk_tilt_alpha_zero_removes_ulcer_penalty() -> None:
    forecasts = _make_forecasts(
        {
            "stable_a": [0.10, 0.10, 0.20, 0.20, 0.10, 0.10],
            "stable_b": [0.10, 0.10, 0.20, 0.20, 0.10, 0.10],
            "volatile_a": [0.80, 0.10, 0.80, 0.10, 0.80, 0.10],
            "volatile_b": [0.80, 0.10, 0.80, 0.10, 0.80, 0.10],
        }
    )
    returns = _make_returns([0.01, -0.30, 0.01, -0.30, 0.01, -0.30])

    alpha_zero = WeightLayer(weight_method="cluster_corr_ulcer", risk_tilt_alpha=0.0, rho_cut=0.7)
    alpha_one = WeightLayer(weight_method="cluster_corr_ulcer", risk_tilt_alpha=1.0, rho_cut=0.7)

    alpha_zero.fit(forecasts, signals=pd.DataFrame(), returns=returns)
    alpha_one.fit(forecasts, signals=pd.DataFrame(), returns=returns)

    zero_diag = alpha_zero.get_diagnostics()["tickers"]["ES"]
    one_diag = alpha_one.get_diagnostics()["tickers"]["ES"]

    zero_cluster_weights = zero_diag["cluster_weights"]
    one_cluster_weights = one_diag["cluster_weights"]

    assert sorted(zero_cluster_weights.values()) == pytest.approx([0.5, 0.5], rel=1e-6)
    assert sorted(one_cluster_weights.values()) != pytest.approx([0.5, 0.5], rel=1e-3)


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
