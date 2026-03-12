"""Unit tests for the four-mode WeightLayer."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ensemble.weight_layer import WeightLayer, WeightLayerConfig


def _make_forecasts(
    columns: dict[str, list[float]],
    *,
    ticker: str = "ES",
) -> list[pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=len(next(iter(columns.values()))), freq="D")
    return [
        pd.DataFrame(
            {
                "ticker": [ticker] * len(values),
                "datetime": index,
                "model_name": [model_name] * len(values),
                "forecast": values,
                "signal": [int(value != 0.0) for value in values],
            }
        )
        for model_name, values in columns.items()
    ]


def _make_returns(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D"))


def test_weight_layer_config_accepts_new_modes_and_rejects_legacy_ones() -> None:
    for method in (
        "equal_signal",
        "inverse_avg_pairwise_corr",
        "hrp_cluster_equal",
        "hrp_classic",
    ):
        assert WeightLayerConfig(weighting_method=method).weighting_method == method

    with pytest.raises(ValueError, match="weighting_method must be one of"):
        WeightLayerConfig(weighting_method="cluster_equal")


def test_weight_layer_rejects_removed_risk_tilt_alpha_kwarg() -> None:
    with pytest.raises(ValueError, match="risk_tilt_alpha is no longer supported"):
        WeightLayer(weight_method="equal_signal", risk_tilt_alpha=0.5)


def test_equal_signal_weights_all_models_equally() -> None:
    layer = WeightLayer(weight_method="equal_signal")
    forecasts = _make_forecasts(
        {
            "model_a": [1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            "model_b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "model_c": [0.2, 0.4, 0.2, 0.4, 0.2, 0.4],
        }
    )

    layer.fit(forecasts, signals=pd.DataFrame())

    weights = layer.weights_["ES"]
    diag = layer.get_diagnostics()["tickers"]["ES"]

    assert weights.to_dict() == pytest.approx(
        {"model_a": 1.0 / 3.0, "model_b": 1.0 / 3.0, "model_c": 1.0 / 3.0}
    )
    assert set(diag["cluster_weights"]) == {"cluster_1", "cluster_2", "cluster_3"}
    assert diag["mean_signal_correlation"] == pytest.approx(diag["mean_cluster_correlation"])


def test_inverse_avg_pairwise_corr_overweights_least_correlated_signal() -> None:
    layer = WeightLayer(weight_method="inverse_avg_pairwise_corr")
    rng = np.random.default_rng(123)
    shared = rng.normal(size=300)
    forecasts = _make_forecasts(
        {
            "model_a": (shared + rng.normal(scale=0.2, size=300)).tolist(),
            "model_b": (shared + rng.normal(scale=0.2, size=300)).tolist(),
            "model_c": rng.normal(size=300).tolist(),
        }
    )

    layer.fit(forecasts, signals=pd.DataFrame())

    weights = layer.weights_["ES"]
    diag = layer.get_diagnostics()["tickers"]["ES"]
    metrics = diag["cluster_metrics"]

    assert weights["model_c"] > weights["model_a"]
    assert weights["model_c"] > weights["model_b"]
    assert metrics[diag["cluster_assignments"]["model_c"]]["score"] > metrics[
        diag["cluster_assignments"]["model_a"]
    ]["score"]


def test_hrp_cluster_equal_groups_correlated_members_and_equal_weights_groups() -> None:
    layer = WeightLayer(weight_method="hrp_cluster_equal", rho_cut=0.7, group_weight_cap=1.0)
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
    assignments = diag["cluster_assignments"]

    assert assignments["fast_1"] == assignments["fast_2"]
    assert assignments["fast_1"] != assignments["slow_1"]
    assert weights["fast_1"] == pytest.approx(0.25)
    assert weights["fast_2"] == pytest.approx(0.25)
    assert weights["slow_1"] == pytest.approx(0.50)
    assert sorted(diag["cluster_weights"].values()) == pytest.approx([0.5, 0.5])


def test_hrp_cluster_equal_respects_group_weight_cap() -> None:
    layer = WeightLayer(weight_method="hrp_cluster_equal", rho_cut=0.7, group_weight_cap=0.55)
    forecasts = _make_forecasts(
        {
            "a1": [1, 0, 1, 0, 1, 0],
            "a2": [1, 0, 1, 0, 1, 0],
            "b1": [0, 1, 0, 1, 0, 1],
            "b2": [0, 1, 0, 1, 0, 1],
            "c1": [0.2, 0.4, 0.2, 0.4, 0.2, 0.4],
        }
    )

    layer.fit(forecasts, signals=pd.DataFrame())

    cluster_weights = layer.get_diagnostics()["tickers"]["ES"]["cluster_weights"]
    assert max(cluster_weights.values()) <= 0.55 + 1e-9


def test_hrp_classic_produces_non_trivial_branch_allocation() -> None:
    layer = WeightLayer(weight_method="hrp_classic", rho_cut=0.7)
    rng = np.random.default_rng(42)
    shared = rng.normal(size=400)
    forecasts = _make_forecasts(
        {
            "clustered_a": (shared + rng.normal(scale=0.05, size=400)).tolist(),
            "clustered_b": (shared + rng.normal(scale=0.05, size=400)).tolist(),
            "diverse_a": rng.normal(size=400).tolist(),
            "diverse_b": rng.normal(size=400).tolist(),
        }
    )

    layer.fit(forecasts, signals=pd.DataFrame())

    weights = layer.weights_["ES"]
    diag = layer.get_diagnostics()["tickers"]["ES"]
    clustered_weight = weights["clustered_a"] + weights["clustered_b"]
    diverse_weight = weights["diverse_a"] + weights["diverse_b"]

    assert weights.sum() == pytest.approx(1.0)
    assert diverse_weight > clustered_weight
    assert len(diag["cluster_weights"]) >= 2


def test_fit_ignores_returns_input_for_new_modes() -> None:
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

    no_returns = WeightLayer(weight_method="inverse_avg_pairwise_corr", rho_cut=0.7)
    series_layer = WeightLayer(weight_method="inverse_avg_pairwise_corr", rho_cut=0.7)
    frame_layer = WeightLayer(weight_method="inverse_avg_pairwise_corr", rho_cut=0.7)

    no_returns.fit(forecasts, signals=pd.DataFrame(), returns=None)
    series_layer.fit(forecasts, signals=pd.DataFrame(), returns=series_returns)
    frame_layer.fit(forecasts, signals=pd.DataFrame(), returns=frame_returns)

    pd.testing.assert_series_equal(no_returns.weights_["ES"], series_layer.weights_["ES"])
    pd.testing.assert_series_equal(no_returns.weights_["ES"], frame_layer.weights_["ES"])


def test_constant_or_short_history_inputs_fall_back_to_equal_weights_and_fdm_one() -> None:
    short_layer = WeightLayer(weight_method="hrp_classic")
    short_forecasts = [
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
    short_layer.fit(short_forecasts, signals=pd.DataFrame())

    constant_layer = WeightLayer(weight_method="hrp_classic")
    constant_forecasts = _make_forecasts(
        {
            "model_a": [1.0] * 12,
            "model_b": [1.0] * 12,
            "model_c": [1.0] * 12,
        }
    )
    constant_layer.fit(constant_forecasts, signals=pd.DataFrame())

    assert short_layer.weights_["ES"].to_dict() == pytest.approx(
        {"model_a": 0.5, "model_b": 0.5}
    )
    assert short_layer.fdm_["ES"] == pytest.approx(1.0)
    assert constant_layer.weights_["ES"].to_dict() == pytest.approx(
        {"model_a": 1.0 / 3.0, "model_b": 1.0 / 3.0, "model_c": 1.0 / 3.0}
    )
    assert constant_layer.fdm_["ES"] == pytest.approx(1.0)


def test_fdm_uses_raw_signal_correlations_and_respects_cap() -> None:
    rng = np.random.default_rng(42)
    model_a = rng.normal(size=80)
    model_b = rng.normal(size=80)
    layer = WeightLayer(weight_method="hrp_classic", fdm_max=1.3)
    forecasts = _make_forecasts({"model_a": model_a.tolist(), "model_b": model_b.tolist()})

    layer.fit(forecasts, signals=pd.DataFrame())

    diag = layer.get_diagnostics()["tickers"]["ES"]
    assert layer.fdm_["ES"] <= 1.3
    assert diag["mean_signal_correlation"] < 1.0


def test_combine_returns_datetime_level_forecasts() -> None:
    layer = WeightLayer(weight_method="equal_signal")
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
