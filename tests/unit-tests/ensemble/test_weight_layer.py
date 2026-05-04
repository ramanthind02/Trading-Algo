"""Unit tests for WeightLayer allocation modes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensemble.weight_layer import (
    WeightLayer,
    WeightLayerConfig,
    deserialize_weight_layer_state,
    serialize_weight_layer_state,
    _correlation_multiplier_from_corr_matrix,
)


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


def _minimal_hierarchy_spec() -> dict[str, object]:
    return {
        "type": "group",
        "id": "root",
        "children": [
            {"type": "leaf", "stream_id": "a"},
            {"type": "leaf", "stream_id": "b"},
        ],
    }


def test_weight_layer_config_accepts_new_modes_and_rejects_legacy_ones() -> None:
    for method in ("equal_signal", "inverse_avg_pairwise_corr"):
        assert WeightLayerConfig(weighting_method=method).weighting_method == method

    spec = _minimal_hierarchy_spec()
    assert (
        WeightLayerConfig(weighting_method="hierarchy_equal", hierarchy_spec=spec).weighting_method
        == "hierarchy_equal"
    )

    with pytest.raises(ValueError, match="weighting_method must be one of"):
        WeightLayerConfig(weighting_method="cluster_equal")

    with pytest.raises(ValueError, match="hierarchy_equal requires"):
        WeightLayerConfig(weighting_method="hierarchy_equal")


def test_weight_layer_config_rejects_invalid_optimize_sortino_weighting_method() -> None:
    with pytest.raises(ValueError, match="weighting_method must be one of"):
        WeightLayerConfig(weighting_method="optimize_sortino_capped")


def test_weight_layer_rejects_removed_risk_tilt_alpha_kwarg() -> None:
    with pytest.raises(ValueError, match="risk_tilt_alpha is no longer supported"):
        WeightLayer(weight_method="equal_signal", risk_tilt_alpha=0.5)


def test_weight_layer_rejects_unknown_kwargs() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        WeightLayer(weight_method="equal_signal", rho_cut=0.7)


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


def test_hierarchy_equal_splits_groups_and_matches_manual_tree() -> None:
    spec: dict[str, object] = {
        "type": "group",
        "id": "root",
        "children": [
            {
                "type": "group",
                "id": "fast",
                "children": [
                    {"type": "leaf", "stream_id": "fast_1"},
                    {"type": "leaf", "stream_id": "fast_2"},
                ],
            },
            {"type": "leaf", "stream_id": "slow_1"},
        ],
    }
    layer = WeightLayer(weight_method="hierarchy_equal", hierarchy_spec=spec)
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

    assert assignments["fast_1"].startswith("root/fast/")
    assert assignments["fast_2"].startswith("root/fast/")
    assert assignments["slow_1"].startswith("root/")
    assert weights["fast_1"] == pytest.approx(0.25)
    assert weights["fast_2"] == pytest.approx(0.25)
    assert weights["slow_1"] == pytest.approx(0.50)


def test_hierarchy_equal_strict_rejects_extra_forecast_streams() -> None:
    spec: dict[str, object] = {
        "type": "group",
        "id": "root",
        "children": [
            {"type": "leaf", "stream_id": "a"},
            {"type": "leaf", "stream_id": "b"},
        ],
    }
    layer = WeightLayer(weight_method="hierarchy_equal", hierarchy_spec=spec)
    forecasts = _make_forecasts(
        {
            "a": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "b": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "c": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="not in hierarchy"):
        layer.fit(forecasts, signals=pd.DataFrame())


def test_deserialize_raises_on_legacy_hrp_snapshot() -> None:
    payload = {
        "config": {"weighting_method": "hrp_classic", "fdm_max": 2.0},
        "state": {"fdm": {}, "is_fitted": False},
    }
    with pytest.raises(ValueError, match="removed"):
        deserialize_weight_layer_state(payload)


def test_serialize_roundtrip_preserves_hierarchy_spec() -> None:
    spec: dict[str, object] = {
        "type": "group",
        "id": "root",
        "children": [
            {"type": "leaf", "stream_id": "x"},
            {"type": "leaf", "stream_id": "y"},
        ],
    }
    layer = WeightLayer(
        config=WeightLayerConfig(weighting_method="hierarchy_equal", hierarchy_spec=spec)
    )
    blob = serialize_weight_layer_state(layer)
    restored = deserialize_weight_layer_state(blob)
    assert restored._wl_config.weighting_method == "hierarchy_equal"
    assert restored._wl_config.hierarchy_spec == spec


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

    no_returns = WeightLayer(weight_method="inverse_avg_pairwise_corr")
    series_layer = WeightLayer(weight_method="inverse_avg_pairwise_corr")
    frame_layer = WeightLayer(weight_method="inverse_avg_pairwise_corr")

    no_returns.fit(forecasts, signals=pd.DataFrame(), returns=None)
    series_layer.fit(forecasts, signals=pd.DataFrame(), returns=series_returns)
    frame_layer.fit(forecasts, signals=pd.DataFrame(), returns=frame_returns)

    pd.testing.assert_series_equal(no_returns.weights_["ES"], series_layer.weights_["ES"])
    pd.testing.assert_series_equal(no_returns.weights_["ES"], frame_layer.weights_["ES"])


def test_constant_or_short_history_inputs_fall_back_to_equal_weights_and_fdm_one() -> None:
    short_layer = WeightLayer(weight_method="equal_signal")
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

    constant_layer = WeightLayer(weight_method="equal_signal")
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
    layer = WeightLayer(weight_method="equal_signal", fdm_max=1.3)
    forecasts = _make_forecasts({"model_a": model_a.tolist(), "model_b": model_b.tolist()})

    layer.fit(forecasts, signals=pd.DataFrame())

    diag = layer.get_diagnostics()["tickers"]["ES"]
    assert layer.fdm_["ES"] <= 1.3
    assert diag["mean_signal_correlation"] < 1.0


def test_correlation_multiplier_helper_floors_negative_correlation_and_caps_result() -> None:
    corr = pd.DataFrame(
        [[1.0, -0.4, 0.6], [-0.4, 1.0, 0.2], [0.6, 0.2, 1.0]],
        columns=["a", "b", "c"],
        index=["a", "b", "c"],
    )

    mean_corr, multiplier = _correlation_multiplier_from_corr_matrix(corr, cap=1.5)

    assert mean_corr == pytest.approx((0.0 + 0.6 + 0.2) / 3.0)
    assert multiplier == pytest.approx(1.5)


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
