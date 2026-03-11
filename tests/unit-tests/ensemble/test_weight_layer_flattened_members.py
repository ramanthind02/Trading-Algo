"""Tests for flattened member names in the clustered weight layer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ensemble.weight_layer import WeightLayer, WeightLayerConfig
from ensemble.portfolio import Portfolio
from utils.core.enums import TimeFrame


def _member_forecasts() -> list[pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    inputs = {
        "ewmac_fast::q33": [0.0, 0.4, 0.0, 0.4, 0.0, 0.4],
        "ewmac_fast::q66": [0.0, 0.4, 0.0, 0.4, 0.0, 0.4],
        "ewmac_slow::lower": [0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
    }
    return [
        pd.DataFrame(
            {
                "ticker": ["ES"] * len(values),
                "datetime": index,
                "model_name": [model_name] * len(values),
                "forecast": values,
                "signal": [int(v != 0.0) for v in values],
            }
        )
        for model_name, values in inputs.items()
    ]


def test_weight_layer_config_supports_flattened_model_names() -> None:
    config = WeightLayerConfig(weighting_method="cluster_equal")
    assert config.weighting_method == "cluster_equal"


def test_weight_layer_clusters_flattened_member_names() -> None:
    weight_layer = WeightLayer(weight_method="cluster_equal", fdm_max=2.0, rho_cut=0.7)
    forecasts = _member_forecasts()

    weight_layer.fit(forecasts, signals=pd.DataFrame())

    assignments = weight_layer.get_diagnostics()["tickers"]["ES"]["cluster_assignments"]
    assert assignments["ewmac_fast::q33"] == assignments["ewmac_fast::q66"]
    assert assignments["ewmac_fast::q33"] != assignments["ewmac_slow::lower"]


def test_weight_layer_weights_preserve_member_identifiers() -> None:
    weight_layer = WeightLayer(weight_method="cluster_equal")
    forecasts = _member_forecasts()

    weight_layer.fit(forecasts, signals=pd.DataFrame())

    weights = weight_layer.weights_["ES"]
    assert all("::" in name for name in weights.index)
    assert abs(float(weights.sum()) - 1.0) < 1e-6


def test_portfolio_predict_accepts_member_level_forecasts() -> None:
    mock_ensemble = MagicMock()
    mock_ensemble.predict.return_value = pd.DataFrame(
        {
            "ticker": ["ES", "ES", "NQ", "NQ"],
            "model_name": ["model_a::m1", "model_a::m2", "model_b::m1", "model_b::m2"],
            "forecast": [0.1, -0.1, 0.15, -0.15],
            "signal": [1, 1, 1, 1],
        }
    )
    mock_ensemble.unique_tickers_ = ["ES", "NQ"]

    portfolio = Portfolio(
        ensembles=[mock_ensemble],
        trading_timeframe=TimeFrame.D,
    )
    portfolio.is_fitted_ = True
    portfolio.idm_ = 1.0

    mock_wl = MagicMock()
    mock_wl.combine.return_value = pd.DataFrame(
        {
            "ticker": ["ES", "NQ"],
            "forecast": [0.05, 0.07],
            "signal": [1, 1],
        }
    )
    portfolio.weight_layer = mock_wl

    result = mock_ensemble.predict()
    model_names = result["model_name"].tolist()

    assert "model_a::m1" in model_names
    assert "model_b::m2" in model_names
