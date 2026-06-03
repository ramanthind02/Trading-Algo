"""Integration smoke: asset-first hierarchy_equal on synthetic global streams."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensemble.portfolio_impl.global_weight_layer_adapter import _GLOBAL_WEIGHT_LAYER_TICKER
from ensemble.vault.hierarchy_spec import build_asset_first_hierarchy_spec
from ensemble.weight_hierarchy import compute_equal_split_weights, parse_hierarchy_spec
from ensemble.weight_layer import (
    WeightLayer,
    WeightLayerConfig,
    deserialize_weight_layer_state,
    serialize_weight_layer_state,
)


def _global_forecasts(stream_ids: list[str], *, n: int = 30) -> list[pd.DataFrame]:
    rng = np.random.default_rng(42)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return [
        pd.DataFrame(
            {
                "ticker": [_GLOBAL_WEIGHT_LAYER_TICKER] * n,
                "datetime": index,
                "model_name": [sid] * n,
                "forecast": rng.normal(size=n).tolist(),
                "signal": [1] * n,
            }
        )
        for sid in stream_ids
    ]


def test_asset_first_weight_layer_fit_and_roundtrip() -> None:
    streams = {
        "equity_indices": {
            "momentum": ["ES::D::m1", "NQ::D::m2"],
        },
        "commodities": {
            "momentum": ["GC::D::m3"],
        },
        "diversified": {
            "buy_hold": ["ES::M::m4", "GC::M::m5"],
        },
    }
    stream_ids = sorted(
        sid for styles in streams.values() for sids in styles.values() for sid in sids
    )
    spec = build_asset_first_hierarchy_spec(streams)
    root = parse_hierarchy_spec(spec)
    weights_pre, assignments, _, _ = compute_equal_split_weights(root, stream_ids)

    layer = WeightLayer(weight_method="hierarchy_equal", hierarchy_spec=spec, fdm_max=2.0)
    layer.fit(_global_forecasts(stream_ids), signals=pd.DataFrame())

    fitted = layer.weights_[_GLOBAL_WEIGHT_LAYER_TICKER]
    assert set(fitted.index) == set(stream_ids)
    for sid in stream_ids:
        assert fitted[sid] == pytest.approx(weights_pre[sid])

    diag = layer.get_diagnostics()["tickers"][_GLOBAL_WEIGHT_LAYER_TICKER]
    assert diag["cluster_assignments"]["ES::D::m1"].startswith("root/equity_indices/momentum/")

    by_asset: dict[str, float] = {}
    for sid, w in fitted.items():
        parts = assignments[sid].split("/")
        if parts and parts[0] == "root":
            parts = parts[1:]
        asset = parts[0] if parts else "unknown"
        by_asset[asset] = by_asset.get(asset, 0.0) + float(w)
    assert by_asset["equity_indices"] == pytest.approx(1.0 / 3.0)
    assert by_asset["commodities"] == pytest.approx(1.0 / 3.0)
    assert by_asset["diversified"] == pytest.approx(1.0 / 3.0)

    blob = serialize_weight_layer_state(layer)
    restored = deserialize_weight_layer_state(blob)
    assert restored._wl_config.hierarchy_spec == spec
    restored.fit(_global_forecasts(stream_ids), signals=pd.DataFrame())
    restored_w = restored.weights_[_GLOBAL_WEIGHT_LAYER_TICKER]
    for sid in stream_ids:
        assert restored_w[sid] == pytest.approx(fitted[sid])


def test_hierarchy_equal_with_sr_adjustment_changes_weights() -> None:
    streams = {
        "equity_indices": {
            "momentum": ["ES::D::m1", "NQ::D::m2"],
        },
        "commodities": {
            "momentum": ["GC::D::m3"],
        },
    }
    stream_ids = sorted(
        sid for styles in streams.values() for sids in styles.values() for sid in sids
    )
    spec = build_asset_first_hierarchy_spec(streams)
    n = 800
    index = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(99)

    def _forecasts() -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for sid in stream_ids:
            sig = rng.normal(0.0, 0.5, size=n)
            frames.append(
                pd.DataFrame(
                    {
                        "ticker": [_GLOBAL_WEIGHT_LAYER_TICKER] * n,
                        "datetime": index,
                        "model_name": [sid] * n,
                        "forecast": sig.tolist(),
                        "signal": sig.tolist(),
                    }
                )
            )
        return frames

    instrument_returns = pd.DataFrame(
        {
            "ES": rng.normal(0.0008, 0.01, size=n),
            "NQ": rng.normal(0.0003, 0.01, size=n),
            "GC": rng.normal(-0.0002, 0.01, size=n),
        },
        index=index,
    )

    equal_layer = WeightLayer(
        config=WeightLayerConfig(
            weighting_method="hierarchy_equal",
            hierarchy_spec=spec,
            fdm_max=2.0,
        )
    )
    equal_layer.fit(_forecasts(), signals=pd.DataFrame())

    sr_layer = WeightLayer(
        config=WeightLayerConfig(
            weighting_method="hierarchy_equal",
            hierarchy_spec=spec,
            fdm_max=2.0,
            sr_adjustment=True,
            sr_min_years=1.0,
            sr_p_step=0.1,
        )
    )
    sr_layer.fit(_forecasts(), signals=pd.DataFrame(), returns=instrument_returns)

    equal_w = equal_layer.weights_[_GLOBAL_WEIGHT_LAYER_TICKER]
    sr_w = sr_layer.weights_[_GLOBAL_WEIGHT_LAYER_TICKER]
    assert float(sr_w.sum()) == pytest.approx(1.0)
    assert not np.allclose(equal_w.values, sr_w.values)
