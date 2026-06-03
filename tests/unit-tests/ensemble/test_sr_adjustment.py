"""Unit tests for Carver SR adjustment and hierarchical tilt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensemble.sr_adjustment import (
    SrAdjustmentParams,
    annualized_sharpe,
    apply_hierarchical_sr_tilt,
    apply_within_group_inv_corr,
    build_stream_pnl_series,
    mini_bootstrap_weight_ratio,
    pivot_stream_signals,
    weights_given_sr_diff,
)
from ensemble.vault.hierarchy_spec import build_asset_first_hierarchy_spec
from ensemble.weight_hierarchy import compute_equal_split_weights, parse_hierarchy_spec


def test_mini_bootstrap_ratio_blog_example() -> None:
    median_weights = weights_given_sr_diff(
        0.2,
        avg_correlation=0.3,
        years_of_data=10.0,
        confidence_interval=0.5,
    )
    assert median_weights[0] == pytest.approx(0.653, rel=0.02)
    assert median_weights[1] == pytest.approx(0.347, rel=0.02)

    ratio = mini_bootstrap_weight_ratio(
        0.2,
        avg_correlation=0.3,
        years_of_data=10.0,
        avg_sr=0.5,
        std=0.15,
        p_step=0.01,
    )
    assert ratio > 1.0
    assert ratio < 1.5


def test_apply_hierarchical_sr_tilt_preserves_total_mass() -> None:
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
    root = parse_hierarchy_spec(spec)
    weights, assignments, _, _ = compute_equal_split_weights(root, stream_ids)

    n = 600
    index = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    stream_pnls = {
        "ES::D::m1": pd.Series(rng.normal(0.001, 0.01, size=n), index=index),
        "NQ::D::m2": pd.Series(rng.normal(0.0002, 0.01, size=n), index=index),
        "GC::D::m3": pd.Series(rng.normal(-0.0005, 0.01, size=n), index=index),
    }

    params = SrAdjustmentParams(sr_min_years=1.0, sr_p_step=0.1)
    adjusted, diag = apply_hierarchical_sr_tilt(
        weights, assignments, stream_pnls, params
    )
    assert float(adjusted.sum()) == pytest.approx(1.0)
    assert len(diag) > 0
    assert not np.allclose(adjusted.values, weights.values)


def test_sr_min_years_skips_tilt() -> None:
    streams = {
        "equity_indices": {
            "momentum": ["ES::D::m1", "NQ::D::m2"],
        },
    }
    stream_ids = ["ES::D::m1", "NQ::D::m2"]
    spec = build_asset_first_hierarchy_spec(streams)
    root = parse_hierarchy_spec(spec)
    weights, assignments, _, _ = compute_equal_split_weights(root, stream_ids)

    index = pd.date_range("2024-01-01", periods=60, freq="B")
    stream_pnls = {
        "ES::D::m1": pd.Series(np.linspace(0.001, 0.002, 60), index=index),
        "NQ::D::m2": pd.Series(np.linspace(0.0005, 0.0015, 60), index=index),
    }
    adjusted, diag = apply_hierarchical_sr_tilt(
        weights,
        assignments,
        stream_pnls,
        SrAdjustmentParams(sr_min_years=5.0),
    )
    assert adjusted.equals(weights)
    assert diag == []


def test_build_stream_pnl_from_signals() -> None:
    n = 50
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    forecasts = pd.DataFrame(
        {
            "ticker": ["__GLOBAL__"] * n,
            "datetime": index,
            "model_name": ["ES::D::m1"] * n,
            "signal": np.linspace(0.5, 1.0, n),
            "forecast": np.linspace(0.5, 1.0, n),
        }
    )
    pivot = pivot_stream_signals(forecasts, ["ES::D::m1"], value_column="forecast")
    rets = pd.DataFrame({"ES": np.full(n, 0.01)}, index=index)
    pnls = build_stream_pnl_series(pivot, rets, ["ES::D::m1"])
    assert "ES::D::m1" in pnls
    assert annualized_sharpe(pnls["ES::D::m1"]) > 0.0
    assert float(pnls["ES::D::m1"].iloc[0]) == pytest.approx(0.0)


def test_build_stream_pnl_lagged_beats_contemporaneous_for_mr_proxy() -> None:
    """Lagged position × return should rank a persistent long bias above noise."""
    n = 600
    index = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    inst_ret = pd.Series(rng.normal(0.0004, 0.01, size=n), index=index)
    good_forecast = pd.Series(5.0, index=index)
    bad_forecast = pd.Series(rng.choice([-5.0, 5.0], size=n), index=index)
    pivot = pd.DataFrame(
        {
            "ES::D::good": good_forecast,
            "ES::D::bad": bad_forecast,
        }
    )
    rets = pd.DataFrame({"ES": inst_ret})
    lagged = build_stream_pnl_series(pivot, rets, list(pivot.columns), lag_positions=True)
    assert annualized_sharpe(lagged["ES::D::good"]) > annualized_sharpe(lagged["ES::D::bad"])
    assert float(lagged["ES::D::good"].iloc[0]) == pytest.approx(0.0)


def test_sr_tilt_increases_weight_on_higher_sr_style_group() -> None:
    streams = {
        "equity_indices": {
            "mean_reversion_indices": ["ES::D::mr"],
            "momentum": ["ES::D::mom"],
        },
    }
    stream_ids = ["ES::D::mr", "ES::D::mom"]
    spec = build_asset_first_hierarchy_spec(streams)
    root = parse_hierarchy_spec(spec)
    weights, assignments, _, _ = compute_equal_split_weights(root, stream_ids)
    n = 800
    index = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(11)
    inst = rng.normal(0.0005, 0.01, size=n)
    mr_forecast = pd.Series(8.0, index=index)
    mom_forecast = pd.Series(rng.choice([-4.0, 4.0], size=n), index=index)
    pivot = pd.DataFrame({"ES::D::mr": mr_forecast, "ES::D::mom": mom_forecast})
    rets = pd.DataFrame({"ES": inst}, index=index)
    stream_pnls = build_stream_pnl_series(pivot, rets, stream_ids, lag_positions=True)
    adjusted, diag = apply_hierarchical_sr_tilt(
        weights,
        assignments,
        stream_pnls,
        SrAdjustmentParams(sr_min_years=1.0, sr_p_step=0.1),
    )
    mr_mass = sum(
        float(adjusted[sid])
        for sid, path in assignments.items()
        if "mean_reversion_indices" in path
    )
    mom_mass = sum(
        float(adjusted[sid])
        for sid, path in assignments.items()
        if "momentum" in path
    )
    assert mr_mass > mom_mass
    assert any("mean_reversion_indices" in d.member_key for d in diag)


def test_apply_hierarchical_sr_tilt_max_depth_skips_leaf_level() -> None:
    """max_depth=2 must skip the within-group (L3) instrument level.

    Hierarchy has depth 3 (root/asset/group/stream).  With max_depth=2 the
    deepest sibling group (root/equity_indices/momentum, depth=3) must not be
    tilted, so both momentum streams keep equal weight.  The asset-level split
    (root, depth=1) is still tilted because depth 1 ≤ 2.
    """
    streams = {
        "equity_indices": {
            "momentum": ["ES::D::fast", "NQ::D::slow"],
        },
        "commodities": {
            "momentum": ["GC::D::gc"],
        },
    }
    stream_ids = ["ES::D::fast", "NQ::D::slow", "GC::D::gc"]
    spec = build_asset_first_hierarchy_spec(streams)
    root = parse_hierarchy_spec(spec)
    weights, assignments, _, _ = compute_equal_split_weights(root, stream_ids)

    n = 800
    index = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)

    # ES fast stream has much higher SR than NQ slow — would receive more weight at L3
    stream_pnls = {
        "ES::D::fast": pd.Series(rng.normal(0.001, 0.008, size=n), index=index),
        "NQ::D::slow": pd.Series(rng.normal(0.0001, 0.01, size=n), index=index),
        "GC::D::gc":   pd.Series(rng.normal(0.0002, 0.01, size=n), index=index),
    }
    params = SrAdjustmentParams(sr_min_years=1.0, sr_p_step=0.1)

    # Without depth cap: L3 tilt should give ES::D::fast more weight than NQ::D::slow
    adjusted_full, _ = apply_hierarchical_sr_tilt(weights, assignments, stream_pnls, params)
    assert float(adjusted_full["ES::D::fast"]) > float(adjusted_full["NQ::D::slow"])

    # With max_depth=2: within-group tilt is skipped so ES and NQ keep equal share
    adjusted_capped, diag_capped = apply_hierarchical_sr_tilt(
        weights, assignments, stream_pnls, params, max_depth=2
    )
    assert float(adjusted_capped["ES::D::fast"]) == pytest.approx(
        float(adjusted_capped["NQ::D::slow"]), rel=1e-6
    )
    # Total mass must still be preserved
    assert float(adjusted_capped.sum()) == pytest.approx(1.0)
    # L1 tilt (root, depth=1) should still fire — commodity vs equity masses differ
    eq_mass = float(adjusted_capped["ES::D::fast"]) + float(adjusted_capped["NQ::D::slow"])
    gc_mass = float(adjusted_capped["GC::D::gc"])
    # equity mass should not equal commodity mass (L1 tilt applied)
    assert not pytest.approx(eq_mass, rel=1e-3) == gc_mass


def test_weight_adjustment_outputs_are_nonnegative() -> None:
    """Clip+renormalize must keep portfolio weights in [0, 1] even after pathological tilt."""
    streams = {
        "equity_indices": {
            "mean_reversion_indices": [
                "ES::D::a",
                "ES::D::b",
                "ES::W::c",
            ],
        },
    }
    stream_ids = [sid for styles in streams.values() for sids in styles.values() for sid in sids]
    spec = build_asset_first_hierarchy_spec(streams)
    root = parse_hierarchy_spec(spec)
    weights, assignments, _, _ = compute_equal_split_weights(root, stream_ids)
    n = 800
    index = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(21)
    stream_pnls = {
        "ES::D::a": pd.Series(rng.normal(0.001, 0.01, n), index=index),
        "ES::D::b": pd.Series(rng.normal(0.0008, 0.01, n), index=index),
        "ES::W::c": pd.Series(rng.normal(-0.002, 0.012, n), index=index),
    }
    params = SrAdjustmentParams(sr_min_years=1.0, sr_p_step=0.01)
    tilted, _ = apply_hierarchical_sr_tilt(weights, assignments, stream_pnls, params)
    adjusted = apply_within_group_inv_corr(tilted, assignments, stream_pnls, min_depth=2)
    assert float(adjusted.min()) >= 0.0
    assert float(adjusted.sum()) == pytest.approx(1.0, abs=1e-9)


def test_apply_within_group_inv_corr_redistributes_within_group() -> None:
    """within_group_inv_corr should upweight the less-correlated stream within each group."""
    streams = {
        "equity_indices": {
            "mean_rev": ["ES::D::ibs", "NQ::D::tat"],
            "momentum": ["ES::D::sma", "NQ::D::algo"],
        },
    }
    stream_ids = ["ES::D::ibs", "NQ::D::tat", "ES::D::sma", "NQ::D::algo"]
    spec = build_asset_first_hierarchy_spec(streams)
    root = parse_hierarchy_spec(spec)
    weights, assignments, _, _ = compute_equal_split_weights(root, stream_ids)

    n = 800
    index = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)

    common = pd.Series(rng.normal(0, 0.01, size=n), index=index)
    # ibs and tat highly correlated (share large common factor)
    stream_pnls = {
        "ES::D::ibs":  common * 0.9 + pd.Series(rng.normal(0, 0.003, size=n), index=index),
        "NQ::D::tat":  common * 0.9 + pd.Series(rng.normal(0, 0.003, size=n), index=index),
        # sma almost independent of algo
        "ES::D::sma":  pd.Series(rng.normal(0, 0.01, size=n), index=index),
        "NQ::D::algo": pd.Series(rng.normal(0, 0.01, size=n), index=index),
    }

    adjusted = apply_within_group_inv_corr(weights, assignments, stream_pnls, min_depth=2)

    # Total mass preserved
    assert float(adjusted.sum()) == pytest.approx(1.0, abs=1e-9)

    # Within mean_rev: ibs and tat are nearly identical in weight (both penalised equally
    # for their high mutual correlation)
    assert float(adjusted["ES::D::ibs"]) == pytest.approx(float(adjusted["NQ::D::tat"]), rel=0.05)

    # Within momentum: sma and algo are uncorrelated → inv_corr gives equal weight
    assert float(adjusted["ES::D::sma"]) == pytest.approx(float(adjusted["NQ::D::algo"]), rel=0.05)

    # Equal-weight baseline: each stream = 0.25; inv-corr must not change total group mass
    mean_rev_total = float(adjusted["ES::D::ibs"]) + float(adjusted["NQ::D::tat"])
    momentum_total = float(adjusted["ES::D::sma"]) + float(adjusted["NQ::D::algo"])
    assert mean_rev_total == pytest.approx(0.5, abs=0.02)
    assert momentum_total == pytest.approx(0.5, abs=0.02)


def test_within_group_method_config_validation() -> None:
    """within_group_method must be 'equal' or 'inverse_avg_pairwise_corr'."""
    from ensemble.weight_layer import WeightLayerConfig

    with pytest.raises(ValueError, match="within_group_method"):
        WeightLayerConfig(weighting_method="equal_signal", within_group_method="invalid")

    cfg = WeightLayerConfig(weighting_method="equal_signal", within_group_method="equal")
    assert cfg.within_group_method == "equal"

    cfg2 = WeightLayerConfig(
        weighting_method="equal_signal",
        within_group_method="inverse_avg_pairwise_corr",
    )
    assert cfg2.within_group_method == "inverse_avg_pairwise_corr"


def test_weight_layer_config_sr_tilt_max_depth_validation() -> None:
    """sr_tilt_max_depth=0 must raise; None and positive ints must be accepted."""
    from ensemble.weight_layer import WeightLayerConfig
    import pytest

    with pytest.raises(ValueError, match="sr_tilt_max_depth"):
        WeightLayerConfig(
            weighting_method="equal_signal",
            sr_tilt_max_depth=0,
        )

    cfg = WeightLayerConfig(weighting_method="equal_signal", sr_tilt_max_depth=2)
    assert cfg.sr_tilt_max_depth == 2

    cfg_none = WeightLayerConfig(weighting_method="equal_signal", sr_tilt_max_depth=None)
    assert cfg_none.sr_tilt_max_depth is None
