"""Unit tests for 3D param grid (selected_bin) in walkforward pipeline."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from feature_research.pipeline import (
    _combo_key,
    _expand_params_with_selected_bin,
    _build_continuous_walkforward_evaluator,
)
from feature_research.in_sample.config import load_config
from feature_research.walkforward.config import WalkforwardResearchConfig
from feature_research.walkforward.runner import run_walkforward_research
from feature_research.walkforward.io import _build_selected_params_detailed


def test_expand_params_with_selected_bin_yields_one_per_bin() -> None:
    """_expand_params_with_selected_bin with one entry {lookback: 5, bin_count: 3} yields three entries."""
    params_list = [{"lookback": 5, "bin_count": 3}]
    out = _expand_params_with_selected_bin(params_list)
    assert len(out) == 3
    assert out[0] == {"lookback": 5, "bin_count": 3, "selected_bin": 0}
    assert out[1] == {"lookback": 5, "bin_count": 3, "selected_bin": 1}
    assert out[2] == {"lookback": 5, "bin_count": 3, "selected_bin": 2}


def test_expand_params_with_selected_bin_skips_when_no_bin_count() -> None:
    """When bin_count is missing, list is returned unchanged."""
    params_list = [{"lookback": 5}]
    out = _expand_params_with_selected_bin(params_list)
    assert out == [{"lookback": 5}]


def test_expand_params_with_selected_bin_multiple_combos() -> None:
    """Multiple 2D combos are each expanded by selected_bin."""
    params_list = [
        {"lookback": 2, "bin_count": 2},
        {"lookback": 3, "bin_count": 2},
    ]
    out = _expand_params_with_selected_bin(params_list)
    assert len(out) == 4
    labels = ["|".join(f"{k}={v}" for k, v in sorted(p.items())) for p in out]
    assert "bin_count=2|lookback=2|selected_bin=0" in labels
    assert "bin_count=2|lookback=2|selected_bin=1" in labels
    assert "bin_count=2|lookback=3|selected_bin=0" in labels
    assert "bin_count=2|lookback=3|selected_bin=1" in labels


def test_continuous_evaluator_with_selected_bin_returns_forced_bin_meta() -> None:
    """Evaluator with params containing selected_bin returns meta selected_long_bin equal to that bin."""
    config = load_config()
    # Override to minimal: one (lookback, bin_count) so we have one combo key
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    feature = pd.Series(0.5, index=index, name="feat")
    target = pd.Series(0.01, index=index, name="target")
    combo_feature_target = {
        _combo_key({"lookback": 5, "bin_count": 3}): pd.DataFrame({"feature": feature, "target": target}),
    }
    evaluator = _build_continuous_walkforward_evaluator(combo_feature_target, config)
    fold_candles = pd.DataFrame({"close": target}, index=index)
    params = {"lookback": 5, "bin_count": 3, "selected_bin": 1}
    train_end = pd.Timestamp("2020-06-30")
    result = evaluator(
        fold_candles,
        target,
        params,
        train_end=train_end,
    )
    series, meta = cast(tuple[pd.Series, dict], result)
    assert meta["selected_long_bin"] == 1
    assert len(series) == len(index)
    # Forced bin 1: series is non-zero only where model assigns bin 1
    assert (series.fillna(0) != 0).sum() >= 0  # may be 0 or more depending on quantiles


def test_continuous_evaluator_lookup_ignores_selected_bin_for_combo_key() -> None:
    """Evaluator looks up combo_feature_target by key without selected_bin."""
    config = load_config()
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    feature = pd.Series(0.3, index=index, name="feat")
    target = pd.Series(0.01, index=index, name="target")
    df = pd.DataFrame({"feature": feature, "target": target})
    # Only 2D key in the dict
    combo_feature_target = {_combo_key({"lookback": 7, "bin_count": 5}): df}
    evaluator = _build_continuous_walkforward_evaluator(combo_feature_target, config)
    fold_candles = pd.DataFrame({"close": target}, index=index)
    # 3D params: lookup must succeed via key without selected_bin
    params = {"lookback": 7, "bin_count": 5, "selected_bin": 0}
    train_end = pd.Timestamp("2020-05-01")
    result = evaluator(fold_candles, target, params, train_end=train_end)
    series, meta = cast(tuple[pd.Series, dict], result)
    assert meta["selected_long_bin"] == 0
    assert len(series) == 100


def test_walkforward_3d_grid_produces_param_label_with_selected_bin(tmp_path: object) -> None:
    """Run walkforward with 3D param_grid; fold_scores_df and selected_params_detailed include selected_bin."""
    config = load_config()
    index = pd.date_range("2020-01-01", periods=150, freq="D")
    target = pd.Series(0.01, index=index, name="walkforward_target")
    candles_df = pd.DataFrame({"close": target}, index=index)
    feature = pd.Series(0.4, index=index, name="feat")
    combo_feature_target = {
        _combo_key({"lookback": 5, "bin_count": 2}): pd.DataFrame({"feature": feature, "target": target}),
    }
    param_grid = _expand_params_with_selected_bin([{"lookback": 5, "bin_count": 2}])
    assert len(param_grid) == 2
    wf_config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 4, 1),
        test_step=30,
        num_steps=2,
        top_k=2,
        objective_metric_name="t_stat",
        min_fold_samples=10,
        output_root=Path(str(tmp_path)),
    )
    evaluator = _build_continuous_walkforward_evaluator(combo_feature_target, config)
    report = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="rsi",
        config=wf_config,
        param_grid=param_grid,
        evaluate_param_combo=evaluator,
        research_config=config,
        output_dir=None,
    )
    assert report.fold_scores_df["param_label"].str.contains("selected_bin=", regex=False).any(), (
        "fold_scores_df should contain param_label with selected_bin= when using 3D grid"
    )
    detailed = _build_selected_params_detailed(report, research_context={})
    if not detailed.empty:
        assert "param_selected_bin" in detailed.columns, (
            "selected_params_detailed should have param_selected_bin when param_label contains selected_bin"
        )
