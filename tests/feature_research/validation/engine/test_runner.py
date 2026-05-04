from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
import pytest

from utils.evaluation.walkforward.config import (
    WeightLayerAlgorithm,
    WalkforwardResearchConfig,
)
from utils.evaluation.walkforward.runner import (
    build_fold_rows_from_explicit_specs,
    run_walkforward_research,
    run_portfolio_simulation,
)
from utils.evaluation.walkforward.walkforward_labels import canonical_param_label
from utils.evaluation.walkforward.portfolio_evaluator import evaluate_fold_portfolio
from utils.evaluation.walkforward.selected_params_codec import serialize_selected_params
from ensemble.weight_layer import WeightLayerConfig
from feature_research.config import FeatureType
from utils.core.enums import TimeFrame, Ticker


def _build_inputs() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    candles_df = pd.DataFrame({"close": range(120)}, index=index)
    target = pd.Series(0.01, index=index, name="target")
    return candles_df, target


def test_build_fold_rows_from_explicit_specs_produces_same_shape_as_walkforward() -> None:
    """Explicit specs produce fold rows with train/test masks and boundaries for OOS reuse."""
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    train_start = datetime(2020, 1, 1)
    train_end = datetime(2020, 2, 10)
    test_start = datetime(2020, 2, 11)
    test_end = datetime(2020, 3, 15)
    rows = build_fold_rows_from_explicit_specs(
        index,
        [(train_start, train_end, test_start, test_end)],
        min_fold_samples=5,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["fold_id"] == 0
    assert "_train_mask" in row and "_test_mask" in row
    assert row["train_samples"] >= 5 and row["test_samples"] >= 5
    assert row["train_start"] <= row["train_end"] < row["test_start"] <= row["test_end"]


def test_run_walkforward_research_deterministic_selection_order() -> None:
    candles_df, target = _build_inputs()

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    param_grid: list[dict[str, object]] = [
        {"x": 1},
        {"x": 2},
        {"x": 3},
        {"x": 4},
        {"x": 5},
        {"x": 6},
    ]

    def evaluate_param_combo(
        _fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        lookup = {1: 0.4, 2: 0.5, 3: 0.7, 4: 0.4, 5: 0.5, 6: 0.4}
        return pd.Series([lookup[cast(int, params["x"])]] * 5)

    first = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )
    second = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )

    pd.testing.assert_frame_equal(first.selection_summary_df, second.selection_summary_df)

    fold_0_scores = (
        first.fold_scores_df[first.fold_scores_df["fold_id"] == 0]
        .sort_values("rank")
        .reset_index(drop=True)
    )
    # Raw ranking: x=3 best; x=2 and x=5 tie; x=1, x=4, x=6 tie — ties broken by param_label
    assert fold_0_scores["param_label"].tolist() == ["x=3", "x=2", "x=5", "x=1", "x=4", "x=6"]
    assert fold_0_scores["rank"].tolist() == [1, 2, 3, 4, 5, 6]

    low_tie = fold_0_scores[fold_0_scores["param_label"].isin(["x=1", "x=4", "x=6"])]
    assert low_tie["raw_objective"].nunique() == 1
    assert low_tie["smoothed_objective"].equals(low_tie["raw_objective"])
    assert low_tie.sort_values("rank")["param_label"].tolist() == ["x=1", "x=4", "x=6"]

    selected = first.selection_summary_df.sort_values("fold_id")["selected_feature"].tolist()
    assert selected == ["x=3", "x=3"]


def test_run_walkforward_research_n_jobs_1_and_2_produce_same_fold_scores() -> None:
    """n_jobs=1 (sequential) and n_jobs=2 (parallel) yield the same fold_scores_df."""
    candles_df, target = _build_inputs()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=1,
        objective_metric_name="mean_return",
        min_fold_samples=10,
        n_jobs=1,
    )
    param_grid: list[dict[str, object]] = [{"x": 1}, {"x": 2}]

    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        val = cast(int, params["x"])
        n = len(fold_candles)
        return pd.Series([0.1 * val] * n, index=fold_candles.index)

    report1 = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )
    config2 = dataclasses.replace(config, n_jobs=2)
    report2 = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config2,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )
    a = report1.fold_scores_df.sort_values("param_label").reset_index(drop=True)
    b = report2.fold_scores_df.sort_values("param_label").reset_index(drop=True)
    assert list(a.columns) == list(b.columns)
    pd.testing.assert_series_equal(a["param_label"], b["param_label"])
    pd.testing.assert_series_equal(a["raw_objective"], b["raw_objective"], rtol=1e-9)
    pd.testing.assert_series_equal(a["oos_objective"], b["oos_objective"], rtol=1e-9)
    pd.testing.assert_series_equal(a["smoothed_objective"], b["smoothed_objective"], rtol=1e-9)
    pd.testing.assert_series_equal(a["rank"], b["rank"])


def test_run_walkforward_research_fold_scores_include_selected_long_bin_when_evaluator_returns_tuple() -> None:
    """When evaluator returns (series, {"selected_long_bin": ...}), fold_scores_df has selected_long_bin."""
    candles_df, target = _build_inputs()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=1,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    param_grid: list[dict[str, object]] = [{"x": 1}, {"x": 2}]

    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> tuple[pd.Series, dict[str, object]]:
        x = cast(int, params["x"])
        n = len(fold_candles)
        return (
            pd.Series([0.5 - x * 0.1] * n, index=fold_candles.index),
            {"selected_long_bin": x},
        )

    report = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )
    assert "selected_long_bin" in report.fold_scores_df.columns
    fold0 = report.fold_scores_df[report.fold_scores_df["fold_id"] == 0]
    labels_to_bin = fold0.set_index("param_label")["selected_long_bin"].to_dict()
    assert labels_to_bin.get("x=1") == 1
    assert labels_to_bin.get("x=2") == 2


def test_run_walkforward_research_excludes_folds_below_minimum_samples() -> None:
    candles_df, target = _build_inputs()

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 1),
        test_step=15,
        num_steps=7,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    param_grid: list[dict[str, object]] = [{"x": 1}]

    seen_folds: list[pd.Timestamp] = []

    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        _params: dict[str, object],
    ) -> pd.Series:
        seen_folds.append(fold_candles.index.min())
        return pd.Series([1.0])

    report = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )

    assert report.folds_df["fold_id"].tolist() == [0, 1, 2, 3, 4, 5], (
        "Expected fold 6 to be excluded because test samples are below min_fold_samples"
    )
    assert report.folds_df["test_samples"].tolist() == [15, 15, 15, 15, 15, 14], (
        "Expected retained folds to keep their measured test sample counts"
    )
    assert report.folds_df["test_samples"].min() >= config.min_fold_samples, (
        "Expected every retained fold to satisfy min_fold_samples for test data"
    )
    assert len(seen_folds) == 6, (
        "Expected evaluator to run only for retained folds that satisfy min_fold_samples"
    )


def test_run_walkforward_research_canonical_param_labels_are_order_independent() -> None:
    candles_df, target = _build_inputs()

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=1,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    param_grid_a: list[dict[str, object]] = [
        {"alpha": 1, "beta": 5},
        {"beta": 3, "alpha": 2},
    ]
    param_grid_b: list[dict[str, object]] = [
        {"beta": 5, "alpha": 1},
        {"alpha": 2, "beta": 3},
    ]

    def evaluate_param_combo(
        _fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        return pd.Series([cast(float, params["alpha"]) + cast(float, params["beta"])])

    first = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid_a,
        evaluate_param_combo=evaluate_param_combo,
    )
    second = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid_b,
        evaluate_param_combo=evaluate_param_combo,
    )

    first_labels = first.fold_scores_df.sort_values("rank")["param_label"].tolist()
    second_labels = second.fold_scores_df.sort_values("rank")["param_label"].tolist()

    assert first_labels == second_labels, (
        "Expected canonical param labels to be deterministic regardless of dict insertion order"
    )
    assert first_labels == ["alpha=1|beta=5", "alpha=2|beta=3"], (
        "Expected canonical param labels to use alphabetical key ordering"
    )


def test_run_walkforward_research_enforces_no_lookahead_fold_boundaries() -> None:
    candles_df, target = _build_inputs()

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 1),
        test_step=15,
        num_steps=3,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    param_grid: list[dict[str, object]] = [{"x": 1}, {"x": 2}]
    seen_boundaries: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        seen_boundaries.append((fold_candles.index.min(), fold_candles.index.max()))
        return pd.Series([cast(float, params["x"])])

    report = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )

    assert (report.folds_df["train_end"] < report.folds_df["test_start"]).all()

    expected_boundaries = [
        (row.train_start, row.test_end)
        for row in report.folds_df.itertuples(index=False)
        for _ in param_grid
    ]
    assert seen_boundaries == expected_boundaries


def test_run_walkforward_research_passes_fold_train_end_to_evaluator() -> None:
    candles_df, target = _build_inputs()

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    param_grid: list[dict[str, object]] = [{"x": 1}, {"x": 2}]
    observed_train_ends: list[pd.Timestamp] = []
    observed_fold_maxes: list[pd.Timestamp] = []

    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        _params: dict[str, object],
        *,
        train_end: pd.Timestamp,
    ) -> pd.Series:
        observed_train_ends.append(pd.Timestamp(train_end))
        observed_fold_maxes.append(pd.Timestamp(fold_candles.index.max()))
        return pd.Series(1.0, index=fold_candles.index)

    report = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )

    expected_train_ends = [
        pd.Timestamp(row.train_end)
        for row in report.folds_df.itertuples(index=False)
        for _ in param_grid
    ]
    assert observed_train_ends == expected_train_ends
    assert all(train_end < fold_max for train_end, fold_max in zip(observed_train_ends, observed_fold_maxes))


@pytest.mark.parametrize(
    ("mutate_inputs", "feature_type", "module_name", "param_grid", "error_message"),
    [
        (
            lambda candles_df, target: (candles_df.reset_index(drop=True), target),
            "continuous",
            "demo",
            [{"x": 1}],
            "candles_df must have a DatetimeIndex",
        ),
        (
            lambda candles_df, target: (candles_df, target.reset_index(drop=True)),
            "continuous",
            "demo",
            [{"x": 1}],
            "target must have a DatetimeIndex",
        ),
        (
            lambda candles_df, target: (candles_df, target.shift(1).dropna()),
            "continuous",
            "demo",
            [{"x": 1}],
            "candles_df and target must share the same index",
        ),
        (
            lambda candles_df, target: (candles_df, target),
            "continuous",
            "demo",
            [],
            "param_grid must contain at least one parameter combination",
        ),
        (
            lambda candles_df, target: (candles_df, target),
            "",
            "demo",
            [{"x": 1}],
            "feature_type must be a non-empty string",
        ),
        (
            lambda candles_df, target: (candles_df, target),
            "continuous",
            "",
            [{"x": 1}],
            "module_name must be a non-empty string",
        ),
    ],
)
def test_run_walkforward_research_validates_inputs(
    mutate_inputs: Callable[[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]],
    feature_type: str,
    module_name: str,
    param_grid: list[dict[str, object]],
    error_message: str,
) -> None:
    candles_df, target = _build_inputs()
    candles_df, target = mutate_inputs(candles_df, target)

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 1),
        test_step=15,
        num_steps=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )

    def evaluate_param_combo(
        _fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        _params: dict[str, object],
    ) -> pd.Series:
        return pd.Series([1.0])

    with pytest.raises(ValueError, match=error_message):
        run_walkforward_research(
            candles_df=candles_df,
            target=target,
            feature_type=feature_type,
            module_name=module_name,
            config=config,
            param_grid=param_grid,
            evaluate_param_combo=evaluate_param_combo,
        )


def test_run_walkforward_research_ranks_by_in_sample_objective_when_series_spans_fold() -> None:
    candles_df, target = _build_inputs()

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=1,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    param_grid: list[dict[str, object]] = [{"x": 1}]

    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        _params: dict[str, object],
    ) -> pd.Series:
        in_sample = pd.Series(1.0, index=fold_candles.index[:40])
        out_of_sample = pd.Series(5.0, index=fold_candles.index[40:])
        return pd.concat([in_sample, out_of_sample])

    report = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )

    # Selection/ranking should be driven by in-sample score, not OOS.
    assert report.selection_summary_df.loc[0, "selected_raw_objective"] == pytest.approx(1.0)
    # OOS score is tracked separately for evaluation.
    assert report.fold_scores_df.loc[0, "oos_objective"] == pytest.approx(5.0)


def test_run_walkforward_research_param_label_includes_bin_count_when_present() -> None:
    candles_df, target = _build_inputs()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=1,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    param_grid: list[dict[str, object]] = [
        {"lookback": 4, "bin_count": 8},
        {"lookback": 6, "bin_count": 10},
    ]

    def evaluate_param_combo(
        _fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        return pd.Series([1.0 if int(params["lookback"]) == 6 else 0.5])

    report = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )

    labels = set(report.fold_scores_df["param_label"].tolist())
    assert "bin_count=8|lookback=4" in labels
    assert "bin_count=10|lookback=6" in labels


def test_run_walkforward_research_objective_uses_active_returns_only() -> None:
    candles_df, target = _build_inputs()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=1,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )
    param_grid: list[dict[str, object]] = [{"x": 1}, {"x": 2}]

    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        if int(params["x"]) == 1:
            # Better active returns, but sparse (zeros are flat/no-position).
            train = pd.Series([1.0] + [0.0] * 39, index=fold_candles.index[:40])
        else:
            train = pd.Series([0.5] * 40, index=fold_candles.index[:40])
        test = pd.Series([0.0] * (len(fold_candles.index) - 40), index=fold_candles.index[40:])
        return pd.concat([train, test])

    report = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=param_grid,
        evaluate_param_combo=evaluate_param_combo,
    )

    assert report.selection_summary_df.loc[0, "selected_feature"] == "x=1"


def test_run_walkforward_research_adds_empty_portfolio_results_without_research_config() -> None:
    candles_df, target = _build_inputs()
    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=1,
        objective_metric_name="mean_return",
        min_fold_samples=10,
    )

    report = run_walkforward_research(
        candles_df=candles_df,
        target=target,
        feature_type="continuous",
        module_name="demo",
        config=config,
        param_grid=[{"x": 1}],
        evaluate_param_combo=lambda _candles, _target, _params: pd.Series(0.01, index=candles_df.index),
    )

    assert "oos_portfolio_sharpe" in report.portfolio_results_df.columns
    assert report.portfolio_results_df.empty


def test_run_portfolio_simulation_records_error_without_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    candles_df = pd.DataFrame(
        {
            "datetime": index,
            "open": np.arange(40) + 100.0,
            "high": np.arange(40) + 101.0,
            "low": np.arange(40) + 99.0,
            "close": np.arange(40) + 100.5,
            "ticker": ["ES"] * 40,
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 40), index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "_train_mask": pd.Series([True] * 20 + [False] * 20, index=index),
            "_test_mask": pd.Series([False] * 20 + [True] * 20, index=index),
        }
    ]
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": "lookback=5",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params({"lookback": 5, "bin_count": 4}),
            }
        ]
    )

    def _boom(**_kwargs: object) -> object:
        raise RuntimeError("sim failed")

    monkeypatch.setattr("utils.evaluation.walkforward.portfolio_evaluator.evaluate_fold_portfolio", _boom)

    research_config = type(
        "ResearchCfg",
        (),
        {
            "tickers": [],
            "binning_params": object(),
            "walkforward": type("WF", (), {"objective_metric_name": "sharpe"})(),
            "bias_spec": {"module_name": "rsi", "timeframes": ["D"]},
        },
    )()

    result, _, _ = run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
    )

    assert result.loc[0, "fold_id"] == 0
    assert pd.isna(result.loc[0, "oos_portfolio_sharpe"])
    assert result.loc[0, "error"] == "sim failed"


def test_run_portfolio_simulation_builds_weight_layer_config_from_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    candles_df = pd.DataFrame(
        {
            "datetime": index,
            "open": np.arange(40) + 100.0,
            "high": np.arange(40) + 101.0,
            "low": np.arange(40) + 99.0,
            "close": np.arange(40) + 100.5,
            "ticker": ["ES"] * 40,
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 40), index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "_train_mask": pd.Series([True] * 20 + [False] * 20, index=index),
            "_test_mask": pd.Series([False] * 20 + [True] * 20, index=index),
        }
    ]
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": "lookback=5",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params({"lookback": 5, "bin_count": 4}),
            }
        ]
    )

    captured: dict[str, object] = {}

    def _fake_evaluate(**kwargs: object) -> object:
        captured.update(kwargs)
        return type(
            "FakeResult",
            (),
            {"oos_portfolio_sharpe": 1.23, "n_params_selected": 1},
        )()

    monkeypatch.setattr(
        "utils.evaluation.walkforward.portfolio_evaluator.evaluate_fold_portfolio",
        _fake_evaluate,
    )

    research_config = type(
        "ResearchCfg",
        (),
        {
            "tickers": [],
            "binning_params": object(),
            "walkforward": type(
                "WF",
                (),
                {
                    "objective_metric_name": "sharpe",
                    "weight_layer_algorithm": WeightLayerAlgorithm.EQUAL_SIGNAL,
                    "weight_layer_config": None,
                },
            )(),
            "bias_spec": {"module_name": "rsi", "timeframes": ["D"]},
        },
    )()

    run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
    )

    forwarded = captured.get("weight_layer_config")
    assert isinstance(forwarded, WeightLayerConfig)
    assert forwarded.weighting_method == WeightLayerAlgorithm.EQUAL_SIGNAL.value


def test_run_portfolio_simulation_overrides_weight_layer_method_but_preserves_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    candles_df = pd.DataFrame(
        {
            "datetime": index,
            "open": np.arange(40) + 100.0,
            "high": np.arange(40) + 101.0,
            "low": np.arange(40) + 99.0,
            "close": np.arange(40) + 100.5,
            "ticker": ["ES"] * 40,
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 40), index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "_train_mask": pd.Series([True] * 20 + [False] * 20, index=index),
            "_test_mask": pd.Series([False] * 20 + [True] * 20, index=index),
        }
    ]
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": "lookback=5",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params({"lookback": 5, "bin_count": 4}),
            }
        ]
    )
    hierarchy_spec: dict[str, object] = {
        "type": "group",
        "id": "root",
        "children": [
            {"type": "leaf", "stream_id": "a"},
            {"type": "leaf", "stream_id": "b"},
        ],
    }
    existing = WeightLayerConfig(
        weighting_method=WeightLayerAlgorithm.EQUAL_SIGNAL.value,
        fdm_max=1.8,
        hierarchy_spec=hierarchy_spec,
    )

    captured: dict[str, object] = {}

    def _fake_evaluate(**kwargs: object) -> object:
        captured.update(kwargs)
        return type(
            "FakeResult",
            (),
            {"oos_portfolio_sharpe": 1.23, "n_params_selected": 1},
        )()

    monkeypatch.setattr(
        "utils.evaluation.walkforward.portfolio_evaluator.evaluate_fold_portfolio",
        _fake_evaluate,
    )

    research_config = type(
        "ResearchCfg",
        (),
        {
            "tickers": [],
            "binning_params": object(),
            "walkforward": type(
                "WF",
                (),
                {
                    "objective_metric_name": "sharpe",
                    "weight_layer_algorithm": WeightLayerAlgorithm.HIERARCHY_EQUAL,
                    "weight_layer_config": existing,
                },
            )(),
            "bias_spec": {"module_name": "rsi", "timeframes": ["D"]},
        },
    )()

    run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
    )

    forwarded = captured.get("weight_layer_config")
    assert isinstance(forwarded, WeightLayerConfig)
    assert forwarded.weighting_method == WeightLayerAlgorithm.HIERARCHY_EQUAL.value
    assert forwarded.fdm_max == pytest.approx(1.8)
    assert forwarded.hierarchy_spec == hierarchy_spec


def test_run_portfolio_simulation_handles_invalid_selected_params_json() -> None:
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    candles_df = pd.DataFrame(
        {
            "datetime": index,
            "open": np.arange(40) + 100.0,
            "high": np.arange(40) + 101.0,
            "low": np.arange(40) + 99.0,
            "close": np.arange(40) + 100.5,
            "ticker": ["ES"] * 40,
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 40), index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "_train_mask": pd.Series([True] * 20 + [False] * 20, index=index),
            "_test_mask": pd.Series([False] * 20 + [True] * 20, index=index),
        }
    ]
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": "lookback=5",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": "not-json",
            }
        ]
    )

    research_config = type(
        "ResearchCfg",
        (),
        {
            "tickers": [],
            "binning_params": object(),
            "walkforward": type("WF", (), {"objective_metric_name": "sharpe"})(),
            "bias_spec": {"module_name": "rsi", "timeframes": ["D"]},
        },
    )()

    result, _, _ = run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
    )

    assert result.loc[0, "fold_id"] == 0
    assert pd.isna(result.loc[0, "oos_portfolio_sharpe"])
    assert result.loc[0, "n_params_selected"] == 0
    assert result.loc[0, "error"] == "no_selected_params"


def test_run_portfolio_simulation_forwards_timeframe_to_tearsheets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    candles_df = pd.DataFrame(
        {
            "datetime": index,
            "open": np.arange(6) + 100.0,
            "high": np.arange(6) + 101.0,
            "low": np.arange(6) + 99.0,
            "close": np.arange(6) + 100.5,
            "ticker": ["ES"] * 6,
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 6), index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "_train_mask": pd.Series([True, True, True, False, False, False], index=index),
            "_test_mask": pd.Series([False, False, False, True, True, True], index=index),
        }
    ]
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": "lookback=5",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params({"lookback": 5}),
            }
        ]
    )

    captured_timeframes: list[TimeFrame] = []

    def _fake_generate_tearsheet(**kwargs: object) -> None:
        captured_timeframes.append(kwargs["timeframe"])  # type: ignore[index]

    def _fake_baseline(_candles: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=index)

    def _fake_evaluate(**_kwargs: object) -> object:
        return type(
            "FakeResult",
            (),
            {
                "oos_portfolio_sharpe": 0.42,
                "n_params_selected": 1,
                "per_signal_oos_sharpe": {},
                "oos_portfolio_returns": pd.Series([0.01, -0.01, 0.02], index=index[-3:]),
                "per_signal_oos_returns": {},
                "per_ticker_oos_returns": {
                    "ES": pd.Series([0.01, -0.01, 0.02], index=index[-3:]),
                },
            },
        )()

    monkeypatch.setattr(
        "utils.evaluation.walkforward.portfolio_evaluator.evaluate_fold_portfolio",
        _fake_evaluate,
    )
    monkeypatch.setattr(
        "metrics.plotting.graphing.quantstats_reports.generate_tearsheet",
        _fake_generate_tearsheet,
    )
    monkeypatch.setattr(
        "ensemble.portfolio_impl.portfolio_tester.calculate_baseline_returns",
        _fake_baseline,
    )

    research_config = type(
        "ResearchCfg",
        (),
        {
            "tickers": [],
            "binning_params": object(),
            "walkforward": type("WF", (), {"objective_metric_name": "sharpe"})(),
            "bias_spec": {"module_name": "rsi", "timeframes": [TimeFrame.H4]},
            "generate_ticker_tearsheets": True,
        },
    )()

    run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
        tearsheets_dir=tmp_path / "tearsheets",
    )

    assert all(t == TimeFrame.H4 for t in captured_timeframes) and len(captured_timeframes) >= 1


def test_run_portfolio_simulation_generates_per_fold_ticker_tearsheets_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    candles_df = pd.DataFrame(
        {
            "datetime": index,
            "open": np.arange(6) + 100.0,
            "high": np.arange(6) + 101.0,
            "low": np.arange(6) + 99.0,
            "close": np.arange(6) + 100.5,
            "ticker": ["ES", "NQ", "ES", "NQ", "ES", "NQ"],
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 6), index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "_train_mask": pd.Series([True, True, True, False, False, False], index=index),
            "_test_mask": pd.Series([False, False, False, True, True, True], index=index),
        }
    ]
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": "lookback=5",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params({"lookback": 5}),
            }
        ]
    )
    output_files: list[str] = []

    def _fake_generate_tearsheet(**kwargs: object) -> None:
        output_files.append(str(kwargs["output_file"]))  # type: ignore[index]

    def _fake_baseline(candles: pd.DataFrame) -> pd.Series:
        idx = pd.DatetimeIndex(pd.to_datetime(candles["datetime"]).unique()).sort_values()
        return pd.Series(0.0, index=idx)

    def _fake_evaluate(**_kwargs: object) -> object:
        return type(
            "FakeResult",
            (),
            {
                "oos_portfolio_sharpe": 0.42,
                "n_params_selected": 1,
                "per_signal_oos_sharpe": {},
                "oos_portfolio_returns": pd.Series([0.01, -0.01, 0.02], index=index[-3:]),
                "per_signal_oos_returns": {},
                "per_ticker_oos_returns": {
                    "ES": pd.Series([0.01, 0.02], index=[index[4], index[5]]),
                    "NQ": pd.Series([-0.01], index=[index[3]]),
                },
            },
        )()

    monkeypatch.setattr(
        "utils.evaluation.walkforward.portfolio_evaluator.evaluate_fold_portfolio",
        _fake_evaluate,
    )
    monkeypatch.setattr(
        "metrics.plotting.graphing.quantstats_reports.generate_tearsheet",
        _fake_generate_tearsheet,
    )
    monkeypatch.setattr(
        "ensemble.portfolio_impl.portfolio_tester.calculate_baseline_returns",
        _fake_baseline,
    )

    research_config = type(
        "ResearchCfg",
        (),
        {
            "tickers": [],
            "binning_params": object(),
            "walkforward": type("WF", (), {"objective_metric_name": "sharpe"})(),
            "bias_spec": {"module_name": "rsi", "timeframes": [TimeFrame.D]},
            "generate_ticker_tearsheets": True,
        },
    )()

    run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
        tearsheets_dir=tmp_path / "tearsheets",
    )

    # Single fold: tearsheets written at tearsheets/ root (no fold_0/), clear names.
    assert any("ES_tearsheet.html" in path for path in output_files)
    assert any("NQ_tearsheet.html" in path for path in output_files)
    assert any("train_ensemble_tearsheet.html" in path for path in output_files)
    assert any("validation_ensemble_tearsheet.html" in path for path in output_files)
    assert any("train_and_validation_ensemble_tearsheet.html" in path for path in output_files)


def test_run_portfolio_simulation_generates_aggregate_ticker_tearsheets_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    candles_df = pd.DataFrame(
        {
            "datetime": index,
            "open": np.arange(10) + 100.0,
            "high": np.arange(10) + 101.0,
            "low": np.arange(10) + 99.0,
            "close": np.arange(10) + 100.5,
            "ticker": ["ES", "NQ"] * 5,
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 10), index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "_train_mask": pd.Series([True, True, True, True, False, False, False, False, False, False], index=index),
            "_test_mask": pd.Series([False, False, False, False, True, True, False, False, False, False], index=index),
        },
        {
            "fold_id": 1,
            "_train_mask": pd.Series([True, True, True, True, True, True, False, False, False, False], index=index),
            "_test_mask": pd.Series([False, False, False, False, False, False, True, True, True, True], index=index),
        },
    ]
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": "x=1",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params({"x": 1}),
            },
            {
                "fold_id": 1,
                "selected_feature": "x=1",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params({"x": 1}),
            },
        ]
    )
    output_files: list[str] = []

    def _fake_generate_tearsheet(**kwargs: object) -> None:
        output_files.append(str(kwargs["output_file"]))  # type: ignore[index]

    def _fake_baseline(candles: pd.DataFrame) -> pd.Series:
        idx = pd.DatetimeIndex(pd.to_datetime(candles["datetime"]).unique()).sort_values()
        return pd.Series(0.0, index=idx)

    def _fake_evaluate(**kwargs: object) -> object:
        test_candles = cast(pd.DataFrame, kwargs["test_candles"])
        test_index = pd.DatetimeIndex(pd.to_datetime(test_candles["datetime"]).unique()).sort_values()
        return type(
            "FakeResult",
            (),
            {
                "oos_portfolio_sharpe": 0.33,
                "n_params_selected": 1,
                "per_signal_oos_sharpe": {},
                "oos_portfolio_returns": pd.Series(0.01, index=test_index),
                "per_signal_oos_returns": {},
                "per_ticker_oos_returns": {
                    "ES": pd.Series(0.01, index=test_index),
                    "NQ": pd.Series(-0.005, index=test_index),
                },
            },
        )()

    monkeypatch.setattr(
        "utils.evaluation.walkforward.portfolio_evaluator.evaluate_fold_portfolio",
        _fake_evaluate,
    )
    monkeypatch.setattr(
        "metrics.plotting.graphing.quantstats_reports.generate_tearsheet",
        _fake_generate_tearsheet,
    )
    monkeypatch.setattr(
        "ensemble.portfolio_impl.portfolio_tester.calculate_baseline_returns",
        _fake_baseline,
    )

    research_config = type(
        "ResearchCfg",
        (),
        {
            "tickers": [],
            "binning_params": object(),
            "walkforward": type("WF", (), {"objective_metric_name": "sharpe"})(),
            "bias_spec": {"module_name": "rsi", "timeframes": [TimeFrame.D]},
            "generate_ticker_tearsheets": True,
        },
    )()

    run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
        tearsheets_dir=tmp_path / "tearsheets",
    )

    assert any("walkforward_ES_tearsheet.html" in path for path in output_files)
    assert any("walkforward_NQ_tearsheet.html" in path for path in output_files)


def test_run_portfolio_simulation_skips_ticker_tearsheets_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    candles_df = pd.DataFrame(
        {
            "datetime": index,
            "open": np.arange(10) + 100.0,
            "high": np.arange(10) + 101.0,
            "low": np.arange(10) + 99.0,
            "close": np.arange(10) + 100.5,
            "ticker": ["ES", "NQ"] * 5,
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 10), index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "_train_mask": pd.Series([True, True, True, True, False, False, False, False, False, False], index=index),
            "_test_mask": pd.Series([False, False, False, False, True, True, False, False, False, False], index=index),
        },
        {
            "fold_id": 1,
            "_train_mask": pd.Series([True, True, True, True, True, True, False, False, False, False], index=index),
            "_test_mask": pd.Series([False, False, False, False, False, False, True, True, True, True], index=index),
        },
    ]
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": "x=1",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params({"x": 1}),
            },
            {
                "fold_id": 1,
                "selected_feature": "x=1",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params({"x": 1}),
            },
        ]
    )
    output_files: list[str] = []

    def _fake_generate_tearsheet(**kwargs: object) -> None:
        output_files.append(str(kwargs["output_file"]))  # type: ignore[index]

    def _fake_baseline(candles: pd.DataFrame) -> pd.Series:
        idx = pd.DatetimeIndex(pd.to_datetime(candles["datetime"]).unique()).sort_values()
        return pd.Series(0.0, index=idx)

    def _fake_evaluate(**kwargs: object) -> object:
        test_candles = cast(pd.DataFrame, kwargs["test_candles"])
        test_index = pd.DatetimeIndex(pd.to_datetime(test_candles["datetime"]).unique()).sort_values()
        return type(
            "FakeResult",
            (),
            {
                "oos_portfolio_sharpe": 0.42,
                "n_params_selected": 1,
                "per_signal_oos_sharpe": {},
                "oos_portfolio_returns": pd.Series(0.01, index=test_index),
                "per_signal_oos_returns": {},
                "per_ticker_oos_returns": {},
            },
        )()

    monkeypatch.setattr(
        "utils.evaluation.walkforward.portfolio_evaluator.evaluate_fold_portfolio",
        _fake_evaluate,
    )
    monkeypatch.setattr(
        "metrics.plotting.graphing.quantstats_reports.generate_tearsheet",
        _fake_generate_tearsheet,
    )
    monkeypatch.setattr(
        "ensemble.portfolio_impl.portfolio_tester.calculate_baseline_returns",
        _fake_baseline,
    )

    research_config = type(
        "ResearchCfg",
        (),
        {
            "tickers": [],
            "binning_params": object(),
            "walkforward": type("WF", (), {"objective_metric_name": "sharpe"})(),
            "bias_spec": {"module_name": "rsi", "timeframes": [TimeFrame.D]},
            "generate_ticker_tearsheets": False,
        },
    )()

    run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
        tearsheets_dir=tmp_path / "tearsheets",
    )

    # Ticker tearsheets disabled: no per-fold and no walkforward ticker files.
    assert not any("fold_0/fold_0_ES_tearsheet.html" in path for path in output_files)
    assert not any("fold_0/fold_0_NQ_tearsheet.html" in path for path in output_files)
    assert not any("walkforward_ES_tearsheet.html" in path for path in output_files)
    assert not any("walkforward_NQ_tearsheet.html" in path for path in output_files)
    # Multi-fold: walkforward aggregate is written.
    assert any("walkforward_ensemble_tearsheet.html" in path for path in output_files)


def test_run_portfolio_simulation_passes_decoded_selected_params_to_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    candles_df = pd.DataFrame(
        {
            "datetime": index,
            "open": np.arange(40) + 100.0,
            "high": np.arange(40) + 101.0,
            "low": np.arange(40) + 99.0,
            "close": np.arange(40) + 100.5,
            "ticker": ["ES"] * 40,
        },
        index=index,
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 40), index=index)
    fold_rows = [
        {
            "fold_id": 0,
            "_train_mask": pd.Series([True] * 20 + [False] * 20, index=index),
            "_test_mask": pd.Series([False] * 20 + [True] * 20, index=index),
        }
    ]
    selection_summary_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "selected_feature": "lookback=5",
                "selected_raw_objective": 0.1,
                "selected_smoothed_objective": 0.1,
                "selected_params_json": serialize_selected_params(
                    {"lookback": 5, "bin_count": 4}
                ),
            }
        ]
    )

    captured: dict[str, object] = {}

    def _fake_evaluate(**kwargs: object) -> object:
        captured.update(kwargs)
        selected = kwargs["selected_params"]
        return type(
            "FakeResult",
            (),
            {
                "oos_portfolio_sharpe": 1.23,
                "n_params_selected": len(cast(list[dict[str, object]], selected)),
            },
        )()

    monkeypatch.setattr(
        "utils.evaluation.walkforward.portfolio_evaluator.evaluate_fold_portfolio",
        _fake_evaluate,
    )

    research_config = type(
        "ResearchCfg",
        (),
        {
            "tickers": [],
            "binning_params": object(),
            "walkforward": type("WF", (), {"objective_metric_name": "sharpe"})(),
            "bias_spec": {"module_name": "rsi", "timeframes": ["D"]},
        },
    )()

    result, _, _ = run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        research_config=research_config,
    )

    assert result.loc[0, "error"] == ""
    assert result.loc[0, "n_params_selected"] == 1
    assert captured["selected_params"] == [{"lookback": 5, "bin_count": 4}]


def test_evaluate_fold_portfolio_returns_per_ticker_oos_returns_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    train_candles = pd.DataFrame(
        {
            "datetime": index[:3],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "ticker": ["ES", "NQ", "ES"],
        }
    )
    test_candles = pd.DataFrame(
        {
            "datetime": index[3:],
            "open": [103.0, 104.0, 105.0],
            "high": [104.0, 105.0, 106.0],
            "low": [102.0, 103.0, 104.0],
            "close": [103.5, 104.5, 105.5],
            "ticker": ["NQ", "ES", "NQ"],
        }
    )
    target = pd.Series(np.linspace(-0.01, 0.01, 6), index=index)
    feature_data_by_combo = {
        tuple(sorted({"lookback": 5}.items())): pd.DataFrame(
            {"signal": pd.Series([0.1] * len(index), index=index), "target": target}
        ),
    }

    result = evaluate_fold_portfolio(
        train_candles=train_candles,
        test_candles=test_candles,
        selected_params=[{"lookback": 5}],
        target_series=target,
        binning_config=object(),
        tickers=[Ticker.ES, Ticker.NQ],
        trading_timeframe=TimeFrame.D,
        feature_data_by_combo=feature_data_by_combo,
    )

    assert result.per_ticker_oos_returns is not None
    assert set(result.per_ticker_oos_returns.keys()) == {"ES", "NQ"}
    assert all(
        isinstance(ser.index, pd.DatetimeIndex)
        and not ser.empty
        for ser in result.per_ticker_oos_returns.values()
    )
