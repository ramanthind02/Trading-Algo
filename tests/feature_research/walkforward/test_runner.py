from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Callable, cast

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_research.walkforward.config import WalkforwardResearchConfig
from feature_research.walkforward.runner import run_walkforward_research


def _build_inputs() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    candles_df = pd.DataFrame({"close": range(120)}, index=index)
    target = pd.Series(0.01, index=index, name="target")
    return candles_df, target


def test_run_walkforward_research_deterministic_selection_order() -> None:
    candles_df, target = _build_inputs()

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=2,
        top_k=2,
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
    assert fold_0_scores["param_label"].tolist() == ["x=3", "x=2", "x=4", "x=1", "x=6", "x=5"]
    assert fold_0_scores["rank"].tolist() == [1, 2, 3, 4, 5, 6]

    smoothed_tie = fold_0_scores[fold_0_scores["param_label"].isin(["x=2", "x=4"])]
    assert smoothed_tie["smoothed_objective"].nunique() == 1, (
        "Expected a true smoothed-objective tie for x=2 and x=4"
    )
    assert smoothed_tie.sort_values("rank")["param_label"].tolist() == ["x=2", "x=4"], (
        "Expected raw_objective descending to break smoothed tie between x=2 and x=4"
    )

    full_tie = fold_0_scores[fold_0_scores["param_label"].isin(["x=1", "x=6"])]
    assert full_tie["smoothed_objective"].nunique() == 1 and full_tie["raw_objective"].nunique() == 1, (
        "Expected a true smoothed/raw tie for x=1 and x=6"
    )
    assert full_tie.sort_values("rank")["param_label"].tolist() == ["x=1", "x=6"], (
        "Expected param_label ascending to break full tie between x=1 and x=6"
    )

    selected = first.selection_summary_df.sort_values("fold_id")["selected_feature"].tolist()
    assert selected == ["x=3", "x=3"]


def test_run_walkforward_research_excludes_folds_below_minimum_samples() -> None:
    candles_df, target = _build_inputs()

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 1),
        test_step=15,
        num_steps=7,
        top_k=1,
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
        top_k=2,
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
        top_k=2,
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
        top_k=1,
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


def test_run_walkforward_research_scores_only_out_of_sample_segment_when_series_spans_fold() -> None:
    candles_df, target = _build_inputs()

    config = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        test_step=20,
        num_steps=1,
        top_k=1,
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

    assert report.selection_summary_df.loc[0, "selected_raw_objective"] == pytest.approx(5.0)
