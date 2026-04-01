from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.evaluation.walkforward.visualization import (
    plot_fold_timeline,
    plot_selection_stability,
)


def test_plot_fold_timeline_returns_deterministic_plot_frame() -> None:
    folds_df = pd.DataFrame(
        {
            "fold_id": [1, 0],
            "train_start": [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-01-01")],
            "train_end": [pd.Timestamp("2020-02-10"), pd.Timestamp("2020-01-10")],
            "test_start": [pd.Timestamp("2020-02-11"), pd.Timestamp("2020-01-11")],
            "test_end": [pd.Timestamp("2020-02-20"), pd.Timestamp("2020-01-20")],
            "train_samples": [10, 10],
            "test_samples": [10, 10],
        }
    )

    _, frame_first = plot_fold_timeline(folds_df)
    _, frame_second = plot_fold_timeline(folds_df)

    pd.testing.assert_frame_equal(frame_first, frame_second)
    assert frame_first.columns.tolist() == ["fold_id", "segment", "start", "end"]
    assert frame_first["fold_id"].tolist() == [0, 0, 1, 1]
    assert frame_first["segment"].tolist() == ["train", "test", "train", "test"]


def test_plot_selection_stability_returns_expected_columns() -> None:
    selection_summary_df = pd.DataFrame(
        {
            "fold_id": [2, 0, 1],
            "selected_feature": ["x=3", "x=1", "x=2"],
            "selected_raw_objective": [0.5, 0.7, 0.6],
            "selected_smoothed_objective": [0.55, 0.75, 0.65],
            "top_k_features": [
                json.dumps(["x=1", "x=2", "x=3"]),
                json.dumps(["x=1", "x=2", "x=3"]),
                json.dumps(["x=2", "x=1", "x=3"]),
            ],
        }
    )

    _, frame = plot_selection_stability(selection_summary_df, top_k=3)

    assert frame.columns.tolist() == [
        "fold_id",
        "selected_feature",
        "selected_rank",
        "selected_smoothed_objective",
    ]
    assert frame["fold_id"].tolist() == [0, 1, 2]
    assert frame["selected_rank"].tolist() == [1, 1, 3]


def test_plot_selection_stability_handles_malformed_top_k_features_json() -> None:
    selection_summary_df = pd.DataFrame(
        {
            "fold_id": [0, 1],
            "selected_feature": ["x=1", "x=2"],
            "selected_raw_objective": [0.7, 0.6],
            "selected_smoothed_objective": [0.75, 0.65],
            "top_k_features": ["not-json", json.dumps({"x": 1})],
        }
    )

    _, frame = plot_selection_stability(selection_summary_df, top_k=2)

    assert frame["fold_id"].tolist() == [0, 1]
    assert frame["selected_rank"].tolist() == [3, 3]
