"""Tabular transforms for walkforward reporting tests (no production dependency)."""

from __future__ import annotations

import pandas as pd


def plot_fold_timeline(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized fold timeline table for downstream reporting."""
    segment_order = pd.CategoricalDtype(categories=["train", "test"], ordered=True)
    return (
        pd.concat(
            [
                folds_df[["fold_id", "train_start", "train_end"]]
                .rename(columns={"train_start": "start", "train_end": "end"})
                .assign(segment="train"),
                folds_df[["fold_id", "test_start", "test_end"]]
                .rename(columns={"test_start": "start", "test_end": "end"})
                .assign(segment="test"),
            ],
            ignore_index=True,
        )
        .assign(segment=lambda frame: frame["segment"].astype(segment_order))
        [["fold_id", "segment", "start", "end"]]
        .sort_values(by=["fold_id", "segment"], ascending=[True, True], kind="mergesort")
        .assign(segment=lambda frame: frame["segment"].astype(str))
        .reset_index(drop=True)
    )


def plot_selection_stability(selection_summary_df: pd.DataFrame) -> pd.DataFrame:
    """One winner per fold: ``selected_rank`` is always 1."""
    return (
        selection_summary_df.assign(selected_rank=1)[
            [
                "fold_id",
                "selected_feature",
                "selected_rank",
                "selected_smoothed_objective",
            ]
        ]
        .sort_values(by=["fold_id", "selected_feature"], ascending=[True, True], kind="mergesort")
        .reset_index(drop=True)
    )
