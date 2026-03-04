from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd


def plot_fold_timeline(folds_df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    segment_order = pd.CategoricalDtype(categories=["train", "test"], ordered=True)
    normalized_frame = (
        pd.concat(
            [
                folds_df[["fold_id", "train_start", "train_end"]].rename(
                    columns={"train_start": "start", "train_end": "end"}
                ).assign(segment="train"),
                folds_df[["fold_id", "test_start", "test_end"]].rename(
                    columns={"test_start": "start", "test_end": "end"}
                ).assign(segment="test"),
            ],
            ignore_index=True,
        )
        .assign(segment=lambda frame: frame["segment"].astype(segment_order))
        [["fold_id", "segment", "start", "end"]]
        .sort_values(by=["fold_id", "segment"], ascending=[True, True], kind="mergesort")
        .assign(segment=lambda frame: frame["segment"].astype(str))
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    for row in normalized_frame.itertuples(index=False):
        ax.plot([row.start, row.end], [row.fold_id, row.fold_id], linewidth=8, label=row.segment)

    ax.set_title("Walkforward Fold Timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fold ID")
    fig.autofmt_xdate()
    return fig, normalized_frame


def _resolve_selected_rank(top_k_features: str, selected_feature: str, top_k: int) -> int:
    try:
        decoded = json.loads(top_k_features)
    except (TypeError, json.JSONDecodeError):
        return top_k + 1
    if not isinstance(decoded, list):
        return top_k + 1
    ranked = [str(value) for value in decoded[:top_k]]
    if selected_feature in ranked:
        return ranked.index(selected_feature) + 1
    return top_k + 1


def plot_selection_stability(
    selection_summary_df: pd.DataFrame,
    top_k: int,
) -> tuple[plt.Figure, pd.DataFrame]:
    summary_frame = (
        selection_summary_df.assign(
            selected_rank=lambda frame: frame.apply(
                lambda row: _resolve_selected_rank(
                    top_k_features=str(row["top_k_features"]),
                    selected_feature=str(row["selected_feature"]),
                    top_k=top_k,
                ),
                axis=1,
            )
        )
        [[
            "fold_id",
            "selected_feature",
            "selected_rank",
            "selected_smoothed_objective",
        ]]
        .sort_values(by=["fold_id", "selected_feature"], ascending=[True, True], kind="mergesort")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(summary_frame["fold_id"], summary_frame["selected_smoothed_objective"])
    ax.set_title("Walkforward Selection Stability")
    ax.set_xlabel("Fold ID")
    ax.set_ylabel("Selected Smoothed Objective")
    return fig, summary_frame
