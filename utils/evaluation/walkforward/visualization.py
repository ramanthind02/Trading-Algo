from __future__ import annotations

import json

import pandas as pd


def plot_fold_timeline(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized fold timeline table for downstream reporting."""
    segment_order = pd.CategoricalDtype(categories=["train", "test"], ordered=True)
    return (
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
) -> pd.DataFrame:
    return (
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
