from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
from typing import Callable, cast

import pandas as pd

from feature_research.walkforward.config import WalkforwardResearchConfig
from feature_research.walkforward.metrics import resolve_objective_metric
from utils.grid_smoothing import add_smoothed_objective


@dataclass(frozen=True)
class FoldScoreRow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    param_label: str
    raw_objective: float
    smoothed_objective: float
    rank: int


@dataclass(frozen=True)
class WalkforwardRunReport:
    folds_df: pd.DataFrame
    fold_scores_df: pd.DataFrame
    selection_summary_df: pd.DataFrame


def _canonical_param_label(params: dict[str, object]) -> str:
    return "|".join(
        f"{key}={params[key]}" for key in sorted(params)
    )


def _build_fold_rows(
    datetime_index: pd.DatetimeIndex,
    config: WalkforwardResearchConfig,
) -> list[dict[str, object]]:
    fold_rows: list[dict[str, object]] = []

    for fold_id in range(config.num_steps):
        offset = timedelta(days=fold_id * config.test_step)
        train_start_boundary = pd.Timestamp(config.train_start + offset)
        train_end_boundary = pd.Timestamp(config.train_end + offset)
        test_start_boundary = train_end_boundary
        test_end_boundary = pd.Timestamp(test_start_boundary + timedelta(days=config.test_step))

        train_mask = (datetime_index >= train_start_boundary) & (datetime_index < train_end_boundary)
        test_mask = (datetime_index >= test_start_boundary) & (datetime_index < test_end_boundary)

        train_index = datetime_index[train_mask]
        test_index = datetime_index[test_mask]
        train_samples = int(train_index.size)
        test_samples = int(test_index.size)

        if train_samples < config.min_fold_samples or test_samples < config.min_fold_samples:
            continue

        train_start = pd.Timestamp(train_index.min())
        train_end = pd.Timestamp(train_index.max())
        test_start = pd.Timestamp(test_index.min())
        test_end = pd.Timestamp(test_index.max())
        if train_end >= test_start:
            raise ValueError("No-lookahead violation: train_end must be less than test_start")

        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_samples": train_samples,
                "test_samples": test_samples,
                "_train_mask": train_mask,
                "_test_mask": test_mask,
            }
        )

    return fold_rows


def _build_fold_scores(
    fold_row: dict[str, object],
    candles_df: pd.DataFrame,
    target: pd.Series,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series],
    objective_metric: Callable[[pd.Series], float],
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_mask = cast(pd.Series, fold_row["_train_mask"])
    test_mask = cast(pd.Series, fold_row["_test_mask"])

    combined_mask = train_mask | test_mask
    fold_candles = candles_df.loc[combined_mask]
    fold_target = target.loc[combined_mask]
    test_index = candles_df.index[test_mask]

    def score_param(params: dict[str, object]) -> float:
        scored_returns = evaluate_param_combo(fold_candles, fold_target, params)
        if isinstance(scored_returns.index, pd.DatetimeIndex):
            scored_returns = scored_returns.loc[scored_returns.index.isin(test_index)]
        elif len(scored_returns) > len(test_index):
            scored_returns = scored_returns.tail(len(test_index))
        return float(objective_metric(scored_returns))

    param_columns = sorted({key for params in param_grid for key in params})
    raw_rows = [
        {
            **{column: params.get(column) for column in param_columns},
            "param_label": _canonical_param_label(params),
            "raw_objective": score_param(params),
        }
        for params in param_grid
    ]

    raw_df = pd.DataFrame(raw_rows)
    smoothed_df = add_smoothed_objective(
        raw_df,
        param_columns=param_columns,
        objective_column="raw_objective",
        output_column="smoothed_objective",
    )

    ranked_df = (
        smoothed_df.sort_values(
            by=["smoothed_objective", "raw_objective", "param_label"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    ranked_df["rank"] = ranked_df.index + 1

    scored_rows = [
        FoldScoreRow(
            fold_id=int(cast(int, fold_row["fold_id"])),
            train_start=cast(pd.Timestamp, fold_row["train_start"]),
            train_end=cast(pd.Timestamp, fold_row["train_end"]),
            test_start=cast(pd.Timestamp, fold_row["test_start"]),
            test_end=cast(pd.Timestamp, fold_row["test_end"]),
            param_label=str(row.param_label),
            raw_objective=float(row.raw_objective),
            smoothed_objective=float(row.smoothed_objective),
            rank=int(row.rank),
        )
        for row in ranked_df.itertuples(index=False)
    ]

    selected_feature = scored_rows[0].param_label
    selected_row = scored_rows[0]
    fold_scores_df = pd.DataFrame(
        [
            {
                "fold_id": row.fold_id,
                "param_label": row.param_label,
                "raw_objective": row.raw_objective,
                "smoothed_objective": row.smoothed_objective,
                "rank": row.rank,
                "selected_feature": row.param_label == selected_feature,
            }
            for row in scored_rows
        ]
    )

    top_k_features = ranked_df["param_label"].head(top_k).tolist()
    summary_row = {
        "fold_id": int(cast(int, fold_row["fold_id"])),
        "selected_feature": selected_feature,
        "selected_raw_objective": selected_row.raw_objective,
        "selected_smoothed_objective": selected_row.smoothed_objective,
        "top_k_features": json.dumps(top_k_features, separators=(",", ":"), ensure_ascii=True),
    }
    return fold_scores_df, summary_row


def run_walkforward_research(
    candles_df: pd.DataFrame,
    target: pd.Series,
    feature_type: str,
    module_name: str,
    config: WalkforwardResearchConfig,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series],
) -> WalkforwardRunReport:
    if not isinstance(feature_type, str) or not feature_type.strip():
        raise ValueError("feature_type must be a non-empty string")
    if not isinstance(module_name, str) or not module_name.strip():
        raise ValueError("module_name must be a non-empty string")

    if not isinstance(candles_df.index, pd.DatetimeIndex):
        raise ValueError("candles_df must have a DatetimeIndex")
    if not isinstance(target.index, pd.DatetimeIndex):
        raise ValueError("target must have a DatetimeIndex")
    if not candles_df.index.equals(target.index):
        raise ValueError("candles_df and target must share the same index")
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter combination")

    objective_metric = resolve_objective_metric(config.objective_metric_name)
    fold_rows = _build_fold_rows(candles_df.index, config)

    folds_df = pd.DataFrame(
        [
            {
                "fold_id": row["fold_id"],
                "train_start": row["train_start"],
                "train_end": row["train_end"],
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "train_samples": row["train_samples"],
                "test_samples": row["test_samples"],
            }
            for row in fold_rows
        ],
        columns=[
            "fold_id",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "train_samples",
            "test_samples",
        ],
    )

    fold_score_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    for fold_row in fold_rows:
        fold_scores_df, summary_row = _build_fold_scores(
            fold_row=fold_row,
            candles_df=candles_df,
            target=target,
            param_grid=param_grid,
            evaluate_param_combo=evaluate_param_combo,
            objective_metric=objective_metric,
            top_k=config.top_k,
        )
        fold_score_parts.append(fold_scores_df)
        selection_rows.append(summary_row)

    fold_scores_df = (
        pd.concat(fold_score_parts, ignore_index=True)
        if fold_score_parts
        else pd.DataFrame(
            columns=[
                "fold_id",
                "param_label",
                "raw_objective",
                "smoothed_objective",
                "rank",
                "selected_feature",
            ]
        )
    )
    selection_summary_df = pd.DataFrame(
        selection_rows,
        columns=[
            "fold_id",
            "selected_feature",
            "selected_raw_objective",
            "selected_smoothed_objective",
            "top_k_features",
        ],
    )

    return WalkforwardRunReport(
        folds_df=folds_df,
        fold_scores_df=fold_scores_df,
        selection_summary_df=selection_summary_df,
    )
