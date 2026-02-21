from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import inspect
import json
from typing import Any, Callable, cast

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
    oos_objective: float
    smoothed_objective: float
    rank: int


@dataclass(frozen=True)
class WalkforwardRunReport:
    folds_df: pd.DataFrame
    fold_scores_df: pd.DataFrame
    selection_summary_df: pd.DataFrame
    portfolio_results_df: pd.DataFrame


def _empty_portfolio_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["fold_id", "oos_portfolio_sharpe", "n_params_selected", "error"]
    )


def _coerce_param_value(value: str) -> object:
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _parse_top_k_param_labels(top_k_features: str) -> list[dict[str, object]]:
    try:
        raw_labels = json.loads(top_k_features)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(raw_labels, list):
        return []

    parsed: list[dict[str, object]] = []
    for label in raw_labels:
        if not isinstance(label, str) or not label:
            continue
        parts = [part for part in label.split("|") if "=" in part]
        parsed.append(
            {
                key.strip(): _coerce_param_value(value.strip())
                for key, value in (part.split("=", 1) for part in parts)
                if key.strip()
            }
        )
    return parsed


def run_portfolio_simulation(
    candles_df: pd.DataFrame,
    target: pd.Series,
    fold_rows: list[dict[str, object]],
    selection_summary_df: pd.DataFrame,
    research_config: Any,
) -> pd.DataFrame:
    from feature_research.walkforward.portfolio_evaluator import (
        ensure_portfolio_candle_columns,
        evaluate_fold_portfolio,
    )

    if "datetime" in candles_df.columns:
        all_datetimes = pd.to_datetime(candles_df["datetime"], utc=False)
    else:
        all_datetimes = pd.DatetimeIndex(candles_df.index)
    if getattr(all_datetimes, "tz", None) is not None:
        all_datetimes = all_datetimes.tz_localize(None)

    trading_timeframe = research_config.bias_spec.get("timeframes", [None])[0]
    objective_metric_name = research_config.walkforward.objective_metric_name
    rows: list[dict[str, object]] = []

    for fold_row in fold_rows:
        fold_id = int(cast(int, fold_row["fold_id"]))
        summary = selection_summary_df.loc[selection_summary_df["fold_id"] == fold_id]
        if summary.empty:
            rows.append(
                {
                    "fold_id": fold_id,
                    "oos_portfolio_sharpe": float("nan"),
                    "n_params_selected": 0,
                    "error": "missing_selection_summary",
                }
            )
            continue

        selected_params = _parse_top_k_param_labels(str(summary.iloc[0]["top_k_features"]))
        if not selected_params:
            rows.append(
                {
                    "fold_id": fold_id,
                    "oos_portfolio_sharpe": float("nan"),
                    "n_params_selected": 0,
                    "error": "no_selected_params",
                }
            )
            continue

        if all(key in fold_row for key in ("train_start", "train_end", "test_start", "test_end")):
            train_start = pd.Timestamp(fold_row["train_start"])
            train_end = pd.Timestamp(fold_row["train_end"])
            test_start = pd.Timestamp(fold_row["test_start"])
            test_end = pd.Timestamp(fold_row["test_end"])
            train_mask = (all_datetimes >= train_start) & (all_datetimes <= train_end)
            test_mask = (all_datetimes >= test_start) & (all_datetimes <= test_end)
        else:
            train_mask = cast(pd.Series, fold_row["_train_mask"])
            test_mask = cast(pd.Series, fold_row["_test_mask"])
        train_candles = candles_df.loc[train_mask].copy()
        test_candles = candles_df.loc[test_mask].copy()
        train_candles = ensure_portfolio_candle_columns(train_candles, trading_timeframe)
        test_candles = ensure_portfolio_candle_columns(test_candles, trading_timeframe)

        try:
            wf_cfg = getattr(research_config, "walkforward", None)
            weight_layer_config = getattr(wf_cfg, "weight_layer_config", None)
            result = evaluate_fold_portfolio(
                train_candles=train_candles,
                test_candles=test_candles,
                selected_params=selected_params,
                target_series=target,
                binning_config=research_config.binning_params,
                tickers=research_config.tickers,
                trading_timeframe=trading_timeframe,
                module_name=str(research_config.bias_spec.get("module_name", "rsi")),
                objective_metric_name=objective_metric_name,
                weight_layer_config=weight_layer_config,
            )
            rows.append(
                {
                    "fold_id": fold_id,
                    "oos_portfolio_sharpe": result.oos_portfolio_sharpe,
                    "n_params_selected": result.n_params_selected,
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive catch
            rows.append(
                {
                    "fold_id": fold_id,
                    "oos_portfolio_sharpe": float("nan"),
                    "n_params_selected": len(selected_params),
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows, columns=["fold_id", "oos_portfolio_sharpe", "n_params_selected", "error"])


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
    evaluate_param_combo: Callable[..., pd.Series],
    objective_metric: Callable[[pd.Series], float],
    top_k: int,
    config: WalkforwardResearchConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_mask = cast(pd.Series, fold_row["_train_mask"])
    test_mask = cast(pd.Series, fold_row["_test_mask"])

    combined_mask = train_mask | test_mask
    fold_candles = candles_df.loc[combined_mask]
    fold_target = target.loc[combined_mask]
    train_index = candles_df.index[train_mask]
    test_index = candles_df.index[test_mask]

    def _call_evaluator(
        fold_data: pd.DataFrame,
        fold_targets: pd.Series,
        params: dict[str, object],
        train_end: pd.Timestamp,
    ) -> pd.Series:
        signature = inspect.signature(evaluate_param_combo)
        accepts_train_end = "train_end" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_train_end:
            return evaluate_param_combo(
                fold_data,
                fold_targets,
                params,
                train_end=train_end,
            )
        return evaluate_param_combo(fold_data, fold_targets, params)

    def score_param(params: dict[str, object]) -> tuple[float, float]:
        scored_returns = _call_evaluator(
            fold_data=fold_candles,
            fold_targets=fold_target,
            params=params,
            train_end=cast(pd.Timestamp, fold_row["train_end"]),
        )

        if isinstance(scored_returns.index, pd.DatetimeIndex):
            train_returns = scored_returns.loc[scored_returns.index.isin(train_index)]
            test_returns = scored_returns.loc[scored_returns.index.isin(test_index)]
        else:
            train_len = len(train_index)
            test_len = len(test_index)
            train_returns = scored_returns.head(train_len)
            test_returns = scored_returns.tail(test_len)

        active_train_returns = train_returns[train_returns != 0]
        active_test_returns = test_returns[test_returns != 0]

        return (
            float(objective_metric(active_train_returns)),
            float(objective_metric(active_test_returns)),
        )

    param_columns = sorted({key for params in param_grid for key in params})
    raw_rows: list[dict[str, object]] = []
    for params in param_grid:
        train_objective, oos_objective = score_param(params)
        raw_rows.append(
            {
                **{column: params.get(column) for column in param_columns},
                "param_label": _canonical_param_label(params),
                "raw_objective": train_objective,
                "oos_objective": oos_objective,
            }
        )

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
            oos_objective=float(row.oos_objective),
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
                "oos_objective": float(row.oos_objective),
                "smoothed_objective": row.smoothed_objective,
                "rank": row.rank,
                "selected_feature": row.param_label == selected_feature,
            }
            for row in scored_rows
        ]
    )

    effective_selection_method = config._effective_selection_method()

    if effective_selection_method in ("enhanced", "stable_region"):
        from feature_research.walkforward.top_k_selection import (
            compute_all_trade_frequencies,
            run_enhanced_selection,
        )

        train_candles = candles_df.loc[train_mask]
        train_target = target.loc[train_mask]
        fold_train_end = cast(pd.Timestamp, fold_row["train_end"])

        def evaluate_training_param_combo(
            training_data: pd.DataFrame,
            training_target: pd.Series,
            params: dict[str, object],
        ) -> pd.Series:
            return _call_evaluator(
                fold_data=training_data,
                fold_targets=training_target,
                params=params,
                train_end=fold_train_end,
            )

        # Shared trade-frequency computation (avoids double evaluation)
        trade_frequencies = compute_all_trade_frequencies(
            training_data=train_candles,
            training_target=train_target,
            param_grid=param_grid,
            evaluate_param_combo=evaluate_training_param_combo,
        )
        # Apply trade_freq_min hard filter for stable region input
        trade_frequencies_filtered = {
            label: freq
            for label, freq in trade_frequencies.items()
            if freq >= config.trade_freq_min
        }

        smoothed_obj_map = {
            str(row.param_label): float(row.smoothed_objective)
            for row in smoothed_df.itertuples(index=False)
        }
        raw_obj_map = {
            str(row.param_label): float(row.raw_objective)
            for row in raw_df.itertuples(index=False)
        }

        if effective_selection_method == "enhanced":
            enhanced_result = run_enhanced_selection(
                training_data=train_candles,
                training_target=train_target,
                param_grid=param_grid,
                evaluate_param_combo=evaluate_training_param_combo,
                smoothed_objectives=smoothed_obj_map,
                config=config,
                precomputed_trade_frequencies=trade_frequencies,
            )
            top_k_features = enhanced_result.selected_labels
            fold_scores_df = fold_scores_df.assign(
                trade_frequency=fold_scores_df["param_label"].map(trade_frequencies),
                selected_in_top_k=fold_scores_df["param_label"].isin(enhanced_result.selected_labels),
            )
        else:  # stable_region
            from feature_research.walkforward.stable_region_selection import (
                StableRegionConfig,
                run_stable_region_selection,
            )
            stable_cfg: StableRegionConfig = (
                config.stable_region
                if isinstance(config.stable_region, StableRegionConfig)
                else StableRegionConfig()
            )
            stable_result = run_stable_region_selection(
                smoothed_objectives=smoothed_obj_map,
                raw_objectives=raw_obj_map,
                trade_frequencies=trade_frequencies_filtered,
                param_grid=param_grid,
                config=stable_cfg,
            )
            top_k_features = stable_result.selected_labels
            # Merge per-param detail into fold_scores_df
            detail = stable_result.per_param_detail.set_index("param_label")
            _map = lambda col: (  # noqa: E731
                fold_scores_df["param_label"].map(detail[col].to_dict())
                if col in detail.columns else float("nan")
            )
            fold_scores_df = fold_scores_df.assign(
                trade_frequency=fold_scores_df["param_label"].map(trade_frequencies),
                selected_in_top_k=fold_scores_df["param_label"].isin(stable_result.selected_labels),
                above_floor=_map("above_floor"),
                region_id=_map("region_id"),
                region_size=_map("region_size"),
            )
    else:
        top_k_features = ranked_df["param_label"].head(top_k).tolist()
        fold_scores_df = fold_scores_df.assign(
            trade_frequency=float("nan"),
            selected_in_top_k=False,
        )

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
    evaluate_param_combo: Callable[..., pd.Series],
    research_config: Any | None = None,
    portfolio_candles_df: pd.DataFrame | None = None,
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
            config=config,
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
                "oos_objective",
                "smoothed_objective",
                "rank",
                "selected_feature",
                "trade_frequency",
                "selected_in_top_k",
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

    if research_config is None:
        portfolio_results_df = _empty_portfolio_results_df()
    else:
        try:
            portfolio_results_df = run_portfolio_simulation(
                candles_df=portfolio_candles_df if portfolio_candles_df is not None else candles_df,
                target=target,
                fold_rows=fold_rows,
                selection_summary_df=selection_summary_df,
                research_config=research_config,
            )
        except Exception as exc:  # pragma: no cover - defensive catch
            portfolio_results_df = pd.DataFrame(
                [
                    {
                        "fold_id": int(cast(int, row["fold_id"])),
                        "oos_portfolio_sharpe": float("nan"),
                        "n_params_selected": 0,
                        "error": str(exc),
                    }
                    for row in fold_rows
                ],
                columns=["fold_id", "oos_portfolio_sharpe", "n_params_selected", "error"],
            )

    return WalkforwardRunReport(
        folds_df=folds_df,
        fold_scores_df=fold_scores_df,
        selection_summary_df=selection_summary_df,
        portfolio_results_df=portfolio_results_df,
    )
