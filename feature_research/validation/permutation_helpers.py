from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_research.config import FeatureType
from feature_research.core_helpers import expand_params_with_selected_bin
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    get_available_date_ranges_for_tickers,
    get_tickers_with_coverage_for_config,
    load_candles_for_config,
    populate_cache_if_needed,
)
from utils.evaluation.walkforward.config import WalkforwardResearchConfig
from utils.evaluation.walkforward.evaluators import (
    build_continuous_walkforward_evaluator,
    build_rule_based_walkforward_evaluator,
)
from utils.evaluation.walkforward.metrics import resolve_objective_metric
from utils.evaluation.walkforward.permutation_helpers import (
    _joblib_tqdm,
    aggregate_oos_metric_from_report,
)
from utils.evaluation.walkforward.research_data import (
    build_reference_target,
    load_continuous_research_data,
    load_portfolio_candles,
    load_rule_based_research_data,
)
from utils.evaluation.walkforward.runner import run_walkforward_research


def load_research_data(config: object) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[dict],
    object,
    object | None,
    dict | None,
    pd.DataFrame | None,
]:
    """Build candles, target, param_grid, evaluator, research_config, combo features, portfolio candles."""
    original_tickers = list(config.tickers)
    covered_tickers = get_tickers_with_coverage_for_config(config)
    if len(covered_tickers) < len(original_tickers):
        dropped = set(original_tickers) - set(covered_tickers)
        print(
            f"[permutation] Tickers without full date coverage for "
            f"{config.start.date()}–{config.end.date()} dropped: {[t.name for t in dropped]}"
        )
    config = replace(config, tickers=covered_tickers)
    if not config.tickers:
        ranges = get_available_date_ranges_for_tickers(
            replace(config, tickers=original_tickers), original_tickers
        )
        hint = (
            " Available ranges: "
            + ", ".join(
                f"{t.name}: {r[0].date()}–{r[1].date()}" for t, r in ranges.items()
            )
            + ". Narrow config.start/end or oos_window.test_end to match."
            if ranges
            else " Check data/ohlc_data or narrow config.start/end."
        )
        raise ValueError(
            "No tickers have OHLC data covering the config date range."
            + hint
        )

    populate_cache_if_needed(config)
    expanded = expand_bias_specs(config.bias_spec)
    feature_type = getattr(config, "feature_type", FeatureType.CONTINUOUS)

    if feature_type == FeatureType.CONTINUOUS:
        continuous_data = load_continuous_research_data(config, expanded)
        if not continuous_data.successful_param_grid or continuous_data.reference_index is None:
            raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

        successful_param_grid = expand_params_with_selected_bin(
            continuous_data.successful_param_grid,
            bin_index_min=config.binning_params.bin_index_min,
            bin_index_max=config.binning_params.bin_index_max,
        )
        reference_target = build_reference_target(
            continuous_data.reference_index,
            continuous_data.reference_target_series,
        )
        reference_candles = pd.DataFrame(
            {"close": reference_target}, index=continuous_data.reference_index
        )
        portfolio_candles = load_portfolio_candles(config)
        evaluator = build_continuous_walkforward_evaluator(
            continuous_data.combo_feature_target, config
        )
        return (
            reference_candles,
            reference_target,
            successful_param_grid,
            evaluator,
            config,
            continuous_data.combo_feature_target,
            portfolio_candles,
        )

    rule_data = load_rule_based_research_data(
        config,
        expanded,
        dedupe_before_multiply=False,
        capture_target_as_reference=True,
    )
    if not rule_data.successful_param_grid or rule_data.reference_index is None:
        raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

    reference_target = build_reference_target(
        rule_data.reference_index,
        rule_data.reference_target_series,
    )
    reference_candles = pd.DataFrame({"close": reference_target}, index=rule_data.reference_index)
    evaluator = build_rule_based_walkforward_evaluator(rule_data.combo_returns)
    return (
        reference_candles,
        reference_target,
        rule_data.successful_param_grid,
        evaluator,
        config,
        None,
        None,
    )


def _two_unit_train_windows(
    fold_rows: list[dict],
    last_ts: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not fold_rows:
        return []
    first = fold_rows[0]
    train_start = pd.Timestamp(first["train_start"])
    train_end = pd.Timestamp(first["train_end"])
    test_start = pd.Timestamp(first["test_start"])
    end_exclusive = last_ts + pd.Timedelta(days=1)
    return [(train_start, train_end), (test_start, end_exclusive)]


def _prepare_candles_for_shuffler(candles_df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in candles_df.columns and hasattr(candles_df.index, "dtype"):
        if str(candles_df.index.dtype).startswith("datetime"):
            df = candles_df.copy()
            df["datetime"] = candles_df.index
            return df
    return candles_df.copy()


def _one_candle_shuffle_rep(
    seed: int,
    config: object,
    candles_prepared: pd.DataFrame,
    train_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    fold_rows: list[dict],
    expanded: list,
    runtime_config: WalkforwardResearchConfig,
    module_name: str,
    feature_type: FeatureType,
    objective_metric_name: str,
) -> float:
    from utils.evaluation.permutation_test.candle_shuffle import permute_walk_forward

    metric_fn = resolve_objective_metric(objective_metric_name)
    shuffled_candles = permute_walk_forward(
        candles_prepared,
        train_windows=train_windows,
        random_seed=seed,
    )
    if feature_type == FeatureType.CONTINUOUS:
        continuous_data = load_continuous_research_data(
            config,
            expanded,
            candles_override=shuffled_candles,
        )
        if not continuous_data.successful_param_grid or continuous_data.reference_index is None:
            return 0.0
        successful_param_grid = expand_params_with_selected_bin(
            continuous_data.successful_param_grid,
            bin_index_min=config.binning_params.bin_index_min,
            bin_index_max=config.binning_params.bin_index_max,
        )
        ref_target = build_reference_target(
            continuous_data.reference_index,
            continuous_data.reference_target_series,
        )
        ref_candles = pd.DataFrame({"close": ref_target}, index=continuous_data.reference_index)
        evaluator = build_continuous_walkforward_evaluator(
            continuous_data.combo_feature_target, config
        )
        portfolio_candles_df = load_portfolio_candles(config)
        report = run_walkforward_research(
            candles_df=ref_candles,
            target=ref_target,
            feature_type="continuous",
            module_name=module_name,
            config=runtime_config,
            param_grid=successful_param_grid,
            evaluate_param_combo=evaluator,
            research_config=config,
            portfolio_candles_df=portfolio_candles_df,
            feature_data_by_combo=continuous_data.combo_feature_target,
            output_dir=None,
            fold_rows_override=fold_rows,
        )
    else:
        rule_data = load_rule_based_research_data(
            config,
            expanded,
            candles_override=shuffled_candles,
            dedupe_before_multiply=False,
            capture_target_as_reference=False,
        )
        if not rule_data.successful_param_grid or rule_data.reference_index is None:
            return 0.0
        ref_target = build_reference_target(rule_data.reference_index, None)
        ref_candles = pd.DataFrame({"close": ref_target}, index=rule_data.reference_index)
        evaluator = build_rule_based_walkforward_evaluator(rule_data.combo_returns)
        report = run_walkforward_research(
            candles_df=ref_candles,
            target=ref_target,
            feature_type="rule_based",
            module_name=module_name,
            config=runtime_config,
            param_grid=rule_data.successful_param_grid,
            evaluate_param_combo=evaluator,
            research_config=config,
            output_dir=None,
            fold_rows_override=fold_rows,
        )
    return float(
        aggregate_oos_metric_from_report(
            report,
            treat_no_selection_as_zero=True,
            metric_fn=metric_fn,
        )
    )


def run_candle_shuffle_null(
    config: object,
    runtime_config: WalkforwardResearchConfig,
    reference_target: pd.Series,
    fold_rows: list[dict],
    nreps: int,
    random_seed: int | None,
    objective_metric_name: str = "sharpe",
    n_jobs: int = 1,
) -> np.ndarray:
    portfolio_candles = load_portfolio_candles(config)
    candles_prepared = _prepare_candles_for_shuffler(portfolio_candles)
    if "datetime" not in candles_prepared.columns:
        candles_prepared["datetime"] = candles_prepared.index
    last_ts = pd.Timestamp(reference_target.index.max())
    train_windows = _two_unit_train_windows(fold_rows, last_ts)
    if not train_windows:
        return np.array([])

    expanded = expand_bias_specs(config.bias_spec)
    module_name = str(getattr(config, "bias_spec", {}).get("module_name", "rsi"))
    feature_type = getattr(config, "feature_type", FeatureType.CONTINUOUS)
    rng = np.random.default_rng(random_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(nreps)]

    if n_jobs == 1:
        null_metrics = np.empty(nreps, dtype=float)
        for i in tqdm(range(nreps), desc="Candle shuffle", unit="rep"):
            null_metrics[i] = _one_candle_shuffle_rep(
                seeds[i],
                config,
                candles_prepared,
                train_windows,
                fold_rows,
                expanded,
                runtime_config,
                module_name,
                feature_type,
                objective_metric_name,
            )
        return null_metrics

    from multiprocessing import cpu_count

    from joblib import Parallel, delayed

    n_jobs_actual = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
    with _joblib_tqdm(nreps, desc="Candle shuffle", unit="rep"):
        results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_one_candle_shuffle_rep)(
                seed,
                config,
                candles_prepared,
                train_windows,
                fold_rows,
                expanded,
                runtime_config,
                module_name,
                feature_type,
                objective_metric_name,
            )
            for seed in seeds
        )
    return np.array(results, dtype=float)
