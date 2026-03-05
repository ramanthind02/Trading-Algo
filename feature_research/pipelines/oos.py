from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from feature_research.config import OOSWindowConfig
from feature_research.core_helpers import expand_params_with_selected_bin
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    get_tickers_with_coverage_for_config,
    populate_cache_if_needed,
)
from utils.evaluation.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
)
from utils.evaluation.walkforward.evaluators import (
    build_continuous_walkforward_evaluator,
    build_rule_based_walkforward_evaluator,
)
from utils.evaluation.walkforward.io import (
    resolve_walkforward_output_dir,
    write_walkforward_artifacts,
)
from utils.evaluation.walkforward.research_data import (
    build_reference_target,
    load_continuous_research_data,
    load_portfolio_candles,
    load_rule_based_research_data,
)
from utils.evaluation.walkforward.runner import (
    build_fold_rows_from_explicit_specs,
    run_walkforward_research,
)
from utils.evaluation.walkforward.visualization import (
    plot_fold_timeline,
    plot_selection_stability,
)

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig
    from utils.evaluation.walkforward.runner import WalkforwardRunReport


def _require_oos_window(config: "ResearchConfig") -> OOSWindowConfig:
    if config.oos_window is None:
        raise ValueError(
            "OOS window is not set (config.oos_window is None). "
            "Set it in feature_research.config.load_config()."
        )
    return config.oos_window


def _effective_oos_window(config: "ResearchConfig") -> OOSWindowConfig:
    oos = _require_oos_window(config)
    if config.validation_window is None:
        return oos
    # OOS must refit on train+validation and test on oos_window.test.
    return OOSWindowConfig(
        train_start=config.validation_window.train_start,
        train_end=config.validation_window.test_end,
        test_start=oos.test_start,
        test_end=oos.test_end,
    )


def _build_runtime_walkforward_config(
    config: "ResearchConfig",
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> WalkforwardResearchConfig:
    test_step = max(1, int((test_end - test_start).days))
    return WalkforwardResearchConfig(
        train_start=train_start.to_pydatetime(),
        train_end=train_end.to_pydatetime(),
        enabled=True,
        test_step=test_step,
        num_steps=1,
        top_k=config.top_k,
        objective_metric_name=config.objective_metric_name,
        min_fold_samples=10,
        output_root=config.output_root,
        selection_method=WalkforwardSelectionMethod.TOP_K,
        n_jobs=config.n_jobs,
        smoothing_self_weight=config.smoothing_self_weight,
    )


def run_oos_pipeline(config: "ResearchConfig") -> "WalkforwardRunReport":
    oos = _effective_oos_window(config)
    data_start = min(config.start, oos.train_start)
    data_end = max(config.end, oos.test_end)
    config_oos = replace(config, start=data_start, end=data_end)

    covered_tickers = get_tickers_with_coverage_for_config(config_oos)
    if not covered_tickers:
        raise ValueError(
            "No tickers have OHLC data covering the OOS date range. "
            "Check data/ohlc_data or narrow dates."
        )

    output_dir = resolve_walkforward_output_dir(
        feature_type=config.feature_type.value,
        module_name=str(config.bias_spec["module_name"]),
        root_dir=config.output_root,
        output_subdir="oos",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    populate_cache_if_needed(config_oos)
    expanded = expand_bias_specs(config.bias_spec)
    feature_type_label = config.feature_type.value.upper()
    print(f"\n{'='*64}")
    print(f"OOS Pipeline: {config.bias_spec['module_name'].upper()} ({feature_type_label})")
    print(f"Tickers : {[t.name for t in config_oos.tickers]}")
    print(f"Train   : {oos.train_start.date()} -> {oos.train_end.date()}")
    print(f"Test    : {oos.test_start.date()} -> {oos.test_end.date()}")
    print(f"Combos  : {len(expanded)}")
    print(f"{'='*64}\n")

    train_start = pd.Timestamp(oos.train_start)
    train_end = pd.Timestamp(oos.train_end)
    test_start = pd.Timestamp(oos.test_start)
    test_end = pd.Timestamp(oos.test_end)
    runtime_config = _build_runtime_walkforward_config(
        config,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    if config.feature_type.value == "continuous":
        data = load_continuous_research_data(config_oos, expanded, print_loaded=True)
        if not data.successful_param_grid or data.reference_index is None:
            raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

        reference_index = data.reference_index
        reference_target = build_reference_target(reference_index, data.reference_target_series)
        reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
        portfolio_candles = load_portfolio_candles(config_oos)
        fold_rows = build_fold_rows_from_explicit_specs(
            reference_index,
            [(oos.train_start, oos.train_end, oos.test_start, oos.test_end)],
            min_fold_samples=runtime_config.min_fold_samples,
        )
        if not fold_rows:
            raise ValueError("OOS fold has insufficient samples. Check oos_window dates and data range.")

        successful_param_grid = expand_params_with_selected_bin(
            data.successful_param_grid,
            bin_index_min=config.binning_params.bin_index_min,
            bin_index_max=config.binning_params.bin_index_max,
        )
        report = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="continuous",
            module_name=str(config.bias_spec["module_name"]),
            config=runtime_config,
            param_grid=successful_param_grid,
            evaluate_param_combo=build_continuous_walkforward_evaluator(data.combo_feature_target, config),
            research_config=config,
            portfolio_candles_df=portfolio_candles,
            feature_data_by_combo=data.combo_feature_target,
            output_dir=output_dir,
            fold_rows_override=fold_rows,
        )
    elif config.feature_type.value == "rule_based":
        data = load_rule_based_research_data(
            config_oos,
            expanded,
            dedupe_before_multiply=False,
            capture_target_as_reference=False,
            print_loaded=True,
        )
        if not data.successful_param_grid or data.reference_index is None:
            raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

        reference_index = data.reference_index
        reference_target = build_reference_target(reference_index, None)
        reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
        portfolio_candles = load_portfolio_candles(config_oos)
        fold_rows = build_fold_rows_from_explicit_specs(
            reference_index,
            [(oos.train_start, oos.train_end, oos.test_start, oos.test_end)],
            min_fold_samples=runtime_config.min_fold_samples,
        )
        if not fold_rows:
            raise ValueError("OOS fold has insufficient samples. Check oos_window dates and data range.")

        report = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="rule_based",
            module_name=str(config.bias_spec["module_name"]),
            config=runtime_config,
            param_grid=data.successful_param_grid,
            evaluate_param_combo=build_rule_based_walkforward_evaluator(data.combo_returns),
            research_config=config,
            portfolio_candles_df=portfolio_candles,
            feature_data_by_combo=data.combo_feature_target,
            output_dir=output_dir,
            fold_rows_override=fold_rows,
        )
    else:
        raise ValueError(f"Unknown feature_type: {config.feature_type}")

    stability_figure, _ = plot_selection_stability(
        selection_summary_df=report.selection_summary_df,
        top_k=config.top_k,
    )
    timeline_figure, _ = plot_fold_timeline(folds_df=report.folds_df)
    write_walkforward_artifacts(
        report=report,
        walkforward_stability_figure=stability_figure,
        fold_timeline_figure=timeline_figure,
        feature_type=config.feature_type.value,
        module_name=str(config.bias_spec["module_name"]),
        root_dir=config.output_root,
        research_context={
            "tickers": [ticker.name for ticker in config.tickers],
            "period_start": str(oos.train_start.date()),
            "period_end": str(oos.test_end.date()),
            "target_col": config.target_col,
            "strategy": config.strategy,
            "oos_train_start": str(oos.train_start.date()),
            "oos_train_end": str(oos.train_end.date()),
            "oos_test_start": str(oos.test_start.date()),
            "oos_test_end": str(oos.test_end.date()),
            "oos_top_k": config.top_k,
            "oos_selection_method": "top_k",
        },
        output_subdir="oos",
    )
    plt.close(stability_figure)
    plt.close(timeline_figure)
    print(f"\nDone. OOS artifacts written to {output_dir}\n")
    return report
