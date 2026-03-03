from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from feature_research.config import FeatureType
from feature_research.core_helpers import expand_params_with_selected_bin
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    get_tickers_with_coverage_for_config,
    populate_cache_if_needed,
)
from feature_research.walkforward.evaluators import (
    build_continuous_walkforward_evaluator,
    build_rule_based_walkforward_evaluator,
)
from feature_research.walkforward.io import resolve_walkforward_output_dir, write_walkforward_artifacts
from feature_research.walkforward.research_data import (
    build_reference_target,
    load_continuous_research_data,
    load_portfolio_candles,
    load_rule_based_research_data,
)
from feature_research.walkforward.runner import run_walkforward_research
from feature_research.walkforward.visualization import plot_fold_timeline, plot_selection_stability

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig
    from feature_research.walkforward.runner import WalkforwardRunReport


def run_walkforward_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> "WalkforwardRunReport":
    output_dir.mkdir(parents=True, exist_ok=True)

    covered_tickers = get_tickers_with_coverage_for_config(config)
    if len(covered_tickers) < len(config.tickers):
        dropped = set(config.tickers) - set(covered_tickers)
        print(
            f"[walkforward] Tickers without full date coverage for "
            f"{config.start.date()}–{config.end.date()} dropped: {[t.name for t in dropped]}"
        )
    config = replace(config, tickers=covered_tickers)
    if not config.tickers:
        raise ValueError(
            "No tickers have OHLC data covering the config date range. "
            "Check data/ohlc_data or narrow config.start/end."
        )

    populate_cache_if_needed(config)
    expanded = expand_bias_specs(config.bias_spec)
    feature_type_label = config.feature_type.value.upper()

    print(f"\n{'='*64}")
    print(f"Walkforward Pipeline: {config.bias_spec['module_name'].upper()} ({feature_type_label})")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Combos  : {len(expanded)}")
    print(f"Selection method: {config.walkforward._effective_selection_method()}")
    print(f"{'='*64}\n")

    output_dir = resolve_walkforward_output_dir(
        feature_type=config.feature_type.value,
        module_name=str(config.bias_spec["module_name"]),
        root_dir=config.walkforward.output_root,
    )

    successful_param_grid: list[dict[str, object]]
    reference_index: pd.DatetimeIndex

    if config.feature_type == FeatureType.CONTINUOUS:
        assert config.binning_params.strategy == "long", (
            "Continuous walkforward expects strategy='long'; got %r" % config.binning_params.strategy
        )
        data = load_continuous_research_data(config, expanded, print_loaded=True)
        if not data.successful_param_grid or data.reference_index is None:
            raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

        reference_index = data.reference_index
        data_end = pd.Timestamp(reference_index.max()).normalize()
        config_end = pd.Timestamp(config.end).normalize()
        if data_end < config_end:
            print(
                f"\n  [WARNING] Loaded data ends {data_end.date()}; config.end is {config_end.date()}. "
                "Last fold will not include 2025. To extend: ensure raw OHLC in data/ohlc_data has "
                "dates through config.end and run with populate_cache=True once to refresh the cache.\n"
            )

        reference_target = build_reference_target(reference_index, data.reference_target_series)
        reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
        portfolio_candles = load_portfolio_candles(config)

        successful_param_grid = expand_params_with_selected_bin(
            data.successful_param_grid,
            bin_index_min=config.binning_params.bin_index_min,
            bin_index_max=config.binning_params.bin_index_max,
        )
        print(
            f"  Param grid (3D): {len(successful_param_grid)} combos (lookback × bin_count × selected_bin).\n",
            flush=True,
        )
        walkforward_report = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="continuous",
            module_name=str(config.bias_spec["module_name"]),
            config=config.walkforward,
            param_grid=successful_param_grid,
            evaluate_param_combo=build_continuous_walkforward_evaluator(data.combo_feature_target, config),
            research_config=config,
            portfolio_candles_df=portfolio_candles,
            feature_data_by_combo=data.combo_feature_target,
            output_dir=output_dir,
        )
    elif config.feature_type == FeatureType.RULE_BASED:
        data = load_rule_based_research_data(
            config,
            expanded,
            dedupe_before_multiply=True,
            capture_target_as_reference=True,
            print_loaded=True,
        )
        if not data.successful_param_grid or data.reference_index is None:
            raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

        reference_index = data.reference_index
        data_end = pd.Timestamp(reference_index.max()).normalize()
        config_end = pd.Timestamp(config.end).normalize()
        if data_end < config_end:
            print(
                f"\n  [WARNING] Loaded data ends {data_end.date()}; config.end is {config_end.date()}. "
                "Last fold will not include 2025. To extend: ensure raw OHLC in data/ohlc_data has "
                "dates through config.end and run with populate_cache=True once to refresh the cache.\n"
            )

        reference_target = build_reference_target(reference_index, data.reference_target_series)
        reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
        portfolio_candles_df = load_portfolio_candles(config)

        successful_param_grid = data.successful_param_grid
        walkforward_report = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="rule_based",
            module_name=str(config.bias_spec["module_name"]),
            config=config.walkforward,
            param_grid=successful_param_grid,
            evaluate_param_combo=build_rule_based_walkforward_evaluator(data.combo_returns),
            research_config=config,
            portfolio_candles_df=portfolio_candles_df,
            feature_data_by_combo=data.combo_feature_target,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unknown feature_type: {config.feature_type}")

    if walkforward_report.folds_df.empty:
        first_ts = pd.Timestamp(reference_index.min())
        last_ts = pd.Timestamp(reference_index.max())
        raise ValueError(
            "No walkforward folds were generated. "
            f"Data window={first_ts.date()}..{last_ts.date()}, "
            f"walkforward train_start={config.walkforward.train_start.date()}, "
            f"train_end={config.walkforward.train_end.date()}, "
            f"test_step={config.walkforward.test_step}, num_steps={config.walkforward.num_steps}."
        )

    stability_figure, _ = plot_selection_stability(
        selection_summary_df=walkforward_report.selection_summary_df,
        top_k=config.walkforward.top_k,
    )
    timeline_figure, _ = plot_fold_timeline(folds_df=walkforward_report.folds_df)
    write_walkforward_artifacts(
        report=walkforward_report,
        walkforward_stability_figure=stability_figure,
        fold_timeline_figure=timeline_figure,
        feature_type=config.feature_type.value,
        module_name=str(config.bias_spec["module_name"]),
        root_dir=config.walkforward.output_root,
        research_context={
            "tickers": [ticker.name for ticker in config.tickers],
            "period_start": str(config.start.date()),
            "period_end": str(config.end.date()),
            "target_col": config.target_col,
            "strategy": config.strategy,
            "walkforward_test_step": config.walkforward.test_step,
            "walkforward_num_steps": config.walkforward.num_steps,
            "walkforward_top_k": config.walkforward.top_k,
            "walkforward_selection_method": config.walkforward._effective_selection_method(),
        },
    )
    plt.close(stability_figure)
    plt.close(timeline_figure)

    print(
        f"\nDone. {len(successful_param_grid)}/{len(expanded)} combos "
        f"-> walkforward artifacts written to {config.walkforward.output_root}\n"
    )
    return walkforward_report


def run_continuous_walkforward_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> "WalkforwardRunReport":
    return run_walkforward_pipeline(config, output_dir)


def run_rule_based_walkforward_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> "WalkforwardRunReport":
    return run_walkforward_pipeline(config, output_dir)
