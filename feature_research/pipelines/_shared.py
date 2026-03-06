"""Shared pipeline utilities for OOS and validation evaluation phases."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import pandas as pd

from feature_research.config import FeatureType, OOSWindowConfig
from feature_research.core_helpers import (
    build_runtime_walkforward_config,
    expand_params_with_selected_bin,
)
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    get_tickers_with_coverage_for_config,
    populate_cache_if_needed,
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


def _effective_oos_window(config: "ResearchConfig") -> OOSWindowConfig:
    """Compute effective OOS window combining validation and OOS windows if both exist.

    If validation_window is set, OOS must refit on train+validation and test on oos_window.test.
    This ensures consistency with the validation phase.

    Parameters
    ----------
    config : ResearchConfig
        Configuration with oos_window and optional validation_window.

    Returns
    -------
    OOSWindowConfig
        Either the oos_window directly, or combined window using validation train
        and oos test boundaries.
    """
    oos = config.oos_window
    if oos is None:
        raise ValueError(
            "OOS window is not set (config.oos_window is None). "
            "Set it in feature_research.config.load_config()."
        )
    if config.validation_window is None:
        return oos
    # OOS must refit on train+validation and test on oos_window.test.
    return OOSWindowConfig(
        train_start=config.validation_window.train_start,
        train_end=config.validation_window.test_end,
        test_start=oos.test_start,
        test_end=oos.test_end,
    )


def _run_evaluation_pipeline(
    phase: Literal["oos", "validation"],
    config: "ResearchConfig",
    output_dir: str | None = None,
) -> "WalkforwardRunReport":
    """Run OOS or validation evaluation pipeline (shared implementation).

    Parameters
    ----------
    phase : {"oos", "validation"}
        Which phase to run. Controls window source, output subdir, and context keys.
    config : ResearchConfig
        Research configuration.
    output_dir : str, optional
        Override output directory (only used for validation). If None, computed
        from config.output_root.

    Returns
    -------
    WalkforwardRunReport
        Walkforward research report with folds and selection summary.
    """
    if phase == "oos":
        window = _effective_oos_window(config)
        phase_label = "OOS"
        phase_subdir = "oos"
        context_prefix = "oos"
        error_msg_window = "oos_window"
        error_msg_range = "oos_window dates"
    elif phase == "validation":
        if config.validation_window is None:
            raise ValueError(
                "Validation window is not set (config.validation_window is None). "
                "Set it in feature_research.config.load_config()."
            )
        window = config.validation_window
        phase_label = "Validation"
        phase_subdir = "validation"
        context_prefix = "validation"
        error_msg_window = "validation_window"
        error_msg_range = "validation_window dates"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    data_start = min(config.start, window.train_start)
    data_end = max(config.end, window.test_end)
    config_phase = replace(config, start=data_start, end=data_end)

    covered_tickers = get_tickers_with_coverage_for_config(config_phase)
    if not covered_tickers:
        raise ValueError(
            f"No tickers have OHLC data covering the {phase} date range. "
            "Check data/ohlc_data or narrow dates."
        )

    if output_dir is None:
        output_dir_path = resolve_walkforward_output_dir(
            feature_type=config.feature_type.value,
            module_name=str(config.eval_bias_spec["module_name"]),
            root_dir=config.output_root,
            output_subdir=phase_subdir,
        )
    else:
        from pathlib import Path
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    populate_cache_if_needed(config_phase)
    expanded = expand_bias_specs(config.eval_bias_spec)
    feature_type_label = config.feature_type.value.upper()
    window_label = "Val" if phase == "validation" else "Test"
    print(f"\n{'='*64}")
    print(f"{phase_label} Pipeline: {config.eval_bias_spec['module_name'].upper()} ({feature_type_label})")
    print(f"Tickers : {[t.name for t in config_phase.tickers]}")
    print(f"Train   : {window.train_start.date()} -> {window.train_end.date()}")
    print(f"{window_label}     : {window.test_start.date()} -> {window.test_end.date()}")
    print(f"Combos  : {len(expanded)}")
    print(f"{'='*64}\n")

    train_start = pd.Timestamp(window.train_start)
    train_end = pd.Timestamp(window.train_end)
    test_start = pd.Timestamp(window.test_start)
    test_end = pd.Timestamp(window.test_end)
    runtime_config = build_runtime_walkforward_config(
        config,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    if config.feature_type == FeatureType.CONTINUOUS or config.feature_type.value == "continuous":
        data = load_continuous_research_data(config_phase, expanded, print_loaded=True)
        if not data.successful_param_grid or data.reference_index is None:
            raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

        reference_index = data.reference_index
        reference_target = build_reference_target(reference_index, data.reference_target_series)
        reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
        portfolio_candles = load_portfolio_candles(config_phase)
        fold_rows = build_fold_rows_from_explicit_specs(
            reference_index,
            [(window.train_start, window.train_end, window.test_start, window.test_end)],
            min_fold_samples=runtime_config.min_fold_samples,
        )
        if not fold_rows:
            raise ValueError(f"{phase_label} fold has insufficient samples. Check {error_msg_range} and data range.")

        successful_param_grid = expand_params_with_selected_bin(
            data.successful_param_grid,
            bin_index_min=config.binning_params.bin_index_min,
            bin_index_max=config.binning_params.bin_index_max,
        )
        report = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="continuous",
            module_name=str(config.eval_bias_spec["module_name"]),
            config=runtime_config,
            param_grid=successful_param_grid,
            evaluate_param_combo=build_continuous_walkforward_evaluator(data.combo_feature_target, config),
            research_config=config,
            portfolio_candles_df=portfolio_candles,
            feature_data_by_combo=data.combo_feature_target,
            output_dir=output_dir_path,
            fold_rows_override=fold_rows,
        )
    elif config.feature_type == FeatureType.RULE_BASED or config.feature_type.value == "rule_based":
        data = load_rule_based_research_data(
            config_phase,
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
        portfolio_candles = load_portfolio_candles(config_phase)
        fold_rows = build_fold_rows_from_explicit_specs(
            reference_index,
            [(window.train_start, window.train_end, window.test_start, window.test_end)],
            min_fold_samples=runtime_config.min_fold_samples,
        )
        if not fold_rows:
            raise ValueError(f"{phase_label} fold has insufficient samples. Check {error_msg_range} and data range.")

        report = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="rule_based",
            module_name=str(config.eval_bias_spec["module_name"]),
            config=runtime_config,
            param_grid=data.successful_param_grid,
            evaluate_param_combo=build_rule_based_walkforward_evaluator(data.combo_returns),
            research_config=config,
            portfolio_candles_df=portfolio_candles,
            feature_data_by_combo=data.combo_feature_target,
            output_dir=output_dir_path,
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
        module_name=str(config.eval_bias_spec["module_name"]),
        root_dir=config.output_root,
        research_context={
            "tickers": [ticker.name for ticker in config.tickers],
            "period_start": str(window.train_start.date()),
            "period_end": str(window.test_end.date()),
            "target_col": config.target_col,
            "strategy": config.strategy,
            f"{context_prefix}_train_start": str(window.train_start.date()),
            f"{context_prefix}_train_end": str(window.train_end.date()),
            f"{context_prefix}_test_start": str(window.test_start.date()),
            f"{context_prefix}_test_end": str(window.test_end.date()),
            f"{context_prefix}_top_k": config.top_k,
            f"{context_prefix}_selection_method": "top_k",
        },
        output_subdir=phase_subdir,
    )
    plt.close(stability_figure)
    plt.close(timeline_figure)
    print(f"\nDone. {phase_label} artifacts written to {output_dir_path}\n")
    return report
