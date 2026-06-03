"""Shared pipeline utilities for validation and portfolio-addition evaluation phases."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import pandas as pd

from feature_research.config import ResearchWindowConfig
from feature_research._internal.core_helpers import (
    build_runtime_walkforward_config,
    normalize_timeframe_from_bias_spec,
)
from feature_research.shared.contracts import OosCorrelationBundle
from feature_research.research_table_exports import (
    walkforward_visualization_csv_dir,
    write_walkforward_equity_csvs,
)
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    get_effective_range_and_tickers,
    get_tickers_with_coverage_for_config,
    populate_cache_if_needed,
)
from utils.evaluation.walkforward.evaluators import build_signed_signal_walkforward_evaluator
from utils.evaluation.walkforward.io import resolve_walkforward_output_dir
from utils.evaluation.walkforward.io import write_walkforward_artifacts
from utils.evaluation.walkforward.research_data import (
    build_reference_target,
    load_portfolio_candles,
    load_signed_signal_research_data,
)
from utils.evaluation.walkforward.runner import (
    build_fold_rows_from_explicit_specs,
    run_walkforward_research,
)

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig
    from utils.evaluation.walkforward.runner import WalkforwardRunReport

EvaluationPipelineResult = tuple["WalkforwardRunReport", OosCorrelationBundle | None]


SIGNED_SIGNAL_FEATURE_TYPE = "signed_signal"


def _require_research_window(config: "ResearchConfig") -> ResearchWindowConfig:
    window = config.research_window
    if window is None:
        raise ValueError(
            "Research window is not set (config.research_window is None). "
            "Set it in feature_research.config.load_config()."
        )
    return window


def _run_evaluation_pipeline(
    phase: Literal["oos", "validation"],
    config: "ResearchConfig",
    output_dir: str | None = None,
) -> EvaluationPipelineResult:
    window = _require_research_window(config)
    if phase == "oos":
        phase_label = "Portfolio Addition"
        phase_subdir = "oos"
    elif phase == "validation":
        phase_label = "Validation"
        phase_subdir = "validation"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    data_start = min(config.start, window.train_start)
    data_end = max(config.end, window.val_end)
    config_phase = replace(config, start=data_start, end=data_end)

    covered_tickers = get_tickers_with_coverage_for_config(
        config_phase,
        bias_spec=config.eval_bias_spec,
    )
    if not covered_tickers:
        effective = get_effective_range_and_tickers(
            config_phase,
            bias_spec=config.eval_bias_spec,
        )
        if effective is None:
            raise ValueError(
                "No OHLC data found for the requested range. "
                "Check data/ohlc_data or narrow dates."
            )
        effective_start, effective_end, effective_tickers = effective
        config_phase = replace(
            config_phase,
            start=effective_start.to_pydatetime(),
            end=effective_end.to_pydatetime(),
            tickers=effective_tickers,
        )
        train_start_c = max(window.train_start, effective_start.to_pydatetime())
        train_end_c = min(window.train_end, effective_end.to_pydatetime())
        val_start_c = max(window.val_start, effective_start.to_pydatetime())
        val_end_c = min(window.val_end, effective_end.to_pydatetime())
        if train_start_c >= train_end_c or val_start_c >= val_end_c or train_end_c > val_start_c:
            raise ValueError(
                "Insufficient data after narrowing to available range "
                "(train/validation window would be empty or invalid)."
            )
        window = ResearchWindowConfig(
            train_start=train_start_c,
            train_end=train_end_c,
            val_start=val_start_c,
            val_end=val_end_c,
        )
        print(
            f"[{phase_label}] No full coverage; using narrowed range "
            f"{effective_start.date()}–{effective_end.date()} and tickers {[t.name for t in effective_tickers]}."
        )

    if output_dir is None:
        output_dir_path = resolve_walkforward_output_dir(
            feature_type=SIGNED_SIGNAL_FEATURE_TYPE,
            module_name=str(config.eval_bias_spec["module_name"]),
            root_dir=config.output_root,
            output_subdir=phase_subdir,
        )
    else:
        from pathlib import Path

        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    _stage_start = time.perf_counter()

    populate_cache_if_needed(config_phase, bias_spec=config.eval_bias_spec)
    expanded = expand_bias_specs(config.eval_bias_spec)
    feature_type_label = SIGNED_SIGNAL_FEATURE_TYPE.upper()
    window_label = "Val"
    print(f"\n{'='*64}")
    print(f"{phase_label} Pipeline: {config.eval_bias_spec['module_name'].upper()} ({feature_type_label})")
    print(f"Tickers : {[t.name for t in config_phase.tickers]}")
    print(f"Train   : {window.train_start.date()} -> {window.train_end.date()}")
    print(f"{window_label}     : {window.val_start.date()} -> {window.val_end.date()}")
    print(f"Combos  : {len(expanded)}")
    print(f"{'='*64}\n")

    train_start = pd.Timestamp(window.train_start)
    train_end = pd.Timestamp(window.train_end)
    val_start = pd.Timestamp(window.val_start)
    val_end = pd.Timestamp(window.val_end)
    runtime_config = build_runtime_walkforward_config(
        config,
        train_start=train_start,
        train_end=train_end,
        test_start=val_start,
        test_end=val_end,
    )

    data = load_signed_signal_research_data(config_phase, expanded, print_loaded=True)
    if not data.successful_param_grid or data.reference_index is None:
        raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

    reference_index = data.reference_index
    reference_target = build_reference_target(reference_index, data.reference_target_series)
    reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
    portfolio_candles = load_portfolio_candles(config_phase)
    fold_rows = build_fold_rows_from_explicit_specs(
        reference_index,
        [(window.train_start, window.train_end, window.val_start, window.val_end)],
        min_fold_samples=runtime_config.min_fold_samples,
    )
    if not fold_rows:
        raise ValueError(f"{phase_label} fold has insufficient samples. Check date range and data.")

    evaluator = build_signed_signal_walkforward_evaluator(data.combo_signal_target)
    report = run_walkforward_research(
        candles_df=reference_candles,
        target=reference_target,
        feature_type=SIGNED_SIGNAL_FEATURE_TYPE,
        module_name=str(config.eval_bias_spec["module_name"]),
        config=runtime_config,
        param_grid=data.successful_param_grid,
        evaluate_param_combo=evaluator,
        research_config=config,
        portfolio_candles_df=portfolio_candles,
        feature_data_by_combo=data.combo_signal_target,
        output_dir=output_dir_path,
        fold_rows_override=fold_rows,
        phase_label=phase_label,
    )
    eval_tf = normalize_timeframe_from_bias_spec(
        config.eval_bias_spec,
        fallback=config.timeframe,
    )
    visualization_dir = walkforward_visualization_csv_dir(config.output_root, phase_subdir)
    oos_correlation_bundle: OosCorrelationBundle | None = None
    if phase == "oos":
        oos_correlation_bundle = OosCorrelationBundle(
            combo_signal_target=data.combo_signal_target,
            selection_summary_df=report.selection_summary_df,
            eval_tf=eval_tf,
            research_eval_bias_spec=dict(config.eval_bias_spec),
            target_col=config.target_col,
            extended_start=window.train_start,
            extended_end=window.val_end,
            module_name=str(config.eval_bias_spec["module_name"]),
        )
    if not report.selection_summary_df.empty:
        holdout_csv_stem = (
            "equity_curve_validation_only"
            if phase == "validation"
            else "equity_curve_portfolio_addition_val_only"
        )
        extended_csv_stem = (
            "equity_curve_train_and_validation"
            if phase == "validation"
            else "equity_curve_train_and_val_for_portfolio_addition"
        )
        visualization_paths = write_walkforward_equity_csvs(
            combo_signal_target=data.combo_signal_target,
            selection_summary_df=report.selection_summary_df,
            module_name=str(config.eval_bias_spec["module_name"]),
            timeframe=eval_tf,
            holdout_start=window.val_start,
            holdout_end=window.val_end,
            extended_start=window.train_start,
            extended_end=window.val_end,
            output_visualization_dir=visualization_dir,
            holdout_csv_stem=holdout_csv_stem,
            extended_csv_stem=extended_csv_stem,
            portfolio_candles=portfolio_candles,
            target_volatility=config.tearsheet_target_annual_volatility,
        )
        print(
            f"[{phase_label}] Visualization equity CSVs: "
            f"{visualization_paths['holdout'].name}, "
            f"{visualization_paths['extended'].name} -> {visualization_dir}"
        )
    write_walkforward_artifacts(
        report=report,
        feature_type=SIGNED_SIGNAL_FEATURE_TYPE,
        module_name=str(config.eval_bias_spec["module_name"]),
        root_dir=config.output_root,
        output_subdir=phase_subdir,
    )
    print(
        f"[{phase_label}] Walkforward stage completed in "
        f"{time.perf_counter() - _stage_start:.1f}s"
    )
    if phase == "validation" and config.validation_robustness.enabled:
        from feature_research.validation.robustness_runner import run_and_write_validation_robustness

        _robustness_start = time.perf_counter()
        visualization_dir.mkdir(parents=True, exist_ok=True)
        robustness_artifacts = run_and_write_validation_robustness(
            config,
            output_dir=visualization_dir,
            selection_summary_df=report.selection_summary_df,
            eval_research_data=data,
        )
        if robustness_artifacts:
            print(
                f"[{phase_label}] Validation robustness report: "
                f"{robustness_artifacts['json'].name} -> {visualization_dir} "
                f"({time.perf_counter() - _robustness_start:.1f}s)"
            )
    if phase == "validation" and config.portfolio_addition_gate.enabled:
        from feature_research.portfolio_addition.gate_runner import (
            run_and_write_portfolio_addition_gate,
        )

        _gate_start = time.perf_counter()
        gate_artifacts = run_and_write_portfolio_addition_gate(
            config,
            output_dir=visualization_dir,
        )
        if gate_artifacts:
            print(
                f"[{phase_label}] Portfolio addition gate report: "
                f"{gate_artifacts['json'].name} -> {visualization_dir} "
                f"({time.perf_counter() - _gate_start:.1f}s)"
            )
    print(
        f"[{phase_label}] Total pipeline time: "
        f"{time.perf_counter() - _stage_start:.1f}s"
    )
    return report, oos_correlation_bundle
