"""Shared pipeline utilities for OOS and validation evaluation phases."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import pandas as pd

from feature_research.config import OOSWindowConfig
from feature_research.core_helpers import build_runtime_walkforward_config
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
    from feature_research.in_sample.config import ResearchConfig
    from utils.evaluation.walkforward.runner import WalkforwardRunReport


SIGNED_SIGNAL_FEATURE_TYPE = "signed_signal"


def _effective_oos_window(config: "ResearchConfig") -> OOSWindowConfig:
    oos = config.oos_window
    if oos is None:
        raise ValueError(
            "OOS window is not set (config.oos_window is None). "
            "Set it in feature_research.config.load_config()."
        )
    if config.validation_window is None:
        return oos
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
    if phase == "oos":
        window = _effective_oos_window(config)
        phase_label = "OOS"
        phase_subdir = "oos"
    elif phase == "validation":
        if config.validation_window is None:
            raise ValueError(
                "Validation window is not set (config.validation_window is None). "
                "Set it in feature_research.config.load_config()."
            )
        window = config.validation_window
        phase_label = "Validation"
        phase_subdir = "validation"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    data_start = min(config.start, window.train_start)
    data_end = max(config.end, window.test_end)
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
        test_start_c = max(window.test_start, effective_start.to_pydatetime())
        test_end_c = min(window.test_end, effective_end.to_pydatetime())
        if train_start_c >= train_end_c or test_start_c >= test_end_c or train_end_c > test_start_c:
            raise ValueError(
                "Insufficient data after narrowing to available range "
                "(train/test window would be empty or invalid)."
            )
        window = OOSWindowConfig(
            train_start=train_start_c,
            train_end=train_end_c,
            test_start=test_start_c,
            test_end=test_end_c,
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

    populate_cache_if_needed(config_phase, bias_spec=config.eval_bias_spec)
    expanded = expand_bias_specs(config.eval_bias_spec)
    feature_type_label = SIGNED_SIGNAL_FEATURE_TYPE.upper()
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

    data = load_signed_signal_research_data(config_phase, expanded, print_loaded=True)
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
    write_walkforward_artifacts(
        report=report,
        feature_type=SIGNED_SIGNAL_FEATURE_TYPE,
        module_name=str(config.eval_bias_spec["module_name"]),
        root_dir=output_dir_path,
        output_subdir=phase_subdir,
    )
    return report
