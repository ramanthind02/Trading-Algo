from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    get_available_date_ranges_for_tickers,
    get_effective_range_and_tickers,
    get_tickers_with_coverage_for_config,
    load_candles_for_config,
    populate_cache_if_needed,
)
from utils.evaluation.walkforward.config import WalkforwardResearchConfig
from utils.evaluation.walkforward.evaluators import build_signed_signal_walkforward_evaluator
from utils.evaluation.walkforward.metrics import resolve_objective_metric
from utils.evaluation.permutation_test.permutation_nulls import (
    _joblib_tqdm,
    aggregate_oos_metric_from_report,
)
from utils.evaluation.walkforward.research_data import (
    build_reference_target,
    load_portfolio_candles,
    load_signed_signal_research_data,
)
from utils.evaluation.walkforward.runner import run_walkforward_research


SIGNED_SIGNAL_FEATURE_TYPE = "signed_signal"


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
    covered_tickers = get_tickers_with_coverage_for_config(
        config,
        bias_spec=config.eval_bias_spec,
    )
    if not covered_tickers:
        effective = get_effective_range_and_tickers(
            config,
            bias_spec=config.eval_bias_spec,
        )
        if effective is not None:
            effective_start, effective_end, effective_tickers = effective
            config = replace(
                config,
                start=effective_start.to_pydatetime(),
                end=effective_end.to_pydatetime(),
                tickers=effective_tickers,
            )
        else:
            ranges = get_available_date_ranges_for_tickers(
                replace(config, tickers=original_tickers),
                original_tickers,
                bias_spec=config.eval_bias_spec,
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

    populate_cache_if_needed(config, bias_spec=config.eval_bias_spec)
    expanded = expand_bias_specs(config.eval_bias_spec)
    data = load_signed_signal_research_data(
        config,
        expanded,
    )
    if not data.successful_param_grid or data.reference_index is None:
        raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

    reference_target = build_reference_target(data.reference_index, data.reference_target_series)
    reference_candles = pd.DataFrame({"close": reference_target}, index=data.reference_index)
    portfolio_candles = load_portfolio_candles(config)
    evaluator = build_signed_signal_walkforward_evaluator(data.combo_signal_target)
    return (
        reference_candles,
        reference_target,
        data.successful_param_grid,
        evaluator,
        config,
        data.combo_signal_target,
        portfolio_candles,
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
    portfolio_candles_df: pd.DataFrame,
    train_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    fold_rows: list[dict],
    expanded: list,
    runtime_config: WalkforwardResearchConfig,
    module_name: str,
    objective_metric_name: str,
) -> float:
    from utils.evaluation.permutation_test.candle_shuffle import permute_walk_forward

    metric_fn = resolve_objective_metric(objective_metric_name)
    shuffled_candles = permute_walk_forward(
        candles_prepared,
        train_windows=train_windows,
        random_seed=seed,
    )
    data = load_signed_signal_research_data(
        config,
        expanded,
        candles_override=shuffled_candles,
    )
    if not data.successful_param_grid or data.reference_index is None:
        return 0.0
    ref_target = build_reference_target(data.reference_index, data.reference_target_series)
    ref_candles = pd.DataFrame({"close": ref_target}, index=data.reference_index)
    evaluator = build_signed_signal_walkforward_evaluator(data.combo_signal_target)
    report = run_walkforward_research(
        candles_df=ref_candles,
        target=ref_target,
        feature_type="signed_signal",
        module_name=module_name,
        config=runtime_config,
        param_grid=data.successful_param_grid,
        evaluate_param_combo=evaluator,
        research_config=config,
        portfolio_candles_df=portfolio_candles_df,
        feature_data_by_combo=data.combo_signal_target,
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

    expanded = expand_bias_specs(config.eval_bias_spec)
    _eval_spec = getattr(config, "eval_bias_spec", None)
    _bias_spec = getattr(config, "bias_spec", None)
    _spec = _eval_spec if _eval_spec is not None else (_bias_spec if _bias_spec is not None else {})
    module_name = str(_spec.get("module_name", "rsi"))
    rng = np.random.default_rng(random_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(nreps)]

    if n_jobs == 1:
        null_metrics = np.empty(nreps, dtype=float)
        for i in tqdm(range(nreps), desc="Candle shuffle", unit="rep"):
            null_metrics[i] = _one_candle_shuffle_rep(
                seeds[i],
                config,
                candles_prepared,
                portfolio_candles,
                train_windows,
                fold_rows,
                expanded,
                runtime_config,
                module_name,
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
                portfolio_candles,
                train_windows,
                fold_rows,
                expanded,
                runtime_config,
                module_name,
                objective_metric_name,
            )
            for seed in seeds
        )
    return np.array(results, dtype=float)


def run_permutation_for_phase(
    phase: Literal["oos", "validation"],
    config: object,
    args: argparse.Namespace,
) -> int:
    """Run permutation test for OOS or validation phase (shared implementation).

    Uses vector shuffle only for the null distribution (``run_vector_shuffle`` must be True).
    This function consolidates the main logic from both run_oos_permutation.py
    and run_validation_permutation.py. The main difference is the window source
    (config.oos_window vs config.validation_window).

    Parameters
    ----------
    phase : {"oos", "validation"}
        Which phase to run permutation test for.
    config : object
        Research configuration (must have oos_window or validation_window, bias_spec, etc.)
    args : argparse.Namespace
        Parsed command-line arguments (nreps, seed, n_jobs, output_dir).

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    # Import here to avoid circular dependencies and expensive imports at module level
    from feature_research.core_helpers import build_runtime_walkforward_config
    from feature_research.config import load_config
    from feature_research.in_sample.data_loader import load_candles_for_config
    from utils.evaluation.walkforward.config import WalkforwardResearchConfig
    from utils.evaluation.walkforward.io import resolve_walkforward_output_dir
    from utils.evaluation.walkforward.metrics import resolve_objective_metric
    from utils.evaluation.permutation_test.permutation_core import (
        _compute_fixed_oos_signal_by_fold,
        aggregate_per_ticker_metrics,
        aggregate_per_ticker_nulls,
        run_vector_shuffle_null,
    )
    from utils.evaluation.permutation_test.permutation_nulls import (
        aggregate_oos_metric_from_report,
        two_unit_masks_from_fold_rows,
    )
    from utils.evaluation.permutation_test.permutation_runtime import (
        build_report_payload,
        compute_significance,
        resolve_objective_metric_name,
        write_permutation_outputs,
    )
    from utils.evaluation.walkforward.runner import (
        build_fold_rows_from_explicit_specs,
        run_walkforward_research,
    )

    # Get window and phase-specific details
    if phase == "oos":
        # Helper for OOS effective window
        def _effective_oos_window(cfg):
            oos = getattr(cfg, "oos_window", None)
            if oos is None:
                raise ValueError("config.oos_window is not set.")
            validation = getattr(cfg, "validation_window", None)
            if validation is None:
                return oos
            from feature_research.config import OOSWindowConfig
            return OOSWindowConfig(
                train_start=validation.train_start,
                train_end=validation.test_end,
                test_start=oos.test_start,
                test_end=oos.test_end,
            )

        window = _effective_oos_window(config)
        phase_label = "OOS"
        phase_subdir = "oos"
        report_filename = "oos_permutation_report.json"
    elif phase == "validation":
        window = getattr(config, "validation_window", None)
        if window is None:
            raise ValueError("config.validation_window is not set.")
        phase_label = "Validation"
        phase_subdir = "validation"
        report_filename = "validation_permutation_report.json"
    else:
        print(f"Error: Unknown phase {phase}.")
        return 1

    # Determine run parameters
    perm_cfg = getattr(config, "permutation", None)
    random_seed = args.seed if args.seed is not None else getattr(perm_cfg, "random_seed", 42)
    alpha = getattr(perm_cfg, "alpha", 0.05)
    n_jobs = (
        args.n_jobs
        if args.n_jobs is not None
        else getattr(
            perm_cfg,
            "n_jobs_reps",
            getattr(perm_cfg, "n_jobs_stage1_reps", 1),
        )
    )
    run_vector_shuffle = getattr(
        perm_cfg,
        "run_vector_shuffle",
        getattr(perm_cfg, "run_stage1", True),
    )
    if not run_vector_shuffle:
        raise ValueError(
            "OOS/validation permutation in feature_research only supports vector shuffle; "
            "set permutation.run_vector_shuffle=True in feature_research.config."
        )
    effective_mode: str = "vector_shuffle"
    nreps = (
        args.nreps
        if args.nreps is not None
        else getattr(
            perm_cfg,
            "nreps",
            getattr(perm_cfg, "nreps_stage1", 200),
        )
    )

    # Build runtime config for objective metric resolution
    def _build_runtime_walkforward_config_for_permutation(
        cfg: object,
        *,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        objective_metric_name: str,
    ) -> WalkforwardResearchConfig:
        """Local wrapper for build_runtime_walkforward_config with objective_metric_name override."""
        test_step = max(1, int((test_end - test_start).days))
        return WalkforwardResearchConfig(
            train_start=train_start.to_pydatetime(),
            train_end=train_end.to_pydatetime(),
            enabled=True,
            test_step=test_step,
            num_steps=1,
            objective_metric_name=objective_metric_name,
            min_fold_samples=10,
            output_root=getattr(cfg, "output_root", Path("feature_research/shared_results")),
            n_jobs=getattr(cfg, "n_jobs", 1),
        )

    perm_obj = getattr(perm_cfg, "objective_metric", None)
    builtin = getattr(perm_obj, "builtin", None) if perm_obj else None
    _objective_from_perm = str(builtin) if builtin is not None else "sharpe"
    objective_metric_name = resolve_objective_metric_name(
        perm_cfg,
        _build_runtime_walkforward_config_for_permutation(
            config,
            train_start=pd.Timestamp(window.train_start),
            train_end=pd.Timestamp(window.train_end),
            test_start=pd.Timestamp(window.test_start),
            test_end=pd.Timestamp(window.test_end),
            objective_metric_name=_objective_from_perm,
        ),
    )

    _eval_spec = getattr(config, "eval_bias_spec", None)
    _bias_spec = getattr(config, "bias_spec", None)
    _spec = _eval_spec if _eval_spec is not None else (_bias_spec if _bias_spec is not None else {})
    module_name = str(_spec.get("module_name", "rsi"))
    tickers = list(getattr(config, "tickers", []))
    per_ticker_reports: dict[str, dict[str, float]] = {}
    per_ticker_nulls: dict[str, np.ndarray] = {}
    per_ticker_originals: dict[str, float] = {}
    skipped_tickers: list[str] = []

    # Local function for single permutation run
    def _run_permutation_once(
        config: object,
        lbl: str | None = None,
        return_return_matrix: bool = False,
    ) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, pd.Index, np.ndarray, pd.Series]:
        prefix = f"[{lbl}] " if lbl else ""

        data_start = min(config.start, window.train_start)
        data_end = max(config.end, window.test_end)
        config_phase = replace(config, start=data_start, end=data_end)

        (
            reference_candles,
            reference_target,
            param_grid,
            evaluator,
            _research_config,
            feature_data_by_combo,
            portfolio_candles_df,
        ) = load_research_data(config_phase)

        train_start = pd.Timestamp(window.train_start)
        train_end = pd.Timestamp(window.train_end)
        test_start = pd.Timestamp(window.test_start)
        test_end = pd.Timestamp(window.test_end)
        runtime_config = _build_runtime_walkforward_config_for_permutation(
            config,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            objective_metric_name=objective_metric_name,
        )

        fold_rows = build_fold_rows_from_explicit_specs(
            reference_target.index,
            [(window.train_start, window.train_end, window.test_start, window.test_end)],
            min_fold_samples=runtime_config.min_fold_samples,
        )
        if not fold_rows:
            raise ValueError(f"{phase_label} fold has insufficient samples. Check {phase} window dates and data range.")

        if feature_data_by_combo is None:
            raise ValueError("Frozen-signal permutation requires feature_data_by_combo.")
        if portfolio_candles_df is None:
            portfolio_candles_df = load_candles_for_config(config_phase)

        print(f"{prefix}Original (unpermuted) {phase_label} run...")
        report0 = run_walkforward_research(
            candles_df=reference_candles,
            target=reference_target,
            feature_type="signed_signal",
            module_name=module_name,
            config=runtime_config,
            param_grid=param_grid,
            evaluate_param_combo=evaluator,
            research_config=config,
            portfolio_candles_df=portfolio_candles_df,
            feature_data_by_combo=feature_data_by_combo,
            output_dir=None,
            fold_rows_override=fold_rows,
        )
        metric_fn = resolve_objective_metric(objective_metric_name)
        original_metric = aggregate_oos_metric_from_report(
            report0,
            treat_no_selection_as_zero=True,
            metric_fn=metric_fn,
        )
        print(f"{prefix}  Original aggregate {phase_label} metric ({objective_metric_name}): {original_metric:.4f}")

        agg_returns = getattr(report0, "aggregate_oos_returns", None)
        agg_returns_clean = agg_returns.dropna() if agg_returns is not None else pd.Series(dtype=float)
        canonical_oos_index = pd.Index([], dtype="datetime64[ns]")
        return_matrix: np.ndarray | None = None

        unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(reference_target.index, fold_rows)
        fixed_oos_signal_by_fold = _compute_fixed_oos_signal_by_fold(
            fold_rows=fold_rows,
            selection_summary_df=report0.selection_summary_df,
            feature_data_by_combo=feature_data_by_combo,
        )
        if not fixed_oos_signal_by_fold:
            print(f"{prefix}Frozen-signal extraction failed; null distribution set to zeros.")
            null_metrics = np.zeros(nreps, dtype=float)
            canonical_oos_index = pd.Index([], dtype="datetime64[ns]")
            return_matrix = np.zeros((nreps, 0), dtype=float) if return_return_matrix else None
        else:
            canonical_oos_index = (
                agg_returns_clean.index if agg_returns is not None else pd.Index([], dtype="datetime64[ns]")
            )
            print(f"{prefix}Running vector shuffle null (nreps={nreps})...")
            null_result = run_vector_shuffle_null(
                reference_candles=reference_candles,
                reference_target=reference_target,
                fold_rows=fold_rows,
                unit1_mask=unit1_mask,
                unit2_mask=unit2_mask,
                nreps=nreps,
                random_seed=random_seed,
                initial_report=report0,
                research_config=config,
                feature_data_by_combo=feature_data_by_combo,
                portfolio_candles_df=portfolio_candles_df,
                n_jobs=n_jobs,
                fixed_oos_signal_by_fold=fixed_oos_signal_by_fold,
                objective_metric_name=objective_metric_name,
                canonical_oos_index=canonical_oos_index,
                return_returns=return_return_matrix,
            )
            if return_return_matrix:
                null_metrics, canonical_oos_index, return_matrix = null_result
            else:
                null_metrics = null_result
                return_matrix = None

        if return_return_matrix:
            canonical_index = canonical_oos_index if canonical_oos_index is not None else pd.Index([], dtype="datetime64[ns]")
            matrix = return_matrix if return_matrix is not None else np.zeros((nreps, 0), dtype=float)
            return original_metric, null_metrics, canonical_index, matrix, agg_returns_clean
        return original_metric, null_metrics

    # Run permutation for each ticker (or single run)
    if len(tickers) > 1:
        print(f"Multiple tickers detected ({len(tickers)}); running per-ticker permutation.")
        for ticker in tickers:
            label = getattr(ticker, "name", str(ticker))
            config_t = replace(config, tickers=[ticker])
            try:
                original_metric, null_metrics = _run_permutation_once(
                    config=config_t,
                    lbl=label,
                    return_return_matrix=False,
                )
            except Exception as exc:
                print(f"[{label}] Error: {exc}")
                skipped_tickers.append(label)
                continue
            if null_metrics.size == 0:
                print(f"[{label}] Warning: empty null distribution; skipping.")
                skipped_tickers.append(label)
                continue
            p_value, critical_value, passed = compute_significance(
                original_metric=original_metric,
                null_metrics=null_metrics,
                nreps=nreps,
                alpha=alpha,
            )
            per_ticker_originals[label] = original_metric
            per_ticker_nulls[label] = null_metrics
            per_ticker_reports[label] = {
                "original_metric": float(original_metric),
                "p_value": float(p_value),
                "critical_value": float(critical_value),
                "passed": bool(passed),
            }
        if not per_ticker_originals:
            print("Error: no tickers produced permutation results.")
            return 1
        original_metric = aggregate_per_ticker_metrics(per_ticker_originals)
        null_metrics = aggregate_per_ticker_nulls(per_ticker_nulls)
    else:
        original_metric, null_metrics = _run_permutation_once(
            config=config,
            lbl=None,
            return_return_matrix=False,
        )

    p_value, critical_value, passed = compute_significance(
        original_metric=original_metric,
        null_metrics=null_metrics,
        nreps=nreps,
        alpha=alpha,
    )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = (
            resolve_walkforward_output_dir(
                feature_type=SIGNED_SIGNAL_FEATURE_TYPE,
                module_name=module_name,
                root_dir=getattr(config, "output_root", Path("feature_research/shared_results")),
                output_subdir=phase_subdir,
            )
            / "permutation"
        )
    report = build_report_payload(
        effective_mode=effective_mode,
        nreps=nreps,
        random_seed=random_seed,
        alpha=alpha,
        original_metric=original_metric,
        p_value=p_value,
        critical_value=critical_value,
        passed=passed,
        per_ticker_reports=per_ticker_reports or None,
        skipped_tickers=skipped_tickers,
    )
    report_path = write_permutation_outputs(
        out_dir=out_dir,
        report_filename=report_filename,
        report_payload=report,
        null_metrics=null_metrics,
        per_ticker_nulls=per_ticker_nulls or None,
    )

    print(f"\n{phase_label} permutation ({effective_mode})")
    print(f"  nreps={nreps}  alpha={alpha}")
    print(f"  Original metric: {original_metric:.4f}  Critical: {critical_value:.4f}")
    print(f"  p-value: {p_value:.4f}  Passed: {passed}")
    if per_ticker_reports:
        print("  Per-ticker summary:")
        for label, stats in per_ticker_reports.items():
            print(
                f"    {label}: p={stats['p_value']:.4f} "
                f"critical={stats['critical_value']:.4f} "
                f"original={stats['original_metric']:.4f} "
                f"passed={stats['passed']}"
            )
    print(f"  Report: {report_path}  Null dist: {out_dir / 'null_distribution.npy'}")
    return 0
