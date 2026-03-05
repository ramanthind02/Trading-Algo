"""OOS permutation test (vector-shuffle or candle-shuffle, single fold from config.oos_window)."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_repo_hint = Path(__file__).resolve().parents[2]
if str(_repo_hint) not in sys.path:
    sys.path.insert(0, str(_repo_hint))

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.config import FeatureType, OOSWindowConfig
from feature_research.in_sample.config import load_config
from feature_research.in_sample.data_loader import load_candles_for_config
from feature_research.validation.permutation_helpers import (
    load_research_data,
    run_candle_shuffle_null,
)
from utils.evaluation.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
)
from utils.evaluation.walkforward.io import resolve_walkforward_output_dir
from utils.evaluation.walkforward.metrics import resolve_objective_metric
from utils.evaluation.walkforward.permutation_core import (
    _compute_fixed_oos_signal_by_fold,
    _compute_rule_based_oos_signal_by_fold,
    aggregate_per_ticker_metrics,
    aggregate_per_ticker_nulls,
    run_vector_shuffle_null,
)
from utils.evaluation.walkforward.permutation_helpers import (
    aggregate_oos_metric_from_report,
    two_unit_masks_from_fold_rows,
)
from utils.evaluation.walkforward.permutation_runtime import (
    build_report_payload,
    compute_significance,
    resolve_objective_metric_name,
    write_permutation_outputs,
)
from utils.evaluation.walkforward.runner import (
    build_fold_rows_from_explicit_specs,
    run_walkforward_research,
)


def _require_oos_window(config: object) -> OOSWindowConfig:
    oos = getattr(config, "oos_window", None)
    if oos is None:
        raise ValueError("config.oos_window is not set.")
    return oos


def _effective_oos_window(config: object) -> OOSWindowConfig:
    oos = _require_oos_window(config)
    validation = getattr(config, "validation_window", None)
    if validation is None:
        return oos
    return OOSWindowConfig(
        train_start=validation.train_start,
        train_end=validation.test_end,
        test_start=oos.test_start,
        test_end=oos.test_end,
    )


def _build_runtime_walkforward_config(
    config: object,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    objective_metric_name: str,
) -> WalkforwardResearchConfig:
    test_step = max(1, int((test_end - test_start).days))
    return WalkforwardResearchConfig(
        train_start=train_start.to_pydatetime(),
        train_end=train_end.to_pydatetime(),
        enabled=True,
        test_step=test_step,
        num_steps=1,
        top_k=getattr(config, "top_k", 1),
        objective_metric_name=objective_metric_name,
        min_fold_samples=10,
        output_root=getattr(config, "output_root", Path("feature_research/shared_results")),
        selection_method=WalkforwardSelectionMethod.TOP_K,
        n_jobs=getattr(config, "n_jobs", 1),
        smoothing_self_weight=getattr(config, "smoothing_self_weight", 1.0),
    )


def _run_oos_permutation_once(
    config: object,
    effective_mode: str,
    nreps: int,
    random_seed: int | None,
    n_jobs: int,
    objective_metric_name: str,
    *,
    label: str | None = None,
    return_return_matrix: bool = False,
) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, pd.Index, np.ndarray, pd.Series]:
    prefix = f"[{label}] " if label else ""
    window = _effective_oos_window(config)

    data_start = min(config.start, window.train_start)
    data_end = max(config.end, window.test_end)
    config_oos = replace(config, start=data_start, end=data_end)

    (
        reference_candles,
        reference_target,
        param_grid,
        evaluator,
        _research_config,
        feature_data_by_combo,
        portfolio_candles_df,
    ) = load_research_data(config_oos)

    train_start = pd.Timestamp(window.train_start)
    train_end = pd.Timestamp(window.train_end)
    test_start = pd.Timestamp(window.test_start)
    test_end = pd.Timestamp(window.test_end)
    runtime_config = _build_runtime_walkforward_config(
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
        raise ValueError("OOS fold has insufficient samples. Check oos_window dates and data range.")

    module_name = str(getattr(config, "bias_spec", {}).get("module_name", "rsi"))
    _ft = getattr(config, "feature_type", None)
    feature_type = (
        _ft.value if isinstance(_ft, FeatureType) else
        ("continuous" if feature_data_by_combo is not None else "rule_based")
    )

    if portfolio_candles_df is None:
        portfolio_candles_df = load_candles_for_config(config_oos)

    print(f"{prefix}Original (unpermuted) OOS run...")
    report0 = run_walkforward_research(
        candles_df=reference_candles,
        target=reference_target,
        feature_type=feature_type,
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
    print(f"{prefix}  Original aggregate OOS metric ({objective_metric_name}): {original_metric:.4f}")

    agg_returns = getattr(report0, "aggregate_oos_returns", None)
    agg_returns_clean = agg_returns.dropna() if agg_returns is not None else pd.Series(dtype=float)
    canonical_oos_index = pd.Index([], dtype="datetime64[ns]")
    return_matrix: np.ndarray | None = None
    if effective_mode == "vector_shuffle":
        unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(reference_target.index, fold_rows)
        if feature_data_by_combo is not None:
            fixed_oos_signal_by_fold = _compute_fixed_oos_signal_by_fold(
                fold_rows=fold_rows,
                selection_summary_df=report0.selection_summary_df,
                reference_target=reference_target,
                research_config=config,
                feature_data_by_combo=feature_data_by_combo,
            )
            canonical_oos_index = (
                agg_returns_clean.index if agg_returns is not None else pd.Index([], dtype="datetime64[ns]")
            )
            print(f"{prefix}Running vector shuffle null (nreps={nreps})...")
            if return_return_matrix:
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
                    return_returns=True,
                )
                null_metrics, canonical_oos_index, return_matrix = null_result
            else:
                null_metrics = run_vector_shuffle_null(
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
                    return_returns=False,
                )
                return_matrix = None
        else:
            if agg_returns is None or agg_returns_clean.empty:
                print(f"{prefix}Rule-based aggregate OOS returns are empty; null distribution set to zeros.")
                null_metrics = np.zeros(nreps, dtype=float)
                canonical_oos_index = pd.Index([], dtype="datetime64[ns]")
                return_matrix = np.zeros((nreps, 0), dtype=float) if return_return_matrix else None
            else:
                canonical_oos_index = (
                    agg_returns_clean.index if agg_returns is not None else pd.Index([], dtype="datetime64[ns]")
                )
                fixed_oos_signal_by_fold = _compute_rule_based_oos_signal_by_fold(
                    fold_rows=fold_rows,
                    reference_candles=reference_candles,
                    reference_target=reference_target,
                    selection_summary_df=report0.selection_summary_df,
                    research_config=config,
                    portfolio_candles_df=portfolio_candles_df,
                )
                if not fixed_oos_signal_by_fold:
                    print(
                        f"{prefix}Rule-based fixed OOS signal extraction failed/empty; "
                        f"falling back to legacy per-rep refit (nreps={nreps})..."
                    )
                    fixed_oos_signal_by_fold = None
                else:
                    print(
                        f"{prefix}Running vector shuffle null for rule-based via fixed OOS signal "
                        f"(nreps={nreps})..."
                    )
                if return_return_matrix:
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
                        feature_data_by_combo=None,
                        portfolio_candles_df=portfolio_candles_df,
                        n_jobs=n_jobs,
                        fixed_oos_signal_by_fold=fixed_oos_signal_by_fold,
                        objective_metric_name=objective_metric_name,
                        canonical_oos_index=canonical_oos_index,
                        return_returns=True,
                    )
                    null_metrics, canonical_oos_index, return_matrix = null_result
                else:
                    null_metrics = run_vector_shuffle_null(
                        reference_candles=reference_candles,
                        reference_target=reference_target,
                        fold_rows=fold_rows,
                        unit1_mask=unit1_mask,
                        unit2_mask=unit2_mask,
                        nreps=nreps,
                        random_seed=random_seed,
                        initial_report=report0,
                        research_config=config,
                        feature_data_by_combo=None,
                        portfolio_candles_df=portfolio_candles_df,
                        n_jobs=n_jobs,
                        fixed_oos_signal_by_fold=fixed_oos_signal_by_fold,
                        objective_metric_name=objective_metric_name,
                        canonical_oos_index=canonical_oos_index,
                        return_returns=False,
                    )
                    return_matrix = None
    else:
        print(f"{prefix}Running candle shuffle null (nreps={nreps})...")
        null_metrics = run_candle_shuffle_null(
            config=config_oos,
            runtime_config=runtime_config,
            reference_target=reference_target,
            fold_rows=fold_rows,
            nreps=nreps,
            random_seed=random_seed,
            objective_metric_name=objective_metric_name,
            n_jobs=n_jobs,
        )

    if return_return_matrix:
        canonical_index = canonical_oos_index if canonical_oos_index is not None else pd.Index([], dtype="datetime64[ns]")
        matrix = return_matrix if return_matrix is not None else np.zeros((nreps, 0), dtype=float)
        return original_metric, null_metrics, canonical_index, matrix, agg_returns_clean
    return original_metric, null_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOS permutation test (vector-shuffle only, single fold from config.oos_window)."
    )
    parser.add_argument(
        "--nreps",
        type=int,
        default=None,
        help="Number of replicates (default from config: nreps_stage1 for vector_shuffle, nreps_stage2 for candle_shuffle).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default from config.in_sample_permutation.random_seed).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Parallel jobs for vector-shuffle replicates (default from config or 1). -1 = all CPUs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for report and null distribution (default: output_root/.../oos/permutation/).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config()
    if config.oos_window is None:
        print("Error: config.oos_window is not set. Set it in feature_research.config.load_config().")
        return 1

    perm_cfg = getattr(config, "in_sample_permutation", None) or getattr(
        config, "permutation_suite", None
    )
    random_seed = args.seed if args.seed is not None else getattr(perm_cfg, "random_seed", 42)
    alpha = getattr(perm_cfg, "alpha", 0.05)
    n_jobs = (
        args.n_jobs
        if args.n_jobs is not None
        else getattr(perm_cfg, "n_jobs_stage1_reps", 1)
    )
    run_stage1 = getattr(perm_cfg, "run_stage1", True)
    effective_mode: str = "candle_shuffle" if not run_stage1 else "vector_shuffle"
    if effective_mode != "vector_shuffle":
        print("Config run_stage1=False: running candle_shuffle instead of vector_shuffle.")
    nreps = (
        args.nreps
        if args.nreps is not None
        else (
            getattr(perm_cfg, "nreps_stage1", 200)
            if effective_mode == "vector_shuffle"
            else getattr(perm_cfg, "nreps_stage2", 100)
        )
    )

    objective_metric_name = resolve_objective_metric_name(
        perm_cfg,
        _build_runtime_walkforward_config(
            config,
            train_start=pd.Timestamp(_effective_oos_window(config).train_start),
            train_end=pd.Timestamp(_effective_oos_window(config).train_end),
            test_start=pd.Timestamp(_effective_oos_window(config).test_start),
            test_end=pd.Timestamp(_effective_oos_window(config).test_end),
            objective_metric_name=getattr(config, "objective_metric_name", "sharpe"),
        ),
    )

    module_name = str(getattr(config, "bias_spec", {}).get("module_name", "rsi"))

    tickers = list(getattr(config, "tickers", []))
    per_ticker_reports: dict[str, dict[str, float]] = {}
    per_ticker_nulls: dict[str, np.ndarray] = {}
    per_ticker_originals: dict[str, float] = {}
    skipped_tickers: list[str] = []

    if len(tickers) > 1:
        print(f"Multiple tickers detected ({len(tickers)}); running per-ticker permutation.")
        for ticker in tickers:
            label = getattr(ticker, "name", str(ticker))
            config_t = replace(config, tickers=[ticker])
            try:
                original_metric, null_metrics = _run_oos_permutation_once(
                    config=config_t,
                    effective_mode=effective_mode,
                    nreps=nreps,
                    random_seed=random_seed,
                    n_jobs=n_jobs,
                    objective_metric_name=objective_metric_name,
                    label=label,
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
        original_metric, null_metrics = _run_oos_permutation_once(
            config=config,
            effective_mode=effective_mode,
            nreps=nreps,
            random_seed=random_seed,
            n_jobs=n_jobs,
            objective_metric_name=objective_metric_name,
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
                feature_type=getattr(config, "feature_type", FeatureType.CONTINUOUS).value,
                module_name=module_name,
                root_dir=getattr(config, "output_root", Path("feature_research/shared_results")),
                output_subdir="oos",
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
        report_filename="oos_permutation_report.json",
        report_payload=report,
        null_metrics=null_metrics,
        per_ticker_nulls=per_ticker_nulls or None,
    )

    print(f"\nOOS permutation ({effective_mode})")
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


if __name__ == "__main__":
    sys.exit(main())
