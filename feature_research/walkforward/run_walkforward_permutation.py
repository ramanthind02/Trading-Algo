"""Walkforward permutation testing: vector (target) shuffle and candle shuffle.

Two-unit blocking: first fold train vs all remaining data. No mixing between units.
When no stable regions are found for a fold, that fold contributes 0 to the OOS metric.
When multiple tickers are configured, permutation runs per ticker and aggregates the
metrics/null distribution by mean across tickers.

Usage:
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/walkforward/run_walkforward_permutation.py [--mode vector_shuffle|candle_shuffle] [--nreps 100] [--seed 42]

Config: uses load_config() and in_sample_permutation for nreps, alpha, random_seed.
When in_sample_permutation.run_stage1 is False, vector_shuffle is skipped and candle_shuffle runs instead (even if --mode vector_shuffle).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_repo_hint = Path(__file__).resolve().parents[2]
if str(_repo_hint) not in sys.path:
    sys.path.insert(0, str(_repo_hint))

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.config import FeatureType
from feature_research.core_helpers import expand_params_with_selected_bin
from feature_research.in_sample.config import load_config
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    get_available_date_ranges_for_tickers,
    get_tickers_with_coverage_for_config,
    load_candles_for_config,
    populate_cache_if_needed,
)
from feature_research.walkforward.evaluators import (
    build_continuous_walkforward_evaluator,
    build_rule_based_walkforward_evaluator,
)
from feature_research.walkforward.io import resolve_walkforward_output_dir
from feature_research.walkforward.permutation_helpers import (
    _joblib_tqdm,
    aggregate_oos_metric_from_report,
    two_unit_masks_from_fold_rows,
)
from feature_research.walkforward.metrics import resolve_objective_metric
from feature_research.walkforward.permutation_runtime import (
    apply_objective_metric,
    build_report_payload,
    compute_significance,
    resolve_objective_metric_name,
    write_permutation_outputs,
)
from feature_research.walkforward.permutation_core import (
    _compute_fixed_oos_signal_by_fold,
    _compute_rule_based_oos_signal_by_fold,
    aggregate_per_ticker_metrics,
    aggregate_per_ticker_nulls,
    run_vector_shuffle_null,
)
from feature_research.walkforward.research_data import (
    build_reference_target,
    load_continuous_research_data,
    load_portfolio_candles,
    load_rule_based_research_data,
)
from feature_research.walkforward.runner import (
    _build_fold_rows,
    run_walkforward_research,
)


def _run_walkforward_permutation_once(
    config: object,
    effective_mode: str,
    nreps: int,
    random_seed: int | None,
    n_jobs: int,
    objective_metric_name: str,
    *,
    label: str | None = None,
) -> tuple[float, np.ndarray, pd.Series]:
    prefix = f"[{label}] " if label else ""

    (
        reference_candles,
        reference_target,
        param_grid,
        evaluator,
        research_config,
        feature_data_by_combo,
        portfolio_candles_df,
    ) = _load_walkforward_data(config)

    wf_config = getattr(config, "walkforward", None)
    fold_rows = _build_fold_rows(reference_target.index, wf_config)
    if not fold_rows:
        raise ValueError("No walkforward folds; cannot run permutation.")

    module_name = str(getattr(config, "bias_spec", {}).get("module_name", "rsi"))
    feature_type = "continuous" if feature_data_by_combo is not None else "rule_based"

    if portfolio_candles_df is None:
        portfolio_candles_df = load_candles_for_config(config)

    print(f"{prefix}Original (unpermuted) walkforward run...")
    report0 = run_walkforward_research(
        candles_df=reference_candles,
        target=reference_target,
        feature_type=feature_type,
        module_name=module_name,
        config=wf_config,
        param_grid=param_grid,
        evaluate_param_combo=evaluator,
        research_config=config,
        portfolio_candles_df=portfolio_candles_df,
        feature_data_by_combo=feature_data_by_combo,
        output_dir=None,
    )
    metric_fn = resolve_objective_metric(objective_metric_name)
    original_metric = aggregate_oos_metric_from_report(
        report0, treat_no_selection_as_zero=True, metric_fn=metric_fn
    )
    print(f"{prefix}  Original aggregate OOS metric ({objective_metric_name}): {original_metric:.4f}")

    agg_returns = getattr(report0, "aggregate_oos_returns", None)
    agg_returns_clean = agg_returns.dropna() if agg_returns is not None else pd.Series(dtype=float)

    if effective_mode == "vector_shuffle":
        if feature_data_by_combo is not None:
            unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(reference_target.index, fold_rows)
            fixed_oos_signal_by_fold = _compute_fixed_oos_signal_by_fold(
                fold_rows=fold_rows,
                selection_summary_df=report0.selection_summary_df,
                reference_target=reference_target,
                research_config=config,
                feature_data_by_combo=feature_data_by_combo,
            )
            print(f"{prefix}Running vector shuffle null for continuous (nreps={nreps})...")
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
            )
        else:
            if agg_returns_clean.empty:
                print(f"{prefix}Rule-based aggregate OOS returns are empty; null distribution set to zeros.")
                null_metrics = np.zeros(nreps, dtype=float)
            else:
                unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(reference_target.index, fold_rows)
                canonical_oos_index = agg_returns_clean.index
                fixed_oos_signal_by_fold = _compute_rule_based_oos_signal_by_fold(
                    fold_rows=fold_rows,
                    reference_candles=reference_candles,
                    reference_target=reference_target,
                    selection_summary_df=report0.selection_summary_df,
                    research_config=config,
                    portfolio_candles_df=portfolio_candles_df,
                )
                if not fixed_oos_signal_by_fold:
                    # Fallback: correct but slow (refit per replicate). Prefer fixing the upstream
                    # selection/portfolio path so we can extract a fixed OOS signal.
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
                )
    else:
        print(f"{prefix}Running candle shuffle null (nreps={nreps})...")
        null_metrics = run_candle_shuffle_null(
            config=config,
            reference_candles=reference_candles,
            reference_target=reference_target,
            param_grid=param_grid,
            fold_rows=fold_rows,
            nreps=nreps,
            random_seed=random_seed,
            objective_metric_name=objective_metric_name,
            n_jobs=n_jobs,
        )

    return original_metric, null_metrics, agg_returns_clean


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walkforward permutation test (vector or candle shuffle, two-unit blocking)."
    )
    parser.add_argument(
        "--mode",
        choices=("vector_shuffle", "candle_shuffle"),
        default="vector_shuffle",
        help="Permutation mode: target permutation (fast) or candle shuffle + re-extract (slow).",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for report and null distribution (default: output_root/permutation/).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for vector-shuffle replicates (default from config or 1). -1 = all CPUs.",
    )
    return parser.parse_args()


def _load_research_data(config: object) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[dict],
    object,
    object | None,
    dict | None,
    pd.DataFrame | None,
]:
    """Build candles, target, param_grid, evaluator, research_config, feature_data_by_combo, portfolio_candles.

    Mirrors run_walkforward_pipeline data loading for CONTINUOUS and RULE_BASED.
    Shared by walkforward and OOS permutation scripts.
    """
    original_tickers = list(config.tickers)
    covered_tickers = get_tickers_with_coverage_for_config(config)
    if len(covered_tickers) < len(original_tickers):
        dropped = set(original_tickers) - set(covered_tickers)
        print(
            f"[walkforward_permutation] Tickers without full date coverage for "
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
        reference_candles = pd.DataFrame({"close": reference_target}, index=continuous_data.reference_index)
        portfolio_candles = load_portfolio_candles(config)
        evaluator = build_continuous_walkforward_evaluator(continuous_data.combo_feature_target, config)
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

    reference_target = build_reference_target(rule_data.reference_index, rule_data.reference_target_series)
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


def _load_walkforward_data(config: object) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[dict],
    object,
    object | None,
    dict | None,
    pd.DataFrame | None,
]:
    """Load research data for walkforward permutation (delegates to _load_research_data)."""
    return _load_research_data(config)


def _two_unit_train_windows(
    fold_rows: list[dict],
    last_ts: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return [(train_start_0, train_end_0), (test_start_0, end_exclusive)] for permute_walk_forward."""
    if not fold_rows:
        return []
    first = fold_rows[0]
    train_start = pd.Timestamp(first["train_start"])
    train_end = pd.Timestamp(first["train_end"])
    test_start = pd.Timestamp(first["test_start"])
    end_exclusive = last_ts + pd.Timedelta(days=1)
    return [(train_start, train_end), (test_start, end_exclusive)]


def _prepare_candles_for_shuffler(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime column for permute_walk_forward."""
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
    wf_config: object,
    module_name: str,
    feature_type: FeatureType,
    objective_metric_name: str,
) -> float:
    """One candle-shuffle null replicate. Module-level for joblib pickling when n_jobs > 1."""
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
        evaluator = build_continuous_walkforward_evaluator(continuous_data.combo_feature_target, config)
        portfolio_candles_df = load_portfolio_candles(config)
        report = run_walkforward_research(
            candles_df=ref_candles,
            target=ref_target,
            feature_type="continuous",
            module_name=module_name,
            config=wf_config,
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
            config=wf_config,
            param_grid=rule_data.successful_param_grid,
            evaluate_param_combo=evaluator,
            research_config=config,
            output_dir=None,
            fold_rows_override=fold_rows,
        )
    return float(
        aggregate_oos_metric_from_report(
            report, treat_no_selection_as_zero=True, metric_fn=metric_fn
        )
    )


def run_candle_shuffle_null(
    config: object,
    reference_candles: pd.DataFrame,
    reference_target: pd.Series,
    param_grid: list[dict],
    fold_rows: list[dict],
    nreps: int,
    random_seed: int | None,
    objective_metric_name: str = "sharpe",
    n_jobs: int = 1,
) -> np.ndarray:
    """Run nreps walkforward with candles permuted in two units and re-extraction; return null distribution.

    n_jobs: parallel jobs for replicates (default 1). -1 = all CPUs.
    When n_jobs > 1, replicates run in parallel via joblib (loky backend).
    """
    portfolio_candles = load_portfolio_candles(config)
    candles_prepared = _prepare_candles_for_shuffler(portfolio_candles)
    if "datetime" not in candles_prepared.columns:
        candles_prepared["datetime"] = candles_prepared.index
    last_ts = pd.Timestamp(reference_target.index.max())
    train_windows = _two_unit_train_windows(fold_rows, last_ts)
    if not train_windows:
        return np.array([])

    expanded = expand_bias_specs(config.bias_spec)
    wf_config = getattr(config, "walkforward", None)
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
                wf_config,
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
                wf_config,
                module_name,
                feature_type,
                objective_metric_name,
            )
            for seed in seeds
        )
    return np.array(results, dtype=float)


def main() -> int:
    args = _parse_args()
    config = load_config()
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
    # When config says run_stage1=False, skip vector shuffle and run candle shuffle (stage 2) only.
    run_stage1 = getattr(perm_cfg, "run_stage1", True)
    effective_mode: str = (
        "candle_shuffle"
        if (args.mode == "vector_shuffle" and not run_stage1)
        else args.mode
    )
    if effective_mode != args.mode:
        print(f"Config run_stage1=False: running {effective_mode} instead of {args.mode}.")
    nreps = (
        args.nreps
        if args.nreps is not None
        else (
            getattr(perm_cfg, "nreps_stage1", 200)
            if effective_mode == "vector_shuffle"
            else getattr(perm_cfg, "nreps_stage2", 100)
        )
    )

    wf_config = getattr(config, "walkforward", None)
    objective_metric_name = resolve_objective_metric_name(perm_cfg, wf_config)
    print(f"Objective metric (from config.permutation): {objective_metric_name}")
    config = apply_objective_metric(config, objective_metric_name)
    wf_config = getattr(config, "walkforward", None)

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
                original_metric, null_metrics, _ticker_returns = _run_walkforward_permutation_once(
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
        # Aggregate across tickers by mean-of-metrics (consistent with per-ticker permutation).
        # Avoid "return-shuffle" aggregation, which is degenerate for our supported metrics.
        original_metric = aggregate_per_ticker_metrics(per_ticker_originals)
        null_metrics = aggregate_per_ticker_nulls(per_ticker_nulls)
    else:
        original_metric, null_metrics = _run_walkforward_permutation_once(
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
        out_dir = resolve_walkforward_output_dir(
            feature_type=getattr(config, "feature_type", FeatureType.CONTINUOUS).value,
            module_name=module_name,
            root_dir=wf_config.output_root,
        ) / "permutation"
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
        report_filename="walkforward_permutation_report.json",
        report_payload=report,
        null_metrics=null_metrics,
        per_ticker_nulls=per_ticker_nulls or None,
    )

    print(f"\nWalkforward permutation ({effective_mode})")
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
