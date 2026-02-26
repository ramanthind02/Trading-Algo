"""Walkforward permutation testing: vector (target) shuffle and candle shuffle.

Two-unit blocking: first fold train vs all remaining data. No mixing between units.
When no stable regions are found for a fold, that fold contributes 0 to the OOS metric.

Usage:
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/walkforward/run_walkforward_permutation.py [--mode vector_shuffle|candle_shuffle] [--nreps 100] [--seed 42]

Config: uses load_config() and in_sample_permutation for nreps, alpha, random_seed.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Repo root for imports
def _find_repo_root(start: Path) -> Path | None:
    search_root = start if start.is_dir() else start.parent
    for parent in (search_root, *search_root.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return None


_repo_root = _find_repo_root(Path(__file__).resolve())
if _repo_root is not None and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from feature_research.config import FeatureType
from feature_research.in_sample.config import load_config
from feature_research.in_sample.data_loader import (
    expand_bias_specs,
    load_candles_for_config,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_research.pipeline import (
    _build_continuous_walkforward_evaluator,
    _build_rule_based_walkforward_evaluator,
    _combo_key,
    _expand_params_with_bin_count,
    _expand_params_with_selected_bin,
    _normalize_datetime_index,
    _normalize_series_datetime_index,
    _unique_sorted_datetime_index,
)
from feature_research.walkforward.io import resolve_walkforward_output_dir
from feature_research.walkforward.permutation_helpers import (
    aggregate_oos_metric_from_report,
    permute_target_in_two_units,
    two_unit_masks_from_fold_rows,
)
from feature_research.walkforward.metrics import resolve_objective_metric
from feature_research.walkforward.portfolio_evaluator import (
    _build_one_base_model_with_members,
    _normalize_strategy,
    _normalize_timeframe,
)
from feature_research.walkforward.permutation_core import (
    _compute_fixed_oos_signal_by_fold,
    aggregate_signal_target_returns,
    run_vector_shuffle_null,
)
from feature_research.walkforward.runner import (
    _build_fold_rows,
    run_walkforward_research,
)
from utils.core.enums import TimeFrame


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
        help="Number of replicates (default from config.in_sample_permutation.nreps).",
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
    from feature_research.config import FeatureType as FT

    populate_cache_if_needed(config)
    expanded = expand_bias_specs(config.bias_spec)
    feature_type = getattr(config, "feature_type", FT.CONTINUOUS)

    if feature_type == FeatureType.CONTINUOUS:
        combo_feature_target: dict = {}
        successful_param_grid: list[dict] = []
        reference_index: pd.DatetimeIndex | None = None
        reference_target_series: pd.Series | None = None

        for single_spec in expanded:
            combo = single_spec["params"]
            label = param_combo_label(combo)
            data = load_features_for_combo(single_spec, config)
            if data is None:
                continue
            feature, target, _ = data
            paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
            if paired.empty:
                continue
            feature = paired["feature"]
            target = paired["target"]
            expanded_combo_params = _expand_params_with_bin_count(
                params=dict(combo),
                bin_counts=config.binning_params.bin_counts,
            )
            normalized_feature = _normalize_series_datetime_index(feature)
            normalized_target = _normalize_series_datetime_index(target)
            for combo_params in expanded_combo_params:
                combo_feature_target[_combo_key(combo_params)] = pd.DataFrame({
                    "feature": normalized_feature,
                    "target": normalized_target,
                })
                successful_param_grid.append(combo_params)
            if reference_index is None:
                reference_index = _normalize_datetime_index(target.index)
                reference_target_series = normalized_target.reindex(reference_index)

        if not successful_param_grid or reference_index is None:
            raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

        successful_param_grid = _expand_params_with_selected_bin(
            successful_param_grid,
            bin_index_min=config.binning_params.bin_index_min,
            bin_index_max=config.binning_params.bin_index_max,
        )
        reference_target = (
            reference_target_series.fillna(0.0).rename("walkforward_target")
            if reference_target_series is not None
            else pd.Series(0.0, index=reference_index, name="walkforward_target")
        )
        reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
        portfolio_candles = load_candles_for_config(config)
        evaluator = _build_continuous_walkforward_evaluator(combo_feature_target, config)
        return (
            reference_candles,
            reference_target,
            successful_param_grid,
            evaluator,
            config,
            combo_feature_target,
            portfolio_candles,
        )

    # RULE_BASED
    combo_returns: dict = {}
    successful_param_grid = []
    reference_index = None

    for single_spec in expanded:
        combo = single_spec["params"]
        data = load_features_for_combo(single_spec, config)
        if data is None:
            continue
        feature, target, _ = data
        paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if paired.empty:
            continue
        feature = paired["feature"]
        target = paired["target"]
        combo_returns[_combo_key(combo)] = _normalize_series_datetime_index(feature.mul(target))
        successful_param_grid.append(dict(combo))
        if reference_index is None:
            reference_index = _unique_sorted_datetime_index(target.index)

    if not successful_param_grid or reference_index is None:
        raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

    reference_target = pd.Series(0.0, index=reference_index, name="walkforward_target")
    reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)
    evaluator = _build_rule_based_walkforward_evaluator(combo_returns)
    return (
        reference_candles,
        reference_target,
        successful_param_grid,
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


def run_candle_shuffle_null(
    config: object,
    reference_candles: pd.DataFrame,
    reference_target: pd.Series,
    param_grid: list[dict],
    fold_rows: list[dict],
    nreps: int,
    random_seed: int | None,
    objective_metric_name: str = "sharpe",
) -> np.ndarray:
    """Run nreps walkforward with candles permuted in two units and re-extraction; return null distribution."""
    from utils.evaluation.permutation_test.candle_shuffle import permute_walk_forward

    metric_fn = resolve_objective_metric(objective_metric_name)
    portfolio_candles = load_candles_for_config(config)
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
    null_metrics = np.empty(nreps, dtype=float)

    for i in tqdm(range(nreps), desc="Candle shuffle", unit="rep"):
        seed = int(rng.integers(0, 2**31))
        shuffled_candles = permute_walk_forward(
            candles_prepared,
            train_windows=train_windows,
            random_seed=seed,
        )
        if feature_type == FeatureType.CONTINUOUS:
            combo_feature_target = {}
            successful_param_grid = []
            reference_index = None
            reference_target_series = None
            for single_spec in expanded:
                combo = single_spec["params"]
                data = load_features_for_combo(single_spec, config, candles_override=shuffled_candles)
                if data is None:
                    continue
                feature, target, _ = data
                paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
                if paired.empty:
                    continue
                feature = paired["feature"]
                target = paired["target"]
                expanded_combo_params = _expand_params_with_bin_count(
                    params=dict(combo),
                    bin_counts=config.binning_params.bin_counts,
                )
                nf = _normalize_series_datetime_index(feature)
                nt = _normalize_series_datetime_index(target)
                for combo_params in expanded_combo_params:
                    combo_feature_target[_combo_key(combo_params)] = pd.DataFrame({
                        "feature": nf,
                        "target": nt,
                    })
                    successful_param_grid.append(combo_params)
                if reference_index is None:
                    reference_index = _normalize_datetime_index(target.index)
                    reference_target_series = nt.reindex(reference_index)

            if not successful_param_grid or reference_index is None:
                null_metrics[i] = 0.0
                continue
            successful_param_grid = _expand_params_with_selected_bin(
                successful_param_grid,
                bin_index_min=config.binning_params.bin_index_min,
                bin_index_max=config.binning_params.bin_index_max,
            )
            ref_target = (
                reference_target_series.fillna(0.0).rename("walkforward_target")
                if reference_target_series is not None
                else pd.Series(0.0, index=reference_index, name="walkforward_target")
            )
            ref_candles = pd.DataFrame({"close": ref_target}, index=reference_index)
            evaluator = _build_continuous_walkforward_evaluator(combo_feature_target, config)
            portfolio_candles_df = load_candles_for_config(config)
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
                feature_data_by_combo=combo_feature_target,
                output_dir=None,
            )
        else:
            combo_returns = {}
            successful_param_grid = []
            reference_index = None
            for single_spec in expanded:
                combo = single_spec["params"]
                data = load_features_for_combo(single_spec, config, candles_override=shuffled_candles)
                if data is None:
                    continue
                feature, target, _ = data
                paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
                if paired.empty:
                    continue
                feature = paired["feature"]
                target = paired["target"]
                combo_returns[_combo_key(combo)] = _normalize_series_datetime_index(feature.mul(target))
                successful_param_grid.append(dict(combo))
                if reference_index is None:
                    reference_index = _unique_sorted_datetime_index(target.index)

            if not successful_param_grid or reference_index is None:
                null_metrics[i] = 0.0
                continue
            ref_target = pd.Series(0.0, index=reference_index, name="walkforward_target")
            ref_candles = pd.DataFrame({"close": ref_target}, index=reference_index)
            evaluator = _build_rule_based_walkforward_evaluator(combo_returns)
            report = run_walkforward_research(
                candles_df=ref_candles,
                target=ref_target,
                feature_type="rule_based",
                module_name=module_name,
                config=wf_config,
                param_grid=successful_param_grid,
                evaluate_param_combo=evaluator,
                research_config=config,
                output_dir=None,
            )
        null_metrics[i] = aggregate_oos_metric_from_report(
            report, treat_no_selection_as_zero=True, metric_fn=metric_fn
        )
    return null_metrics


def main() -> int:
    args = _parse_args()
    config = load_config()
    perm_cfg = getattr(config, "in_sample_permutation", None) or getattr(
        config, "permutation_suite", None
    )
    nreps = args.nreps if args.nreps is not None else (getattr(perm_cfg, "nreps", 100))
    random_seed = args.seed if args.seed is not None else getattr(perm_cfg, "random_seed", 42)
    alpha = getattr(perm_cfg, "alpha", 0.05)
    n_jobs = (
        args.n_jobs
        if args.n_jobs is not None
        else getattr(perm_cfg, "n_jobs_walkforward_reps", 1)
    )

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
        print("No walkforward folds; cannot run permutation.")
        return 1

    # Permutation config's objective is the source of truth (e.g. t_stat, sortino)
    _perm_obj = getattr(perm_cfg, "objective_metric", None)
    _builtin = getattr(_perm_obj, "builtin", None) if _perm_obj else None
    objective_metric_name = (
        str(_builtin)
        if _builtin is not None
        else (str(getattr(wf_config, "objective_metric_name", "sharpe")) if wf_config else "sharpe")
    )
    print(f"Objective metric (from config.permutation): {objective_metric_name}")
    if wf_config is not None:
        wf_config = replace(wf_config, objective_metric_name=objective_metric_name)
    config = replace(config, walkforward=wf_config)

    unit1_mask, unit2_mask = two_unit_masks_from_fold_rows(reference_target.index, fold_rows)
    module_name = str(getattr(config, "bias_spec", {}).get("module_name", "rsi"))
    feature_type = "continuous" if feature_data_by_combo is not None else "rule_based"

    print("Original (unpermuted) walkforward run...")
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

    fixed_oos_signal_by_fold: dict[int, pd.Series] | None = None
    if args.mode == "vector_shuffle" and feature_data_by_combo is not None:
        fixed_oos_signal_by_fold = _compute_fixed_oos_signal_by_fold(
            fold_rows=fold_rows,
            selection_summary_df=report0.selection_summary_df,
            reference_target=reference_target,
            research_config=config,
            feature_data_by_combo=feature_data_by_combo,
        )

    # When using fixed-signal null, original must use same construction (signal*target) for a valid test
    canonical_oos_index = None
    if fixed_oos_signal_by_fold is not None:
        agg_signal_target = aggregate_signal_target_returns(
            fixed_oos_signal_by_fold, reference_target, fold_rows
        )
        if agg_signal_target.empty:
            original_metric = 0.0
        else:
            used = agg_signal_target.dropna()
            canonical_oos_index = used.index
            val = metric_fn(used)
            original_metric = float(val) if np.isfinite(val) else 0.0
    else:
        original_metric = aggregate_oos_metric_from_report(
            report0, treat_no_selection_as_zero=True, metric_fn=metric_fn
        )
    print(f"  Original aggregate OOS metric ({objective_metric_name}): {original_metric:.4f}")
    if fixed_oos_signal_by_fold is not None:
        print("  (metric on full OOS period: signal×target with zeros when inactive)")

    if args.mode == "vector_shuffle":
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
        )
    else:
        null_metrics = run_candle_shuffle_null(
            config=config,
            reference_candles=reference_candles,
            reference_target=reference_target,
            param_grid=param_grid,
            fold_rows=fold_rows,
            nreps=nreps,
            random_seed=random_seed,
            objective_metric_name=objective_metric_name,
        )

    n_ge = int((null_metrics >= original_metric).sum())
    p_value = float(1 + n_ge) / float(nreps + 1)
    critical_value = float(np.percentile(null_metrics, (1.0 - alpha) * 100.0))
    passed = bool(original_metric > critical_value)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = resolve_walkforward_output_dir(
            feature_type=getattr(config, "feature_type", FeatureType.CONTINUOUS).value,
            module_name=module_name,
            root_dir=wf_config.output_root,
        ) / "permutation"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "mode": args.mode,
        "nreps": nreps,
        "random_seed": random_seed,
        "alpha": alpha,
        "original_metric": original_metric,
        "p_value": p_value,
        "critical_value": critical_value,
        "passed": passed,
    }
    report_path = out_dir / "walkforward_permutation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(out_dir / "null_distribution.npy", null_metrics)

    print(f"\nWalkforward permutation ({args.mode})")
    print(f"  nreps={nreps}  alpha={alpha}")
    print(f"  Original metric: {original_metric:.4f}  Critical: {critical_value:.4f}")
    print(f"  p-value: {p_value:.4f}  Passed: {passed}")
    print(f"  Report: {report_path}  Null dist: {out_dir / 'null_distribution.npy'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
