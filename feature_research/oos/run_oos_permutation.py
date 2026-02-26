"""OOS permutation test (vector-shuffle or candle-shuffle, single fold from config.oos_window).

Reuses shared permutation null logic from permutation_core and shared data loading
from run_walkforward_permutation._load_research_data.

Usage:
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/oos/run_oos_permutation.py [--nreps 100] [--seed 42] [--n-jobs 1]

Config: uses load_config(); config.oos_window required. nreps, seed, alpha, n_jobs
from config.in_sample_permutation or config.permutation_suite.
When in_sample_permutation.run_stage1 is False, candle_shuffle runs instead of vector_shuffle.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

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
from feature_research.in_sample.data_loader import load_candles_for_config
from feature_research.walkforward.io import resolve_walkforward_output_dir
from feature_research.walkforward.permutation_core import run_return_shuffle_null
from feature_research.walkforward.metrics import resolve_objective_metric
from feature_research.walkforward.permutation_helpers import aggregate_oos_metric_from_report
from feature_research.walkforward.runner import (
    build_fold_rows_from_explicit_specs,
    run_walkforward_research,
)
from feature_research.walkforward.run_walkforward_permutation import (
    _load_research_data,
    run_candle_shuffle_null,
)


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
        else getattr(perm_cfg, "n_jobs_walkforward_reps", 1)
    )
    # When config says run_stage1=False, skip vector shuffle and run candle shuffle (stage 2) only.
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

    oos = config.oos_window
    data_start = min(config.start, oos.train_start)
    data_end = max(config.end, oos.test_end)
    config_oos = replace(config, start=data_start, end=data_end)

    (
        reference_candles,
        reference_target,
        param_grid,
        evaluator,
        research_config,
        feature_data_by_combo,
        portfolio_candles_df,
    ) = _load_research_data(config_oos)

    fold_rows = build_fold_rows_from_explicit_specs(
        reference_target.index,
        [(oos.train_start, oos.train_end, oos.test_start, oos.test_end)],
        min_fold_samples=config.walkforward.min_fold_samples,
    )
    if not fold_rows:
        print("Error: OOS fold has insufficient samples. Check oos_window dates and data range.")
        return 1

    wf_config = getattr(config, "walkforward", None)
    _perm_obj = getattr(perm_cfg, "objective_metric", None)
    _builtin = getattr(_perm_obj, "builtin", None) if _perm_obj else None
    objective_metric_name = (
        str(_builtin)
        if _builtin is not None
        else (str(getattr(wf_config, "objective_metric_name", "sharpe")) if wf_config else "sharpe")
    )
    module_name = str(getattr(config, "bias_spec", {}).get("module_name", "rsi"))
    _ft = getattr(config, "feature_type", None)
    feature_type = (
        _ft.value if isinstance(_ft, FeatureType) else
        ("continuous" if feature_data_by_combo is not None else "rule_based")
    )

    # Portfolio simulation requires full OHLCV candles; reference_candles has only close.
    if portfolio_candles_df is None:
        portfolio_candles_df = load_candles_for_config(config_oos)

    print("Original (unpermuted) OOS run...")
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
        fold_rows_override=fold_rows,
    )
    metric_fn = resolve_objective_metric(objective_metric_name)
    # Use same statistic as permutation replicates (aggregate or per-fold mean) for valid p-value.
    original_metric = aggregate_oos_metric_from_report(
        report0, treat_no_selection_as_zero=True, metric_fn=metric_fn
    )
    print(f"  Original aggregate OOS metric ({objective_metric_name}): {original_metric:.4f}")

    agg_returns = getattr(report0, "aggregate_oos_returns", None)
    if effective_mode == "vector_shuffle":
        if agg_returns is None or agg_returns.dropna().empty:
            print("Error: no aggregate OOS returns from walkforward run; cannot build null.")
            return 1
        print(f"Running vector shuffle null (nreps={nreps})...")
        null_metrics = run_return_shuffle_null(
            aggregate_oos_returns=agg_returns,
            nreps=nreps,
            random_seed=random_seed,
            objective_metric_name=objective_metric_name,
            n_jobs=n_jobs,
        )
    else:
        print(f"Running candle shuffle null (nreps={nreps})...")
        null_metrics = run_candle_shuffle_null(
            config=config_oos,
            reference_candles=reference_candles,
            reference_target=reference_target,
            param_grid=param_grid,
            fold_rows=fold_rows,
            nreps=nreps,
            random_seed=random_seed,
            objective_metric_name=objective_metric_name,
            n_jobs=n_jobs,
        )

    n_ge = int((null_metrics >= original_metric).sum())
    p_value = float(1 + n_ge) / float(nreps + 1)
    critical_value = float(np.percentile(null_metrics, (1.0 - alpha) * 100.0))
    passed = bool(original_metric > critical_value)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = (
            resolve_walkforward_output_dir(
                feature_type=getattr(config, "feature_type", FeatureType.CONTINUOUS).value,
                module_name=module_name,
                root_dir=wf_config.output_root,
                output_subdir="oos",
            )
            / "permutation"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "mode": effective_mode,
        "nreps": nreps,
        "random_seed": random_seed,
        "alpha": alpha,
        "original_metric": original_metric,
        "p_value": p_value,
        "critical_value": critical_value,
        "passed": passed,
    }
    report_path = out_dir / "oos_permutation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(out_dir / "null_distribution.npy", null_metrics)

    print(f"\nOOS permutation ({effective_mode})")
    print(f"  nreps={nreps}  alpha={alpha}")
    print(f"  Original metric: {original_metric:.4f}  Critical: {critical_value:.4f}")
    print(f"  p-value: {p_value:.4f}  Passed: {passed}")
    print(f"  Report: {report_path}  Null dist: {out_dir / 'null_distribution.npy'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
