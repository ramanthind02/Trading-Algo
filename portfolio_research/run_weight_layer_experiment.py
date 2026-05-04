"""Weight layer method experiment: compare 6 methods on combined val+test window.

Writes a CSV summary table to:
  portfolio_research/results/weight_layer_experiment/experiment_results.csv

Columns: method, cagr, vol, sharpe, sortino, max_drawdown

Usage
-----
    .\.venv\Scripts\python.exe portfolio_research/run_weight_layer_experiment.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path


def _prepend_repo_root_to_syspath() -> None:
    start = Path(__file__).resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise RuntimeError("Could not locate repository root.")


_prepend_repo_root_to_syspath()

from utils.repo_bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

import numpy as np
import pandas as pd

from portfolio_research.config import load_config
from portfolio_research.pipelines.portfolio_test import (
    _evaluate_phase,
    _group_ensembles_by_timeframe,
    run_portfolio_research_cache_preflight,
)
from ensemble.vault_manager import load_ensemble_from_vault
from ensemble.vault.hierarchy_spec import build_hierarchy_spec_for_ensemble_dirs

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _compute_metrics(returns: pd.Series, method: str) -> dict:
    clean = returns.dropna()
    ann_ret = float(clean.mean() * 252)
    ann_vol = float(clean.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0.0 else float("nan")
    downside_vol = float(clean[clean < 0].std() * np.sqrt(252))
    sortino = ann_ret / downside_vol if downside_vol > 0.0 else float("nan")

    cum = (1.0 + clean).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = float(drawdown.min())

    return {
        "method": method,
        "cagr": round(ann_ret, 4),
        "vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
    }


def _enable_cache(ensemble, use_cache: bool):
    ensemble.use_cache = use_cache
    if use_cache and hasattr(ensemble, "retry_on_cache_miss"):
        ensemble.retry_on_cache_miss = False
    for model in getattr(ensemble, "base_models", {}).values():
        setattr(model, "use_cache", use_cache)
    return ensemble


def run_experiment() -> None:
    base_config = load_config()

    hierarchy_spec = build_hierarchy_spec_for_ensemble_dirs(
        _REPO_ROOT,
        base_config.ensemble_dirs,
        strict_group=True,
        portfolio_ticker_names=frozenset(t.name for t in base_config.tickers),
    )

    methods: list[tuple[str, str, dict]] = [
        ("equal_signal",              "equal_signal",              {"fdm_max": 2.0}),
        ("inverse_avg_pairwise_corr", "inverse_avg_pairwise_corr", {"fdm_max": 2.0}),
        ("hierarchy_equal",           "hierarchy_equal",           {"fdm_max": 2.0, "hierarchy_spec": hierarchy_spec}),
        ("inverse_corr_hierarchy",    "inverse_corr_hierarchy",    {"fdm_max": 2.0, "hierarchy_spec": hierarchy_spec}),
        ("ledoit_wolf_min_corr",      "ledoit_wolf_min_corr",      {"fdm_max": 2.0}),
        ("risk_parity_corr",          "risk_parity_corr",          {"fdm_max": 2.0}),
        ("hierarchy_theme_inv_corr",  "hierarchy_theme_inv_corr",  {"fdm_max": 2.0, "hierarchy_spec": hierarchy_spec}),
        ("hierarchy_theme_ledoit",    "hierarchy_theme_ledoit",    {"fdm_max": 2.0, "hierarchy_spec": hierarchy_spec}),
        ("ledoit_wolf_hierarchy_within", "ledoit_wolf_hierarchy_within", {"fdm_max": 2.0, "hierarchy_spec": hierarchy_spec}),
    ]

    run_portfolio_research_cache_preflight(base_config)

    named_ensembles = [
        (
            name,
            _enable_cache(
                load_ensemble_from_vault(
                    path,
                    refit=True,
                    target_volatility=base_config.target_volatility,
                    exclude_feature_stems_by_ensemble=getattr(
                        base_config, "exclude_feature_stems_by_ensemble", None
                    ),
                ),
                True,
            ),
        )
        for name, path in base_config.ensemble_dirs.items()
    ]
    grouped_ensembles = _group_ensembles_by_timeframe(named_ensembles)
    unique_timeframes = sorted(grouped_ensembles.keys())

    train_start = pd.Timestamp(base_config.train_window.start)
    train_end   = pd.Timestamp(base_config.train_window.end)
    val_start   = pd.Timestamp(base_config.validation_window.start)
    val_end     = pd.Timestamp(base_config.validation_window.end)
    test_start  = pd.Timestamp(base_config.test_window.start)
    test_end    = pd.Timestamp(base_config.test_window.end)

    output_dir = Path(base_config.output_root) / "weight_layer_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for label, wl_method, wl_kwargs in methods:
        print(f"\n{'='*60}")
        print(f"Method: {label}")
        print(f"{'='*60}")

        method_config = dataclasses.replace(
            base_config,
            weight_layer_method=wl_method,
            weight_layer_kwargs=wl_kwargs,
            output_root=output_dir / label,
            export_per_timeframe_tearsheets=False,
            export_per_ensemble_tearsheets=False,
        )

        print("  Running Validation phase...")
        val_result, _ = _evaluate_phase(
            phase_title="Validation",
            output_dir_name="validation",
            fit_start=train_start,
            fit_end=train_end,
            test_start=val_start,
            test_end=val_end,
            config=method_config,
            grouped_ensembles=grouped_ensembles,
            unique_timeframes=unique_timeframes,
            emit_tearsheets=False,
            run_purpose="metrics_only",
        )

        print("  Running Test phase...")
        test_result, _ = _evaluate_phase(
            phase_title="Test",
            output_dir_name="test",
            fit_start=train_start,
            fit_end=val_end,
            test_start=test_start,
            test_end=test_end,
            config=method_config,
            grouped_ensembles=grouped_ensembles,
            unique_timeframes=unique_timeframes,
            emit_tearsheets=False,
            run_purpose="metrics_only",
        )

        combined_returns = pd.concat(
            [val_result.combined_strategy_returns, test_result.combined_strategy_returns]
        ).sort_index()
        all_rows.append(_compute_metrics(combined_returns, label))

    results_df = pd.DataFrame(all_rows)

    csv_path = output_dir / "experiment_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    run_experiment()
