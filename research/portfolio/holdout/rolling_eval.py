"""Two-fold holdout evaluation: train→validation, then rolled train→test."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from research.portfolio.config import (
    HoldoutFoldRole,
    HoldoutFoldSpec,
    PortfolioFitMode,
    PortfolioResearchConfig,
    build_holdout_fold_specs,
)
from research.portfolio.holdout.returns_export import write_return_matrices
from research.portfolio.pipelines.portfolio_test import (
    PhaseResult,
    _enable_cache,
    _group_ensembles_by_timeframe,
    _evaluate_phase,
    run_portfolio_research_cache_preflight,
)
from research.portfolio.shared.visualization_paths import holdout_root, holdout_test_dir
from research.portfolio.weight_layer_export import write_weight_layer_csv
from research.portfolio.weight_layer_report import write_portfolio_weight_layer_report
from ensemble.vault_manager import load_ensemble_from_vault


@dataclass(frozen=True)
class RollingHoldoutResult:
    """Holdout outputs from two-fold rolling or single-fit evaluation."""

    portfolio_returns_holdout: pd.Series
    portfolio_returns_research: pd.Series
    strategy_returns_holdout: pd.DataFrame
    strategy_returns_research: pd.DataFrame
    weight_layer_df: pd.DataFrame
    fold_manifest_path: Path


def _load_grouped_ensembles(config: PortfolioResearchConfig) -> dict[Any, list[Any]]:
    named = [
        (
            name,
            _enable_cache(
                load_ensemble_from_vault(
                    path,
                    refit=config.ensemble_vault_refit,
                    target_volatility=config.target_volatility,
                    exclude_feature_stems_by_ensemble=config.exclude_feature_stems_by_ensemble,
                ),
                config.use_cache,
            ),
        )
        for name, path in config.ensemble_dirs.items()
    ]
    return _group_ensembles_by_timeframe(named)


def _research_windows(config: PortfolioResearchConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(config.train_window.start),
        pd.Timestamp(config.validation_window.end),
    )


def run_single_holdout_evaluation(
    config: PortfolioResearchConfig,
    *,
    emit_tearsheets: bool = True,
) -> RollingHoldoutResult:
    """Fit once on train+validation and score the full test window."""

    run_portfolio_research_cache_preflight(config)
    grouped = _load_grouped_ensembles(config)
    unique_timeframes = sorted(grouped.keys())
    research_start, research_end = _research_windows(config)
    test_start = pd.Timestamp(config.test_window.start)
    test_end = pd.Timestamp(config.test_window.end)

    research_result, _ = _evaluate_phase(
        phase_title="Research",
        output_dir_name="research_snapshot",
        fit_start=pd.Timestamp(config.train_window.start),
        fit_end=research_end,
        test_start=research_start,
        test_end=research_end,
        config=config,
        grouped_ensembles=grouped,
        unique_timeframes=unique_timeframes,
        emit_tearsheets=False,
        run_purpose="metrics_only",
        collect_strategy_returns=True,
    )
    test_result, weight_layer_df = _evaluate_phase(
        phase_title="Holdout",
        output_dir_name="test",
        fit_start=pd.Timestamp(config.train_window.start),
        fit_end=research_end,
        test_start=test_start,
        test_end=test_end,
        config=config,
        grouped_ensembles=grouped,
        unique_timeframes=unique_timeframes,
        emit_tearsheets=emit_tearsheets,
        run_purpose="full",
        collect_strategy_returns=True,
    )
    holdout_dir = holdout_test_dir(config.output_root)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    if emit_tearsheets:
        portfolio_dir = holdout_dir / "portfolio"
        portfolio_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "fit_mode": PortfolioFitMode.SINGLE_FIT.name,
        "folds": [
            {
                "fold_id": 0,
                "fit_start": str(config.train_window.start),
                "fit_end": str(research_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
            }
        ],
    }
    manifest_path = holdout_root(config.output_root) / "fold_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_weight_layer_csv(
        weight_layer_df,
        holdout_dir / "weight_layer_weights_holdout.csv",
    )
    write_portfolio_weight_layer_report(
        weight_layer_df,
        holdout_dir / "weight_layer_report.json",
        config=config,
        phase="Holdout",
        fit_start=pd.Timestamp(config.train_window.start),
        fit_end=pd.Timestamp(config.validation_window.end),
        predict_start=pd.Timestamp(config.test_window.start),
        predict_end=pd.Timestamp(config.test_window.end),
    )
    return_paths = write_return_matrices(
        output_root=config.output_root,
        research_strategy_returns=research_result.strategy_returns_by_ensemble,
        holdout_strategy_returns=test_result.strategy_returns_by_ensemble,
        holdout_portfolio_returns=test_result.combined_strategy_returns,
        research_portfolio_returns=research_result.combined_strategy_returns,
    )
    _ = return_paths
    return RollingHoldoutResult(
        portfolio_returns_holdout=test_result.combined_strategy_returns,
        portfolio_returns_research=research_result.combined_strategy_returns,
        strategy_returns_holdout=test_result.strategy_returns_by_ensemble,
        strategy_returns_research=research_result.strategy_returns_by_ensemble,
        weight_layer_df=weight_layer_df,
        fold_manifest_path=manifest_path,
    )


def _fold_output_dir_name(spec: HoldoutFoldSpec) -> str:
    return spec.role.value


def run_rolling_holdout_evaluation(
    config: PortfolioResearchConfig,
    *,
    emit_tearsheets: bool = True,
    fold_specs: tuple[HoldoutFoldSpec, ...] | None = None,
) -> RollingHoldoutResult:
    """Run validation fold (train fit) and test holdout fold (rolled train fit)."""

    run_portfolio_research_cache_preflight(config)
    grouped = _load_grouped_ensembles(config)
    unique_timeframes = sorted(grouped.keys())
    specs = fold_specs if fold_specs is not None else build_holdout_fold_specs(config)

    holdout_portfolio: pd.Series | None = None
    holdout_strategy: pd.DataFrame | None = None
    holdout_baseline: pd.Series | None = None
    holdout_weight_df = pd.DataFrame()

    for spec in specs:
        phase_result, weight_df = _evaluate_phase(
            phase_title=f"HoldoutFold{spec.fold_id}_{spec.role.value}",
            output_dir_name=_fold_output_dir_name(spec),
            fit_start=pd.Timestamp(spec.fit_start),
            fit_end=pd.Timestamp(spec.fit_end),
            test_start=pd.Timestamp(spec.test_start),
            test_end=pd.Timestamp(spec.test_end),
            config=config,
            grouped_ensembles=grouped,
            unique_timeframes=unique_timeframes,
            emit_tearsheets=False,
            run_purpose="metrics_only",
            collect_strategy_returns=True,
        )
        if spec.role == HoldoutFoldRole.HOLDOUT_TEST:
            holdout_portfolio = phase_result.combined_strategy_returns
            holdout_strategy = phase_result.strategy_returns_by_ensemble
            holdout_baseline = phase_result.combined_baseline_returns
            holdout_weight_df = weight_df

    if holdout_portfolio is None or holdout_strategy is None:
        raise ValueError("Holdout fold specs must include one holdout_test fold.")

    portfolio_holdout = holdout_portfolio
    strategy_holdout = holdout_strategy
    weight_layer_df = holdout_weight_df

    research_start, research_end = _research_windows(config)
    research_result, _ = _evaluate_phase(
        phase_title="Research",
        output_dir_name="research_snapshot",
        fit_start=pd.Timestamp(config.train_window.start),
        fit_end=research_end,
        test_start=research_start,
        test_end=research_end,
        config=config,
        grouped_ensembles=grouped,
        unique_timeframes=unique_timeframes,
        emit_tearsheets=False,
        run_purpose="metrics_only",
        collect_strategy_returns=True,
    )

    holdout_dir = holdout_test_dir(config.output_root)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    write_weight_layer_csv(weight_layer_df, holdout_dir / "weight_layer_weights_holdout.csv")
    write_portfolio_weight_layer_report(
        weight_layer_df,
        holdout_dir / "weight_layer_report.json",
        config=config,
        phase="Holdout",
        fit_start=pd.Timestamp(config.train_window.start),
        fit_end=pd.Timestamp(config.validation_window.end),
        predict_start=pd.Timestamp(config.test_window.start),
        predict_end=pd.Timestamp(config.test_window.end),
    )

    manifest_path = holdout_root(config.output_root) / "fold_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fit_mode": PortfolioFitMode.ROLLING_HOLDOUT.name,
                "folds": [
                    {
                        **asdict(spec),
                        "role": spec.role.name,
                    }
                    for spec in specs
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if emit_tearsheets and not portfolio_holdout.empty:
        from lib.plotting.graphing.quantstats_reports import generate_tearsheet
        from research.portfolio.pipelines.portfolio_test import (
            _benchmark_tearsheet_title,
        )

        portfolio_dir = holdout_dir / "portfolio"
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        baseline_source = (
            holdout_baseline
            if holdout_baseline is not None and not holdout_baseline.empty
            else research_result.combined_baseline_returns
        )
        baseline = baseline_source.reindex(portfolio_holdout.index, method="ffill").fillna(0.0)
        generate_tearsheet(
            strategy_returns=portfolio_holdout,
            baseline_returns=baseline,
            feature_name="Portfolio Holdout (test)",
            output_file=str(portfolio_dir / "Portfolio_Holdout_test_tearsheet.html"),
            mode="html",
            benchmark_title=_benchmark_tearsheet_title(config),
        )

    write_return_matrices(
        output_root=config.output_root,
        research_strategy_returns=research_result.strategy_returns_by_ensemble,
        holdout_strategy_returns=strategy_holdout,
        holdout_portfolio_returns=portfolio_holdout,
        research_portfolio_returns=research_result.combined_strategy_returns,
    )

    return RollingHoldoutResult(
        portfolio_returns_holdout=portfolio_holdout,
        portfolio_returns_research=research_result.combined_strategy_returns,
        strategy_returns_holdout=strategy_holdout,
        strategy_returns_research=research_result.strategy_returns_by_ensemble,
        weight_layer_df=weight_layer_df,
        fold_manifest_path=manifest_path,
    )


def run_holdout_evaluation(
    config: PortfolioResearchConfig,
    *,
    emit_tearsheets: bool = True,
) -> RollingHoldoutResult:
    """Dispatch rolling or single-fit holdout evaluation from config."""

    if config.portfolio_fit_mode == PortfolioFitMode.SINGLE_FIT:
        return run_single_holdout_evaluation(config, emit_tearsheets=emit_tearsheets)
    return run_rolling_holdout_evaluation(config, emit_tearsheets=emit_tearsheets)
