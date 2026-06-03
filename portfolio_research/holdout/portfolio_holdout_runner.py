"""Orchestrate portfolio-level holdout report generation and persistence."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from quantfoundry_core.robustness.portfolio_holdout import (
    PortfolioISParams,
    compute_portfolio_holdout_report,
)
from quantfoundry_core.robustness.validation import ValidationRobustnessReport

from portfolio_research.config import PortfolioResearchConfig
from portfolio_research.holdout.monitoring_policy import (
    MonitoringWindow,
    build_monitoring_status,
    build_reference_calibration,
    slice_trailing_window,
)
from portfolio_research.holdout.strategy_monitoring import _write_strategy_holdout_summary
from portfolio_research.shared.visualization_paths import holdout_portfolio_visualization_dir
from utils.evaluation.holdout_robustness import HoldoutRobustnessConfig, run_holdout_robustness_pipeline


def build_portfolio_is_params(
    research_strategy_returns: pd.DataFrame,
    research_portfolio_returns: pd.Series,
    *,
    strategy_weights: dict[str, float] | None = None,
) -> PortfolioISParams:
    """Derive IS/research calibration parameters for portfolio holdout tests."""

    portfolio = research_portfolio_returns.dropna()
    portfolio_array = portfolio.to_numpy(dtype=np.float64)
    mu = float(portfolio_array.mean())
    sigma = float(portfolio_array.std(ddof=1))
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-12
    corr = research_strategy_returns.corr().to_numpy(dtype=np.float64)
    names = tuple(str(name) for name in research_strategy_returns.columns)
    weights = strategy_weights or {name: 1.0 / len(names) for name in names}
    sharpes = {
        name: float(
            research_strategy_returns[name].mean()
            / max(research_strategy_returns[name].std(ddof=1), 1e-12)
            * math.sqrt(252)
        )
        for name in names
    }
    weight_vec = np.asarray([weights.get(name, 0.0) for name in names], dtype=np.float64)
    cov = research_strategy_returns.cov().to_numpy(dtype=np.float64) * 252.0
    expected_vol = float(math.sqrt(max(weight_vec @ cov @ weight_vec, 0.0)))
    sr = float(mu / sigma * math.sqrt(252)) if sigma > 0 else 0.0
    return PortfolioISParams(
        mu_portfolio=mu,
        sigma_portfolio=sigma,
        sr_portfolio=sr,
        expected_vol=expected_vol,
        correlation_matrix=corr,
        strategy_names=names,
        strategy_weights=weights,
        strategy_sharpes=sharpes,
    )


def _slice_portfolio_reference(
    series: pd.Series,
    config: PortfolioResearchConfig,
) -> pd.Series:
    start = pd.Timestamp(config.validation_window.start)
    end = pd.Timestamp(config.validation_window.end)
    clipped = series.dropna().sort_index()
    return clipped.loc[(clipped.index >= start) & (clipped.index <= end)]


def _slice_portfolio_holdout(
    series: pd.Series,
    config: PortfolioResearchConfig,
) -> pd.Series:
    start = pd.Timestamp(config.test_window.start)
    end = pd.Timestamp(config.test_window.end)
    clipped = series.dropna().sort_index()
    return clipped.loc[(clipped.index >= start) & (clipped.index <= end)]


def run_portfolio_holdout_report(
    config: PortfolioResearchConfig,
    *,
    research_strategy_returns: pd.DataFrame,
    holdout_strategy_returns: pd.DataFrame,
    research_portfolio_returns: pd.Series,
    holdout_portfolio_returns: pd.Series,
) -> tuple[object, dict[str, Path]]:
    """Run portfolio holdout analytics and write JSON + CSV artifacts."""

    is_params = build_portfolio_is_params(
        research_strategy_returns,
        research_portfolio_returns,
    )
    holdout = config.holdout_robustness
    report = compute_portfolio_holdout_report(
        is_params=is_params,
        research_strategy_returns=research_strategy_returns,
        holdout_strategy_returns=holdout_strategy_returns,
        holdout_portfolio_returns=holdout_portfolio_returns,
        research_portfolio_returns=research_portfolio_returns,
        n_bootstrap=holdout.n_bootstrap,
        random_seed=holdout.random_seed,
        sharpe_confidence=holdout.sharpe_confidence,
        cusum_alpha=holdout.cusum_alpha,
    )
    out_dir = holdout_portfolio_visualization_dir(config.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "portfolio_holdout_report.json"
    json_path.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")

    corr = report.correlation_realisation
    corr_names = corr.strategy_names
    pd.DataFrame(corr.C_is, index=corr_names, columns=corr_names).to_csv(
        out_dir / "correlation_matrix_research.csv"
    )
    pd.DataFrame(corr.C_holdout, index=corr_names, columns=corr_names).to_csv(
        out_dir / "correlation_matrix_holdout.csv"
    )

    contribution = report.contribution
    contribution_names = tuple(
        dict.fromkeys(
            [
                *corr_names,
                *contribution.strategy_contributions.keys(),
                *is_params.strategy_names,
            ]
        )
    )
    pd.DataFrame(
        [
            {
                "strategy": name,
                "realised_contribution": contribution.strategy_contributions.get(name, 0.0),
                "expected_contribution": contribution.expected_contributions.get(name, 0.0),
            }
            for name in contribution_names
        ]
    ).to_csv(out_dir / "strategy_contributions.csv", index=False)

    summary_path = out_dir / "portfolio_holdout_summary.md"
    summary_path.write_text(_markdown_portfolio_holdout_summary(report), encoding="utf-8")

    from portfolio_research.visualization.portfolio_holdout_reports import (
        write_portfolio_holdout_plots,
    )

    holdout_cfg = HoldoutRobustnessConfig(
        sharpe_confidence=holdout.sharpe_confidence,
        n_bootstrap=holdout.n_bootstrap,
        random_seed=holdout.random_seed,
        cusum_alpha=holdout.cusum_alpha,
        equity_band_fraction_limit=holdout.equity_band_fraction_limit,
        rolling_window=holdout.rolling_window,
        rolling_z_threshold=holdout.rolling_z_threshold,
        rolling_fraction_limit=holdout.rolling_fraction_limit,
        periods_per_year=holdout.periods_per_year,
    )
    train_start = pd.Timestamp(config.train_window.start)
    train_end = pd.Timestamp(config.train_window.end)
    val_start = pd.Timestamp(config.validation_window.start)
    val_end = pd.Timestamp(config.validation_window.end)
    train_returns = research_portfolio_returns.dropna().sort_index()
    train_returns = train_returns.loc[
        (train_returns.index >= train_start) & (train_returns.index <= train_end)
    ]
    validation_returns = _slice_portfolio_reference(research_portfolio_returns, config)
    full_holdout = _slice_portfolio_holdout(holdout_portfolio_returns, config)
    split_calibration = holdout.split_reference_calibration and train_returns.shape[0] >= 2
    calibration = build_reference_calibration(
        train_returns,
        validation_returns,
        regime_shift_threshold=holdout.reference_vol_regime_shift_threshold,
        recent_weight_on_shift=holdout.reference_vol_recent_weight_on_shift,
        split_calibration=split_calibration,
    )
    reference_volatility_returns = (
        pd.concat([train_returns, validation_returns]).sort_index()
        if split_calibration
        else validation_returns
    )
    holdout_end = pd.Timestamp(config.test_window.end)
    eval_returns = slice_trailing_window(
        full_holdout,
        end=holdout_end,
        months=holdout.evaluation_trailing_months,
    )
    monitoring_report = run_holdout_robustness_pipeline(
        validation_returns,
        eval_returns,
        config=holdout_cfg,
        include_rank_correlation=False,
        reference_sigma=calibration.sigma,
        reference_volatility_returns=reference_volatility_returns,
    )
    assert isinstance(monitoring_report, ValidationRobustnessReport)
    reference_window = MonitoringWindow(start=val_start, end=val_end)
    vol_window = MonitoringWindow(
        start=pd.Timestamp(reference_volatility_returns.index.min()),
        end=pd.Timestamp(reference_volatility_returns.index.max()),
    )
    eval_window = MonitoringWindow(
        start=pd.Timestamp(eval_returns.index.min()),
        end=pd.Timestamp(eval_returns.index.max()),
    )
    portfolio_override = (
        None
        if holdout.monitoring_weight_overrides is None
        else holdout.monitoring_weight_overrides.get("portfolio")
    )
    monitoring_status = build_monitoring_status(
        monitoring_report,
        evaluation_window=eval_window,
        reference_window=reference_window,
        override_weight_fraction=portfolio_override,
    )
    matplotlib_dir = out_dir / "matplotlib"
    monitoring_paths = _write_strategy_holdout_summary(
        monitoring_report,
        out_dir,
        strategy_name="portfolio",
        holdout_datetimes=[str(index) for index in eval_returns.index],
        monitoring_status=monitoring_status,
        reference_calibration=calibration,
        reference_volatility_window=vol_window,
        artifact_prefix="portfolio",
        plot_output_dir=matplotlib_dir,
    )

    plot_paths = write_portfolio_holdout_plots(
        report,
        output_dir=matplotlib_dir,
        correlation_research_csv=out_dir / "correlation_matrix_research.csv",
        correlation_holdout_csv=out_dir / "correlation_matrix_holdout.csv",
        contributions_csv=out_dir / "strategy_contributions.csv",
        holdout_portfolio_returns=holdout_portfolio_returns,
    )
    return report, {
        "json": json_path,
        "summary": summary_path,
        "plot_dir": out_dir / "matplotlib",
        **monitoring_paths,
        **{f"plot_{index}": path for index, path in enumerate(plot_paths)},
    }


def _markdown_portfolio_holdout_summary(report: object) -> str:
    from quantfoundry_core.robustness.portfolio_holdout import PortfolioHoldoutReport

    assert isinstance(report, PortfolioHoldoutReport)
    sharpe = report.portfolio_sharpe
    corr = report.correlation_realisation
    dd = report.drawdown_correlation
    contrib = report.contribution
    lines = [
        "# Portfolio Holdout Summary",
        "",
        f"**Overall:** {'PASS' if report.all_passed else 'ADVISORY'} — {report.interpretation}",
        "",
        "## Portfolio Sharpe",
        f"- Research Sharpe: {sharpe.sr_is:.4f}",
        f"- Holdout Sharpe: {sharpe.sr_val:.4f}",
        f"- Degradation ratio: {sharpe.degradation_ratio:.4f}",
        "",
        "## Correlation realisation",
        f"- Avg correlation shift: {corr.avg_corr_shift:+.3f} (alert: {corr.alert})",
        f"- Effective rank research/holdout: {corr.eff_rank_is:.2f} / {corr.eff_rank_holdout:.2f}",
        "",
        "## Drawdown correlation",
        f"- Max DD correlation shift: {dd.max_dd_corr_shift:.3f}",
        f"- Max overlap: {dd.max_overlap:.3f}",
        "",
        "## Vol calibration",
        f"- Vol ratio (realised/expected): {report.vol_ratio:.3f} (IDM alert: {report.idm_alert})",
        "",
        "## Contribution concentration",
        f"- HHI: {contrib.hhi:.3f} (alert > 0.40: {contrib.hhi_alert})",
        f"- Negative contributors: {contrib.negative_contributor_count}",
    ]
    return "\n".join(lines)
