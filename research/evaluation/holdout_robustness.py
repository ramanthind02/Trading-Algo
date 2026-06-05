"""Returns-only holdout robustness suite (shared by feature and portfolio research)."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from quantfoundry_core.robustness import BootstrapCI, SharpeCI, sharpe_confidence_interval
from quantfoundry_core.robustness.validation import (
    RankCorrelationResult,
    ValidationRobustnessReport,
    assemble_validation_report,
    block_bootstrap_sharpe_ci,
    cusum_vs_is_params,
    equity_curve_confidence_bands,
    parameter_rank_correlation,
    rolling_sharpe_zscore,
    sharpe_comparison,
)


@dataclass(frozen=True)
class HoldoutRobustnessConfig:
    """Configuration for IS-vs-holdout robustness on daily return series."""

    sharpe_confidence: float = 0.95
    n_bootstrap: int = 1000
    random_seed: int | None = 42
    cusum_alpha: float = 0.05
    equity_band_fraction_limit: float = 0.20
    rolling_window: int = 60
    rolling_z_threshold: float = -1.5
    rolling_fraction_limit: float = 0.30
    rank_correlation_floor: float = 0.20
    periods_per_year: int = 252


def _per_period_sharpe(returns: pd.Series, *, periods_per_year: int) -> float:
    values = returns.to_numpy(dtype=np.float64)
    if values.shape[0] < 2:
        return 0.0
    std = float(values.std(ddof=1))
    if not math.isfinite(std) or std <= 0.0:
        return 0.0
    return float(values.mean() / std * math.sqrt(periods_per_year))


def _finite_stat_or_zero(value: float) -> float:
    return 0.0 if not math.isfinite(value) else value


def _resolve_rolling_window(requested_window: int, is_len: int, holdout_len: int) -> int:
    """Use the requested window when both series are long enough; otherwise shrink."""

    bounded = max(2, int(requested_window))
    cap = min(max(2, is_len), max(2, holdout_len))
    return min(bounded, cap)


def _skipped_rank_correlation() -> RankCorrelationResult:
    return RankCorrelationResult(
        spearman_rho=1.0,
        p_value=0.0,
        n_combinations=2,
        passed_floor=0.0,
        passed=True,
        interpretation="Rank correlation not applicable (single locked definition).",
    )


def run_holdout_robustness_pipeline(
    is_returns: pd.Series,
    holdout_returns: pd.Series,
    *,
    config: HoldoutRobustnessConfig,
    include_rank_correlation: bool = False,
    rank_is_metrics: Sequence[float] | None = None,
    rank_holdout_metrics: Sequence[float] | None = None,
    reference_sigma: float | None = None,
    reference_volatility_returns: pd.Series | None = None,
) -> ValidationRobustnessReport:
    """Run Sharpe, CUSUM, equity bands, rolling SR z, and optional rank correlation.

    When ``reference_sigma`` is set, CUSUM and equity bands use validation ``mu``
    from ``is_returns`` with that sigma. When ``reference_volatility_returns`` is set,
    rolling Sharpe z-scores use that longer series for the IS rolling distribution.
    """

    is_clean = is_returns.dropna().sort_index(kind="mergesort")
    holdout_clean = holdout_returns.dropna().sort_index(kind="mergesort")
    if is_clean.shape[0] < 2 or holdout_clean.shape[0] < 2:
        raise ValueError("IS and holdout return series each need at least 2 observations.")

    is_array = is_clean.to_numpy(dtype=np.float64)
    holdout_array = holdout_clean.to_numpy(dtype=np.float64)
    mu_is = float(is_array.mean())
    sigma_is = (
        float(reference_sigma)
        if reference_sigma is not None
        else float(is_array.std(ddof=1))
    )
    if not math.isfinite(sigma_is) or sigma_is <= 0.0:
        sigma_is = 1e-12

    rolling_is_array = (
        reference_volatility_returns.dropna().sort_index(kind="mergesort").to_numpy(dtype=np.float64)
        if reference_volatility_returns is not None
        else is_array
    )

    periods = config.periods_per_year
    sr_is = _per_period_sharpe(is_clean, periods_per_year=periods)
    sharpe_ci_is = sharpe_confidence_interval(
        sr_observed=sr_is,
        n_obs=int(is_clean.shape[0]),
        skewness=_finite_stat_or_zero(float(is_clean.skew())),
        excess_kurtosis=_finite_stat_or_zero(float(is_clean.kurt())),
        confidence=config.sharpe_confidence,
    )

    sr_holdout = _per_period_sharpe(holdout_clean, periods_per_year=periods)
    bootstrap_ci_holdout = block_bootstrap_sharpe_ci(
        holdout_array,
        n_bootstrap=config.n_bootstrap,
        ci_level=config.sharpe_confidence,
        seed=config.random_seed,
        periods_per_year=periods,
    )

    sharpe_result = sharpe_comparison(
        sr_is=sr_is,
        sharpe_ci_is=sharpe_ci_is,
        sr_val=sr_holdout,
        n_val=int(holdout_clean.shape[0]),
        skewness_val=_finite_stat_or_zero(float(holdout_clean.skew())),
        excess_kurtosis_val=_finite_stat_or_zero(float(holdout_clean.kurt())),
        bootstrap_ci_val=bootstrap_ci_holdout,
    )
    cusum_result = cusum_vs_is_params(
        holdout_array,
        mu_is,
        sigma_is,
        alpha=config.cusum_alpha,
    )
    bands_result = equity_curve_confidence_bands(
        holdout_array,
        mu_is,
        sigma_is,
        z_crit=1.96,
        fraction_limit=config.equity_band_fraction_limit,
    )
    rolling_window = _resolve_rolling_window(
        config.rolling_window,
        int(rolling_is_array.shape[0]),
        int(holdout_clean.shape[0]),
    )
    rolling_result = rolling_sharpe_zscore(
        rolling_is_array,
        holdout_array,
        window=rolling_window,
        z_threshold=config.rolling_z_threshold,
        fraction_limit=config.rolling_fraction_limit,
    )

    if include_rank_correlation:
        if rank_is_metrics is None or rank_holdout_metrics is None:
            raise ValueError(
                "rank_is_metrics and rank_holdout_metrics are required when "
                "include_rank_correlation=True."
            )
        is_metrics = np.asarray(rank_is_metrics, dtype=np.float64)
        holdout_metrics = np.asarray(rank_holdout_metrics, dtype=np.float64)
        if is_metrics.shape[0] < 3:
            raise ValueError(
                "Rank correlation requires at least three parameter combinations "
                "(Spearman p-value is not finite for n=2)."
            )
        rank_result = parameter_rank_correlation(
            is_metrics.tolist(),
            holdout_metrics.tolist(),
            passed_floor=config.rank_correlation_floor,
        )
    else:
        rank_result = _skipped_rank_correlation()

    return assemble_validation_report(
        sharpe_result,
        cusum_result,
        bands_result,
        rolling_result,
        rank_result,
    )
