"""Map feature-research config into shared holdout robustness settings."""
from __future__ import annotations

from research.feature.config import ResearchConfig, ValidationRobustnessConfig
from research.evaluation.holdout_robustness import HoldoutRobustnessConfig


def holdout_config_from_research(config: ResearchConfig) -> HoldoutRobustnessConfig:
    """Build shared holdout config from feature-research validation settings."""

    validation = config.validation_robustness
    return HoldoutRobustnessConfig(
        sharpe_confidence=config.robustness.sharpe_confidence,
        n_bootstrap=validation.n_bootstrap,
        random_seed=validation.random_seed,
        cusum_alpha=validation.cusum_alpha,
        equity_band_fraction_limit=validation.equity_band_fraction_limit,
        rolling_window=validation.rolling_window,
        rolling_z_threshold=validation.rolling_z_threshold,
        rolling_fraction_limit=validation.rolling_fraction_limit,
        rank_correlation_floor=validation.rank_correlation_floor,
        periods_per_year=1,
    )


def holdout_config_from_validation_robustness(
    validation: ValidationRobustnessConfig,
    *,
    sharpe_confidence: float,
) -> HoldoutRobustnessConfig:
    """Build shared holdout config from validation robustness fields only."""

    return HoldoutRobustnessConfig(
        sharpe_confidence=sharpe_confidence,
        n_bootstrap=validation.n_bootstrap,
        random_seed=validation.random_seed,
        cusum_alpha=validation.cusum_alpha,
        equity_band_fraction_limit=validation.equity_band_fraction_limit,
        rolling_window=validation.rolling_window,
        rolling_z_threshold=validation.rolling_z_threshold,
        rolling_fraction_limit=validation.rolling_fraction_limit,
        rank_correlation_floor=validation.rank_correlation_floor,
        periods_per_year=1,
    )
