"""Signed-signal feature EDA (T003): per-level stats and bootstrap CI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_selection.eda.eda_dataclasses import (
    BootstrapCI,
    BootstrapCIResults,
    LevelStats,
    PerLevelStats,
)

_VOL_THRESHOLD = 1e-10


def _validate_aligned_index(feature: pd.Series, target: pd.Series) -> None:
    """Raise ValueError when feature/target indices are not identical."""
    if not feature.index.equals(target.index):
        raise ValueError("feature and target index mismatch")


def _validate_feature_levels(feature: pd.Series, levels: list[int]) -> None:
    """Raise ValueError when feature contains values outside levels."""
    allowed = set(levels)
    observed = set(feature.dropna().unique())
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise ValueError(f"unexpected feature level values: {unexpected}")


def compute_per_level_stats(
    feature: pd.Series,
    target: pd.Series,
    levels: list[int] = [-1, 0, 1],
) -> PerLevelStats:
    """Compute per-level mean, volatility, Sharpe, and reliability metadata."""
    _validate_aligned_index(feature, target)
    _validate_feature_levels(feature, levels)
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()

    stats_by_level: dict[int, LevelStats] = {
        level: _compute_single_level_stats(level=level, returns=aligned.loc[aligned["f"] == level, "t"])
        for level in levels
    }
    return PerLevelStats(stats_by_level=stats_by_level)


def _compute_single_level_stats(level: int, returns: pd.Series) -> LevelStats:
    """Compute summary statistics for one level of returns."""
    sample_count = int(len(returns))
    mean_return = float(returns.mean()) if sample_count > 0 else float("nan")
    volatility = float(returns.std()) if sample_count > 0 else float("nan")
    sharpe = (
        float("nan")
        if np.isnan(volatility) or abs(volatility) <= _VOL_THRESHOLD
        else float(mean_return / volatility)
    )
    return LevelStats(
        level=level,
        mean_return=mean_return,
        volatility=volatility,
        sharpe=sharpe,
        adjusted_sharpe=sharpe,
        sample_count=sample_count,
        is_reliable=sample_count >= 10,
    )


def compute_bootstrap_ci(
    returns_by_level: dict[int, pd.Series],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCIResults:
    """Compute bootstrap confidence intervals for mean return by level."""
    if n_iterations < 100:
        raise ValueError("n_iterations must be >= 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must satisfy 0 < confidence < 1")

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0

    ci_by_level = {
        int(level): _bootstrap_for_level(
            level=int(level),
            returns=series,
            n_iterations=n_iterations,
            alpha=alpha,
            rng=rng,
        )
        for level, series in sorted(returns_by_level.items(), key=lambda item: item[0])
    }
    return BootstrapCIResults(ci_by_level=ci_by_level)


def _bootstrap_for_level(
    level: int,
    returns: pd.Series,
    n_iterations: int,
    alpha: float,
    rng: np.random.Generator,
) -> BootstrapCI:
    """Bootstrap mean-return CI for one level."""
    clean = returns.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        nan_distribution = np.full(n_iterations, np.nan, dtype=float)
        return BootstrapCI(
            level=level,
            mean_return=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            bootstrap_distribution=nan_distribution,
        )

    bootstrap_distribution = np.array(
        [
            float(rng.choice(clean, size=clean.size, replace=True).mean())
            for _ in range(n_iterations)
        ],
        dtype=float,
    )
    return BootstrapCI(
        level=level,
        mean_return=float(clean.mean()),
        ci_lower=float(np.quantile(bootstrap_distribution, alpha)),
        ci_upper=float(np.quantile(bootstrap_distribution, 1.0 - alpha)),
        bootstrap_distribution=bootstrap_distribution,
    )
