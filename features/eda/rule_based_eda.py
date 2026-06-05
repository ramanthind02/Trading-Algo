"""Signed-signal feature EDA (T003): per-level stats and bootstrap CI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.eda.common_eda import align_feature_target, sharpe_from_mean_vol
from features.eda.eda_dataclasses import (
    BootstrapCI,
    BootstrapCIResults,
    LevelStats,
    PerLevelStats,
)


def _validate_aligned_index(feature: pd.Series, target: pd.Series) -> None:
    """Raise ValueError when feature/target indices are not identical."""
    if not feature.index.equals(target.index):
        raise ValueError("feature and target index mismatch")


def _validate_feature_levels(feature: pd.Series, levels: list[int]) -> None:
    """Raise ValueError when feature contains values outside levels."""
    allowed = set(levels)
    observed = {int(round(float(x))) for x in feature.dropna().unique()}
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise ValueError(f"unexpected feature level values: {unexpected}")


def infer_discrete_feature_levels(feature: pd.Series) -> list[int]:
    """Integer levels present in *feature* (NaNs dropped), sorted ascending.

    Used for stacked / multi-step discrete nodes (e.g. BasicBreakout, BasicMR) whose
    outputs are not confined to ``{-1, 0, 1}``.
    """
    return sorted({int(round(float(x))) for x in feature.dropna().unique()})


def compute_per_level_stats(
    feature: pd.Series,
    target: pd.Series,
    levels: list[int] | None = None,
) -> PerLevelStats:
    """Compute per-level mean, volatility, Sharpe, and reliability metadata.

    Uses **strategy** returns ``feature * target`` per row (same as in-sample / OOS
    ``signal * target``), not raw forward returns. Short legs (``feature == -1``)
    therefore show P&L consistent with cumulative equity, not inverted market drift.

    Parameters
    ----------
    levels
        Allowed discrete levels. If ``None``, levels are inferred from *feature*
        (see :func:`infer_discrete_feature_levels`). Pass ``[-1, 0, 1]`` explicitly
        when you require ternary-only validation.
    """
    _validate_aligned_index(feature, target)
    resolved = infer_discrete_feature_levels(feature) if levels is None else list(levels)
    _validate_feature_levels(feature, resolved)
    aligned = align_feature_target(feature, target)
    strategy_returns = aligned["f"].astype(float) * aligned["t"].astype(float)

    stats_by_level: dict[int, LevelStats] = {
        level: _compute_single_level_stats(
            level=level,
            returns=strategy_returns.loc[aligned["f"].astype(float) == float(level)],
        )
        for level in resolved
    }
    return PerLevelStats(stats_by_level=stats_by_level)


def _compute_single_level_stats(level: int, returns: pd.Series) -> LevelStats:
    """Compute summary statistics for one level of returns."""
    sample_count = int(len(returns))
    mean_return = float(returns.mean()) if sample_count > 0 else float("nan")
    volatility = float(returns.std()) if sample_count > 0 else float("nan")
    sharpe = sharpe_from_mean_vol(mean_return, volatility)
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
