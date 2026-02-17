"""Rule-based feature EDA (T003): per-level stats, bootstrap CI, transitions, plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_selection.eda.eda_dataclasses import (
    BootstrapCI,
    BootstrapCIResults,
    LevelStats,
    PerLevelStats,
    RuleBasedEDAPlots,
    TransitionMatrix,
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


def compute_transition_matrix(
    feature: pd.Series,
    levels: list[int] = [-1, 0, 1],
) -> TransitionMatrix:
    """Compute level-to-level transition counts and row-normalized probabilities."""
    clean = feature.dropna()
    _validate_feature_levels(clean, levels)

    level_to_idx = {level: idx for idx, level in enumerate(levels)}
    n_levels = len(levels)
    counts = np.zeros((n_levels, n_levels), dtype=int)

    values = clean.to_numpy(dtype=float)
    for prev_value, next_value in zip(values[:-1], values[1:]):
        prev_level = int(prev_value)
        next_level = int(next_value)
        counts[level_to_idx[prev_level], level_to_idx[next_level]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    identity_rows = np.eye(n_levels, dtype=float)
    probs = np.vstack(
        [
            row.astype(float) / row_sum.item() if row_sum.item() > 0 else identity_rows[idx]
            for idx, (row, row_sum) in enumerate(zip(counts, row_sums))
        ]
    )
    return TransitionMatrix(transition_counts=counts, transition_probs=probs)


def create_rule_based_eda_plots(
    per_level_stats: PerLevelStats,
    bootstrap_ci: BootstrapCIResults,
) -> RuleBasedEDAPlots:
    """Create level bar plot and a transition heatmap placeholder figure."""
    levels = sorted(per_level_stats.stats_by_level.keys())
    means = [per_level_stats.stats_by_level[level].mean_return for level in levels]

    lower_errors = [
        _compute_lower_error(level=level, mean_value=mean_value, bootstrap_ci=bootstrap_ci)
        for level, mean_value in zip(levels, means)
    ]
    upper_errors = [
        _compute_upper_error(level=level, mean_value=mean_value, bootstrap_ci=bootstrap_ci)
        for level, mean_value in zip(levels, means)
    ]

    fig_level, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        levels,
        np.nan_to_num(np.array(means, dtype=float)),
        yerr=np.array([lower_errors, upper_errors], dtype=float),
        capsize=4,
        color="steelblue",
    )
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title("Mean return by discrete level")
    ax.set_xlabel("Level")
    ax.set_ylabel("Mean return")
    fig_level.tight_layout()
    plt.close(fig_level)

    fig_heatmap, ax = plt.subplots(figsize=(5, 4))
    placeholder = np.eye(len(levels), dtype=float)
    image = ax.imshow(placeholder, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Transition probabilities (placeholder)")
    ax.set_xticks(range(len(levels)))
    ax.set_yticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_yticklabels(levels)
    fig_heatmap.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig_heatmap.tight_layout()
    plt.close(fig_heatmap)

    return RuleBasedEDAPlots(
        level_plot_fig=fig_level,
        transition_heatmap_fig=fig_heatmap,
    )


def _compute_lower_error(
    level: int,
    mean_value: float,
    bootstrap_ci: BootstrapCIResults,
) -> float:
    """Return non-negative lower error bar size for a level."""
    ci = bootstrap_ci.ci_by_level.get(level)
    if ci is None or np.isnan(mean_value) or np.isnan(ci.ci_lower):
        return 0.0
    return float(max(0.0, mean_value - ci.ci_lower))


def _compute_upper_error(
    level: int,
    mean_value: float,
    bootstrap_ci: BootstrapCIResults,
) -> float:
    """Return non-negative upper error bar size for a level."""
    ci = bootstrap_ci.ci_by_level.get(level)
    if ci is None or np.isnan(mean_value) or np.isnan(ci.ci_upper):
        return 0.0
    return float(max(0.0, ci.ci_upper - mean_value))
